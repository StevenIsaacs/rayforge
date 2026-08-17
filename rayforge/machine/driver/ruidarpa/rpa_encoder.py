"""
RPA Encoder - Produces rpascript output for Ruida Protocol Analyzer driver.

Rpascript is the native command format for RdDriver. The encoder drives
the ruida-pa GlueScript API (``rd_gluescript.GlueScript``), which owns
job framing, layer attribute blocks, per-layer action routing, and the
bounding-box math. Coordinates use mm natively (no unit conversion needed).
"""

from __future__ import annotations

import copy
import inspect
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from raygeo.geo.types import Point3D
from raygeo.ops import Ops
from raygeo.ops.state import AirAssistMode, CoolantMode
from raygeo.ops.types import CommandType, RasterMode, SectionType

from rayforge.pipeline.encoder.base import (
    EncodedOutput,
    MachineCodeOpMap,
    OpsEncoder,
)

try:
    from ruidadriver.rd_gluescript import GlueScript
except ImportError:
    GlueScript = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from rayforge.core.doc import Doc
    from rayforge.core.layer import Layer
    from rayforge.machine.models.machine import Machine

logger = logging.getLogger(__name__)

# Ruida controllers reject a layer minimum power below 8% (see GlueScript
# declare_layer) — any layer power below this is clamped up.
_MIN_LAYER_POWER_PERCENT = 8.0
_DEFAULT_LAYER_SPEED_MMS = 100.0
_DEFAULT_LAYER_FREQUENCY_KHZ = 20.0
_DEFAULT_LAYER_POWER = 0.2  # fraction, i.e. 20%
_DEFAULT_JOB_LABEL = "Rayforge Job"
_DEFAULT_LAYER_COLOR = "#00ccff"

# Maps the framework WCS slot names to the Ruida reference point strings
# accepted by GlueScript.declare_job. The framework default WCS ("G54")
# is deliberately absent — it maps to "MACHINE" to keep golden output
# byte-identical.
_WCS_TO_REF_POINT = {
    "MACHINE": "MACHINE",
    "ANCHOR": "ABSOLUTE",
    "CURRENT": "CURRENT",
    "SET_POINT": "SET_POINT",
}

# Last WCS that triggered the G54 fallback warning. Encoders are fresh
# per encode, so dedup state lives at module level; the warning fires
# only when the value changes (first G54, and after any real WCS).
# Worst case under concurrent encodes: a duplicated or suppressed
# warning — assignments are atomic under the GIL and output is
# unaffected.
_last_fallback_wcs: Optional[str] = None


class _RecordedCall:
    """Callable that records an invocation into the plan, then forwards.

    Keyword arguments are bound to the target's signature and converted
    to positional order, so the recorded plan stays a uniform list of
    ``(name, args)`` pairs replayable on an RPC GlueScript sink.
    """

    def __init__(
        self,
        name: str,
        target: Any,
        plan: List[Tuple[str, Tuple]],
    ) -> None:
        self._name = name
        self._target = target
        self._plan = plan

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs:
            bound = inspect.signature(self._target).bind(*args, **kwargs)
            bound.apply_defaults()
            args = tuple(bound.arguments.values())
        self._plan.append((self._name, copy.deepcopy(args)))
        return self._target(*args)


class _RecordingGlueScriptProxy:
    """GlueScript proxy that records the encoder's call plan.

    Every attribute access forwards to the wrapped GlueScript, so
    private state reads (``_job_header``, ``_layer_actions``, ...) keep
    resolving through ``__getattr__``. Callable attributes return a
    ``_RecordedCall`` that records the invocation before forwarding.
    Staging/finalize methods (``stage_gluescript`` and
    ``stage_gluescript_delta``) are forwarded without recording — they
    have no replay dispatch case on the RPC sink.
    """

    _UNRECORDED = frozenset({"stage_gluescript", "stage_gluescript_delta"})

    def __init__(
        self,
        gluescript: Any,
        plan: List[Tuple[str, Tuple]],
    ) -> None:
        self._wrapped = gluescript
        self._plan = plan

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._wrapped, name)
        if name in self._UNRECORDED or not callable(attr):
            return attr
        return _RecordedCall(name, attr, self._plan)


class RuidaRPAEncoder(OpsEncoder):
    """Encodes Ops commands into rpascript text via ruida-pa GlueScript.

    Each Ops command is translated into a GlueScript call so the staged
    rpascript stays controller-valid and the bounding boxes are computed
    by GlueScript from the actual cut extents.
    """

    def __init__(self) -> None:
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset all encoder state for a new encoding session."""
        self.current_pos: Point3D = (0.0, 0.0, 0.0)
        self.active_laser: int = 1
        self.doc: Optional["Doc"] = None
        self.machine: Optional["Machine"] = None
        self.op_map: Optional[MachineCodeOpMap] = None
        self._gluescript: Any = None
        self._layer_index_by_uid: Dict[str, int] = {}
        self._layer_key: int = 0
        self._layer: int = 0
        self._section_type: Optional[SectionType] = None
        self._section_raster_mode: Optional[RasterMode] = None
        self._layer_mode: str = "VECTOR"
        self._snapshot_key: int = 0
        self._header_len: int = 0
        self._actions_len: int = 0
        self._op_count: int = 0
        self._op_contributions: Dict[int, List[Tuple]] = {}
        self._gluescript_plan: List[Tuple[str, Tuple]] = []

    # -- Public API ---------------------------------------------------------

    def gluescript_plan(self) -> List[Tuple[str, Tuple]]:
        """Return a deep copy of the recorded GlueScript call plan.

        The plan holds the exact interleaved sequence of GlueScript
        method calls made while encoding, as ``(method_name, args)``
        pairs. It can be replayed on an RPC GlueScript sink to re-stage
        the job server-side without resending the assembled rpascript.
        """
        return copy.deepcopy(self._gluescript_plan)

    def encode(
        self, ops: Ops, machine: "Machine", doc: "Doc"
    ) -> EncodedOutput:
        """Encode Ops commands into rpascript text.

        Args:
            ops: Ops object from raygeo containing commands to encode.
            machine: The machine configuration (used for laser head lookup).
            doc: The document being processed.

        Returns:
            EncodedOutput with rpascript text, op_map, and no driver_data.

        Raises:
            RuntimeError: If the ruida-pa GlueScript API is unavailable or
                the job was not completed (missing JOB_END).
        """
        self._reset_state()
        if GlueScript is None:
            raise RuntimeError(
                "ruidadriver GlueScript is unavailable — install the "
                "ruida-pa package to use the ruidarpa driver"
            )
        if not hasattr(GlueScript, "stage_gluescript"):
            raise RuntimeError(
                "GlueScript.stage_gluescript() is missing — ruida-pa "
                ">= 0.15.2 is required to use the ruidarpa driver"
            )
        if ops.len() == 0:
            self.op_map = MachineCodeOpMap()
            return EncodedOutput(text="", op_map=self.op_map)

        self.doc = doc
        self.machine = machine
        self.op_map = MachineCodeOpMap()
        self._op_count = ops.len()
        self._layer_index_by_uid = {
            layer.uid: i for i, layer in enumerate(doc.layers)
        }
        gluescript_type: Any = GlueScript
        self._gluescript = _RecordingGlueScriptProxy(
            gluescript_type(), self._gluescript_plan
        )

        for i in range(ops.len()):
            self._snapshot_sections()
            self._handle_command(ops, i, machine)
            self._record_contribution(i)

        try:
            self._gluescript.stage_gluescript()
            lines = self._gluescript.rpascript
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to stage rpascript — the ops sequence must start "
                "with JOB_START and end with JOB_END"
            ) from exc

        self._build_op_map(len(lines))
        text = "\n".join(lines)
        return EncodedOutput(
            text=text,
            op_map=self.op_map,
            rpa_plan=self.gluescript_plan(),
        )

    # -- Command dispatch ---------------------------------------------------

    def _handle_command(self, ops: Ops, idx: int, machine: "Machine") -> None:
        """Dispatch a single Ops command to the appropriate handler."""
        ct = ops.command_type(idx)
        if ct == CommandType.SET_POWER:
            self._handle_set_power(ops, idx)
        elif ct == CommandType.SET_FEED_RATE:
            self._handle_set_cut_speed(ops, idx)
        elif ct == CommandType.SET_RAPID_RATE:
            self._handle_set_travel_speed(ops, idx)
        elif ct == CommandType.SET_FREQUENCY:
            self._handle_set_frequency(ops, idx)
        elif ct == CommandType.SET_PULSE_WIDTH:
            self._handle_set_pulse_width(ops, idx)
        elif ct == CommandType.SET_COOLANT:
            self._handle_coolant_as_air_assist(ops, idx)
        elif ct == CommandType.SET_HEAD:
            self._handle_set_laser(ops, idx, machine)
        elif ct == CommandType.MOVE_TO:
            self._handle_move_to(ops, idx)
        elif ct == CommandType.LINE_TO:
            self._handle_line_to(ops, idx)
        elif ct == CommandType.ARC_TO:
            self._handle_arc_to(ops, idx)
        elif ct == CommandType.SCAN_LINE:
            self._handle_scan_line(ops, idx)
        elif ct == CommandType.DWELL:
            self._handle_dwell(ops, idx)
        elif ct == CommandType.BEZIER_TO:
            self._handle_bezier_to(ops, idx)
        elif ct == CommandType.QUADRATIC_BEZIER_TO:
            self._handle_quadratic_bezier_to(ops, idx)
        elif ct == CommandType.JOB_START:
            self._handle_job_start()
        elif ct == CommandType.JOB_END:
            self._handle_job_end(idx)
        elif ct == CommandType.LAYER_START:
            self._handle_layer_start(ops, idx)
        elif ct == CommandType.LAYER_END:
            # GlueScript closes layers implicitly; an unclosed ops
            # section must not leak into the next layer. Reset the
            # layer mode so a stray section after LAYER_END fails
            # loudly instead of reusing the previous layer's mode.
            self._section_type = None
            self._section_raster_mode = None
            self._layer_mode = "VECTOR"
        elif ct == CommandType.WORKPIECE_START:
            self._handle_workpiece_start(ops, idx)
        elif ct == CommandType.WORKPIECE_END:
            self._handle_workpiece_end()
        elif ct == CommandType.OPS_SECTION_START:
            self._handle_ops_section_start(ops, idx)
        elif ct == CommandType.OPS_SECTION_END:
            self._handle_ops_section_end()
        elif ct == CommandType.SET_AIR_ASSIST:
            self._handle_set_air_assist(ops, idx)
        elif ct == CommandType.SET_SPINDLE_RPM:
            pass  # Ruida is laser-only; spindle not applicable
        elif ct == CommandType.SET_HEAD_COOLANT:
            pass  # Per-head coolant not yet supported
        elif ct == CommandType.STATE_BLOCK_START:
            pass  # Structural marker; no rpascript output
        elif ct == CommandType.STATE_BLOCK_END:
            pass  # Structural marker; no rpascript output
        else:
            raise ValueError(f"Unknown command type: {ct}")

    # -- Helpers ------------------------------------------------------------

    def _require_active_layer(self) -> None:
        """Fail fast when a layer-scoped op arrives before any LAYER_START."""
        if self._layer_key == 0:
            raise ValueError(
                "Layer-scoped op encountered before LAYER_START — "
                "GlueScript routing requires an active layer"
            )

    def _add_layer_action(self, lines: List[str]) -> None:
        """Route raw rpascript lines into the current layer's action block."""
        self._require_active_layer()
        self._gluescript.add_layer_action(self._layer_key, lines)

    def _clamp_power_pct(self, power_pct: float, source: str) -> float:
        """Clamp a power percent up to the controller minimum.

        Ruida controllers reject power below ``_MIN_LAYER_POWER_PERCENT``,
        so both layer attributes and per-op action lines clamp at the
        boundary, warning when the value is adjusted.

        Args:
            power_pct: Power percent to clamp.
            source: Description of the power source used in the warning.

        Returns:
            The clamped power percent.
        """
        if power_pct >= _MIN_LAYER_POWER_PERCENT:
            return power_pct
        logger.warning(
            "%s power %.1f%% is below the %d%% minimum — "
            "clamping min and max power to %d%%",
            source,
            power_pct,
            _MIN_LAYER_POWER_PERCENT,
            _MIN_LAYER_POWER_PERCENT,
        )
        return _MIN_LAYER_POWER_PERCENT

    def _emit_power(self, power_fraction: float) -> None:
        """Emit laser power for the current layer action block.

        Power-modulated greyscale and depth-map sections pass through
        GlueScript.power() unclamped; every other section uses
        power_range() so accel/decel over-burn on raster scans and
        line cuts is compensated.
        """
        section_type = self._section_type
        raster_mode = self._section_raster_mode
        if (
            section_type is None
            or raster_mode is None
            or section_type != SectionType.RASTER_FILL
            or raster_mode
            not in (RasterMode.VARIABLE_POWER, RasterMode.DEPTH_MAP)
        ):
            self._require_active_layer()
            power_pct = self._clamp_power_pct(power_fraction * 100.0, "Per-op")
            self._gluescript.power_range(power_pct, power_pct)
            return

        # Correct only because _compute_layer_mode derives IMAGE/DEPTHMAP
        # for image sections at LAYER_START.
        if self._layer_mode not in ("IMAGE", "DEPTHMAP"):
            raise ValueError(
                f"Image section {section_type.name} with raster mode "
                f"{raster_mode.name} requires an IMAGE/DEPTHMAP layer, "
                f"but the current layer mode is {self._layer_mode!r} — "
                f"missing LAYER_START before the section"
            )
        self._gluescript.power(power_fraction * 100.0)

    def _find_layer(self, layer_uid: str) -> Optional["Layer"]:
        """Look up a document layer by uid, or None when unknown."""
        if self.doc is None:
            return None
        return next(
            (layer for layer in self.doc.layers if layer.uid == layer_uid),
            None,
        )

    def _layer_settings(
        self, layer: Optional["Layer"]
    ) -> Tuple[float, float, float]:
        """Extract (speed_mms, frequency_khz, power_pct) for a layer.

        Reads the first workflow step, falling back to safe defaults. The
        power percent is clamped up to the controller minimum so the
        GlueScript power validation never rejects the job.
        """
        speed_mms = _DEFAULT_LAYER_SPEED_MMS
        power_fraction = _DEFAULT_LAYER_POWER
        frequency_hz = 0
        if (
            layer is not None
            and layer.workflow is not None
            and layer.workflow.steps
        ):
            first_step = layer.workflow.steps[0]
            # cut_speed is stored in mm/min; GlueScript expects mm/s.
            speed_mms = float(first_step.cut_speed) / 60.0
            power_fraction = float(first_step.power)
            frequency_hz = int(first_step.frequency)

        layer_label = (
            f"Layer {layer.name!r}" if layer is not None else "Layer ?"
        )
        power_pct = self._clamp_power_pct(power_fraction * 100.0, layer_label)

        frequency_khz = (
            frequency_hz / 1000.0
            if frequency_hz > 0
            else _DEFAULT_LAYER_FREQUENCY_KHZ
        )
        return speed_mms, frequency_khz, power_pct

    # -- Movement handlers --------------------------------------------------

    def _handle_move_to(self, ops: Ops, idx: int) -> None:
        """Rapid move (laser off) to an absolute position."""
        x, y, z = ops.endpoint(idx)
        self.current_pos = (x, y, z)
        self._require_active_layer()
        if z != 0:
            logger.warning(
                "Ignoring Z=%.3fmm on MOVE_TO — laser jobs are 2D "
                "and rpascript has no Z move for this driver",
                z,
            )
        self._gluescript.move_xy_to(x, y)

    def _handle_line_to(self, ops: Ops, idx: int) -> None:
        """Cutting move (laser on) to an absolute position."""
        x, y, z = ops.endpoint(idx)
        self.current_pos = (x, y, z)
        self._require_active_layer()
        if z != 0:
            logger.warning(
                "Ignoring Z=%.3fmm on LINE_TO — laser jobs are 2D "
                "and rpascript has no Z move for this driver",
                z,
            )
        self._gluescript.cut_xy_to(x, y)

    def _linearize_curve(self, ops: Ops, idx: int) -> None:
        """Linearize a curve op into cut and power actions.

        Rpascript has no native arc/bezier command, so curves are
        decomposed via ops.linearize() into cut segments and per-segment
        power adjustments.
        """
        self._require_active_layer()
        start_pos = self.current_pos
        end = ops.endpoint(idx)

        sub_ops = ops.linearize(idx, start_pos)
        for j in range(sub_ops.len()):
            sub_ct = sub_ops.command_type(j)
            if sub_ct == CommandType.LINE_TO:
                sx, sy, _ = sub_ops.endpoint(j)
                self._gluescript.cut_xy_to(sx, sy)
            elif sub_ct == CommandType.SET_POWER:
                self._emit_power(sub_ops.power(j))

        self.current_pos = end

    def _handle_arc_to(self, ops: Ops, idx: int) -> None:
        self._linearize_curve(ops, idx)

    def _handle_scan_line(self, ops: Ops, idx: int) -> None:
        self._linearize_curve(ops, idx)

    def _handle_bezier_to(self, ops: Ops, idx: int) -> None:
        self._linearize_curve(ops, idx)

    def _handle_quadratic_bezier_to(self, ops: Ops, idx: int) -> None:
        self._linearize_curve(ops, idx)

    def _handle_dwell(self, ops: Ops, idx: int) -> None:
        """Dwell is unsupported — the Ruida controller has no direct
        equivalent, so nothing is emitted."""
        logger.warning(
            "DWELL is not supported — the Ruida controller has no "
            "direct equivalent; no delay was emitted."
        )

    # -- Configuration handlers ---------------------------------------------

    def _handle_set_power(self, ops: Ops, idx: int) -> None:
        """Set laser power for the remaining cuts on this layer."""
        self._emit_power(ops.power(idx))

    def _handle_set_cut_speed(self, ops: Ops, idx: int) -> None:
        """Set cutting speed in mm/s."""
        self._require_active_layer()
        # ops.rate is in mm/min; GlueScript expects mm/s.
        self._gluescript.cut_speed(float(ops.rate(idx)) / 60.0)

    def _handle_set_travel_speed(self, ops: Ops, idx: int) -> None:
        """Set travel (rapid move) speed in mm/s."""
        self._require_active_layer()
        # ops.rate is in mm/min; GlueScript expects mm/s.
        self._gluescript.move_speed(float(ops.rate(idx)) / 60.0)

    def _handle_set_frequency(self, ops: Ops, idx: int) -> None:
        """Set laser frequency (Hz → KHz)."""
        freq_khz = ops.frequency(idx) / 1000.0
        self._require_active_layer()
        self._gluescript.frequency(freq_khz)

    def _handle_set_pulse_width(
        self,
        ops: Ops,
        idx: int,
    ) -> None:
        """Set laser pulse width in microseconds."""
        self._require_active_layer()
        self._gluescript.pwm(ops.pulse_width(idx))

    def _handle_coolant_as_air_assist(self, ops: Ops, idx: int) -> None:
        """Handle legacy SET_COOLANT used for air assist.

        ``ops.coolant()`` returns a ``CoolantMode`` enum (OFF/FLOOD/MIST),
        never the legacy ``"Air"`` string, so the comparison below never
        enables air assist. A non-OFF mode is logged as unsupported
        rather than silently dropped; use SET_AIR_ASSIST instead.
        """
        mode = ops.coolant(idx)
        if mode != CoolantMode.OFF:
            logger.warning(
                "SET_COOLANT %s is not acted upon — air assist "
                "requires SET_AIR_ASSIST",
                mode.name,
            )
        self._set_air_assist(mode == "Air")

    def _handle_set_air_assist(self, ops: Ops, idx: int) -> None:
        """Handle SET_AIR_ASSIST by reading AirAssistMode directly."""
        self._set_air_assist(ops.air_assist(idx) == AirAssistMode.ON)

    def _set_air_assist(self, enabled: bool) -> None:
        """Toggle air assist for the current layer."""
        self._require_active_layer()
        if enabled:
            self._gluescript.air_assist_on()
        else:
            self._gluescript.air_assist_off()

    def _handle_set_laser(
        self,
        ops: Ops,
        idx: int,
        machine: "Machine",
    ) -> None:
        """Select laser device by resolving laser_uid to a tool number.

        Tries machine.heads first, then falls back to a deterministic
        device number derived from the laser_uid suffix.
        """
        laser_uid = ops.head_uid(idx)
        try:
            device = ((int(laser_uid.split("_")[-1]) - 1) % 2) + 1
        except (ValueError, IndexError):
            device = (sum(ord(c) for c in laser_uid) % 2) + 1

        try:
            laser_head = next(
                (head for head in machine.heads if head.uid == laser_uid),
                None,
            )
            if laser_head is not None:
                device = laser_head.tool_number
        except (AttributeError, TypeError):
            logger.debug(
                "machine.heads not available, falling back to "
                "parsed laser_uid modulo for "
                "1-based laser device selection"
            )

        if device == self.active_laser:
            return
        self.active_laser = device
        self._gluescript.select_laser(device)

    # -- Structural handlers ------------------------------------------------

    def _handle_job_start(self) -> None:
        """Declare the job in GlueScript, which emits the job header."""
        global _last_fallback_wcs
        label = (
            self.doc.name
            if self.doc is not None and self.doc.name
            else _DEFAULT_JOB_LABEL
        )
        machine = self.machine
        if machine is None:
            ref_point = "MACHINE"
        else:
            wcs = machine.active_wcs
            if wcs in _WCS_TO_REF_POINT:
                _last_fallback_wcs = None
                ref_point = _WCS_TO_REF_POINT[wcs]
            elif wcs == "G54":
                if wcs != _last_fallback_wcs:
                    logger.warning(
                        "Active WCS %s is the framework default — "
                        "using the MACHINE reference point",
                        wcs,
                    )
                _last_fallback_wcs = wcs
                ref_point = "MACHINE"
            else:
                raise ValueError(
                    f"Unsupported WCS for Ruida reference point: {wcs!r} "
                    f"— valid names: {', '.join(_WCS_TO_REF_POINT)}"
                )
        self._gluescript.declare_job(label, ref_point, None, 1, 1, 0.0, 0.0)

    def _handle_layer_start(self, ops: Ops, idx: int) -> None:
        """Declare the layer with settings from its first workflow step."""
        layer_uid = ops.layer_uid(idx)
        layer = self._find_layer(layer_uid)
        self._layer = self._layer_index_by_uid.get(layer_uid, 0)
        self._layer_key += 1
        layer_key = self._layer_key

        speed_mms, frequency_khz, power_pct = self._layer_settings(layer)
        layer_mode = self._compute_layer_mode(ops, idx)
        self._layer_mode = layer_mode
        self._gluescript.declare_layer(
            label=(
                layer.name if layer is not None else f"Layer {layer_key - 1}"
            ),
            color=(layer.color if layer is not None else _DEFAULT_LAYER_COLOR),
            mode=layer_mode,
            overscan="NONE",
            speed=speed_mms,
            frequency=frequency_khz,
            min_power_1=power_pct,
            max_power_1=power_pct,
        )
        self._op_contributions.setdefault(idx, []).append(("attrs", layer_key))

    def _compute_layer_mode(self, ops: Ops, idx: int) -> str:
        """Derive the layer mode from its ops sections.

        Scans forward from the LAYER_START command to the next layer or
        job boundary. DEPTH_MAP beats VARIABLE_POWER regardless of
        section order, so the first DEPTH_MAP section yields "DEPTHMAP",
        any VARIABLE_POWER section yields "IMAGE", and anything else
        defaults to "VECTOR".
        """
        seen_variable_power = False
        for i in range(idx + 1, ops.len()):
            command = ops.command_type(i)
            if command in (
                CommandType.LAYER_END,
                CommandType.LAYER_START,
                CommandType.JOB_END,
            ):
                break
            if command != CommandType.OPS_SECTION_START:
                continue
            _, _, raster_mode = ops.section_params(i)
            if raster_mode == RasterMode.DEPTH_MAP:
                return "DEPTHMAP"
            if raster_mode == RasterMode.VARIABLE_POWER:
                seen_variable_power = True
        return "IMAGE" if seen_variable_power else "VECTOR"

    def _handle_job_end(self, idx: int) -> None:
        """Finalize the job in GlueScript, which emits END_JOB and EOF."""
        self._gluescript.end_job()
        self._op_contributions.setdefault(idx, []).append(("tail",))

    def _handle_workpiece_start(self, ops: Ops, idx: int) -> None:
        """Emit a workpiece start marker comment."""
        wp_uid = ops.workpiece_uid(idx)
        self._gluescript.comment([f"# Workpiece Start uid={wp_uid}"])

    def _handle_workpiece_end(self) -> None:
        """Emit a workpiece end marker comment."""
        self._gluescript.comment(["# Workpiece End"])

    def _handle_ops_section_start(self, ops: Ops, idx: int) -> None:
        """Record the active section and emit a comment for its start."""
        self._section_type, _workpiece_uid, self._section_raster_mode = (
            ops.section_params(idx)
        )
        self._gluescript.comment(
            [
                "# Ops Actions",
                "# Ops Section Start",
            ]
        )

    def _handle_ops_section_end(self) -> None:
        """Clear the active section and emit a comment for its end."""
        self._section_type = None
        self._section_raster_mode = None
        self._gluescript.comment(["# Ops Section End"])

    # -- Op map bookkeeping --------------------------------------------------

    def _snapshot_sections(self) -> None:
        """Record section lengths before dispatching the current op."""
        gs = self._gluescript
        self._snapshot_key = self._layer_key
        self._header_len = len(gs._job_header)
        self._actions_len = len(gs._layer_actions.get(self._snapshot_key, []))

    def _record_contribution(self, op_index: int) -> None:
        """Record which output sections the last op contributed to."""
        gs = self._gluescript
        contributions: List[Tuple] = []

        header_delta = len(gs._job_header) - self._header_len
        if header_delta > 0:
            contributions.append(("header", self._header_len, header_delta))

        actions_delta = (
            len(gs._layer_actions.get(self._snapshot_key, []))
            - self._actions_len
        )
        if actions_delta > 0:
            contributions.append(
                (
                    "actions",
                    self._snapshot_key,
                    self._actions_len,
                    actions_delta,
                )
            )

        if contributions:
            self._op_contributions[op_index] = contributions

    def _build_op_map(self, line_count: int) -> None:
        """Populate the op_map from the staged output layout.

        GlueScript assembles the final rpascript as: job header, all layer
        attribute blocks (sorted), LAST_LAYER, per-layer action blocks with
        SELECT_LAYER prefixes (sorted), then END_JOB/EOF. The recorded
        per-op contributions map onto that fixed layout exactly.
        """
        gs = self._gluescript
        header_len = len(gs._job_header) + len(gs._inline_prelude)

        attr_keys = sorted(gs._layer_attributes)
        attrs_start: Dict[int, int] = {}
        offset = header_len
        for key in attr_keys:
            attrs_start[key] = offset
            offset += len(gs._layer_attributes[key])

        last_layer_pos: Optional[int] = None
        if attr_keys:
            last_layer_pos = offset
            offset += 1

        action_keys = sorted(gs._layer_actions)
        actions_start: Dict[int, int] = {}
        for key in action_keys:
            actions_start[key] = offset + 1  # after the SELECT_LAYER line
            offset += 1 + len(gs._layer_actions[key])

        # Layout invariant: GlueScript assembles the rpascript as job
        # header (+ inline prelude), sorted layer attribute blocks,
        # LAST_LAYER, sorted per-layer action blocks (SELECT_LAYER +
        # actions), then END_JOB and EOF. declare_layer/end_job write
        # only to header/attrs, so the inline epilogue must stay empty
        # and the tail is pinned to [offset, offset + 1] (END_JOB, EOF).
        # If the epilogue ever becomes populated, tail lines shift and
        # the op_map silently mis-maps them — update the layout-pin
        # tests alongside any upstream GlueScript change.
        tail_positions: List[int] = []
        if last_layer_pos is not None:
            tail_positions.append(last_layer_pos)
        for key in action_keys:
            tail_positions.append(actions_start[key] - 1)  # SELECT_LAYER
        tail_positions.extend([offset, offset + 1])  # END_JOB, EOF

        for op_index in range(self._op_count):
            block = []
            contributions = self._op_contributions.get(op_index, [])
            for contribution in contributions:
                kind = contribution[0]
                if kind == "header":
                    _, index_in_header, count = contribution
                    block.extend(
                        range(index_in_header, index_in_header + count)
                    )
                elif kind == "attrs":
                    _, key = contribution
                    start = attrs_start[key]
                    block.extend(
                        range(start, start + len(gs._layer_attributes[key]))
                    )
                elif kind == "actions":
                    _, key, index_in_actions, count = contribution
                    start = actions_start[key] + index_in_actions
                    block.extend(range(start, start + count))
                elif kind == "tail":
                    block.extend(tail_positions)
            block.sort()
            assert self.op_map is not None
            self.op_map.op_to_machine_code[op_index] = block
            for line_num in block:
                self.op_map.machine_code_to_op[line_num] = op_index

        assert self.op_map is not None
        for op_index in self.op_map.op_to_machine_code:
            for line_num in self.op_map.op_to_machine_code[op_index]:
                assert 0 <= line_num < line_count
