"""
RPA Encoder - Produces rpascript output for Ruida Protocol Analyzer driver.

Rpascript is the native command format for RdDriver. The encoder drives
the ruida-pa GlueScript API (``rd_gluescript.GlueScript``), which owns
job framing, layer attribute blocks, per-layer action routing, and the
bounding-box math. Coordinates use mm natively (no unit conversion needed).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from raygeo.geo.types import Point3D
from raygeo.ops import Ops
from raygeo.ops.state import AirAssistMode, CoolantMode
from raygeo.ops.types import CommandType

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
        self._snapshot_key: int = 0
        self._header_len: int = 0
        self._actions_len: int = 0
        self._op_count: int = 0
        self._op_contributions: Dict[int, List[Tuple]] = {}

    # -- Public API ---------------------------------------------------------

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
        self._gluescript = gluescript_type()

        for i in range(ops.len()):
            self._snapshot_sections()
            self._handle_command(ops, i, machine)
            self._record_contribution(i)

        try:
            lines = self._gluescript.stage_rpascript()
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to stage rpascript — the ops sequence must start "
                "with JOB_START and end with JOB_END"
            ) from exc

        self._build_op_map(len(lines))
        text = "\n".join(lines)
        return EncodedOutput(text=text, op_map=self.op_map)

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
            pass  # GlueScript closes layers implicitly
        elif ct == CommandType.WORKPIECE_START:
            self._handle_workpiece_start(ops, idx)
        elif ct == CommandType.WORKPIECE_END:
            self._handle_workpiece_end()
        elif ct == CommandType.OPS_SECTION_START:
            self._handle_ops_section_start()
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
            source, power_pct, _MIN_LAYER_POWER_PERCENT,
            _MIN_LAYER_POWER_PERCENT,
        )
        return _MIN_LAYER_POWER_PERCENT

    def _emit_power_lines(self, power_fraction: float) -> None:
        """Emit MIN/MAX power lines for the current layer action block."""
        power_pct = self._clamp_power_pct(
            power_fraction * 100.0, "Per-op"
        )
        self._add_layer_action(
            [
                f"MIN_POWER_1 Power={power_pct:.1f}%",
                f"MAX_POWER_1 Power={power_pct:.1f}%",
            ]
        )

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
            speed_mms = float(first_step.cut_speed)
            power_fraction = float(first_step.power)
            frequency_hz = int(first_step.frequency)

        layer_label = (
            f"Layer {layer.name!r}"
            if layer is not None
            else "Layer ?"
        )
        power_pct = self._clamp_power_pct(
            power_fraction * 100.0, layer_label
        )

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
                self._emit_power_lines(sub_ops.power(j))

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
        """Emit a dwell (pause) command; DELAY accepts milliseconds."""
        duration_ms = ops.dwell_duration(idx)
        self._add_layer_action([f"DELAY {duration_ms:.3f}ms"])

    # -- Configuration handlers ---------------------------------------------

    def _handle_set_power(self, ops: Ops, idx: int) -> None:
        """Set laser power for the remaining cuts on this layer."""
        self._emit_power_lines(ops.power(idx))

    def _handle_set_cut_speed(self, ops: Ops, idx: int) -> None:
        """Set cutting speed in mm/s."""
        speed = float(ops.rate(idx))
        self._add_layer_action([f"SPEED_LASER_1 Speed={speed:.3f}mm/S"])

    def _handle_set_travel_speed(self, ops: Ops, idx: int) -> None:
        """Set travel (rapid move) speed in mm/s."""
        speed = float(ops.rate(idx))
        self._add_layer_action([f"SPEED_AXIS Speed={speed:.3f}mm/S"])

    def _handle_set_frequency(self, ops: Ops, idx: int) -> None:
        """Set laser frequency (Hz → KHz)."""
        freq_hz = ops.frequency(idx)
        freq_khz = freq_hz / 1000.0
        self._add_layer_action(
            [
                f"LAYER_FREQUENCY Laser={self.active_laser}"
                f" Layer={self._layer} Freq={freq_khz:.3f}KHz"
            ]
        )

    def _handle_set_pulse_width(
        self,
        ops: Ops,
        idx: int,
    ) -> None:
        """Set laser pulse width (µs → mS)."""
        pw_us = ops.pulse_width(idx)
        pw_ms = pw_us / 1000.0
        self._add_layer_action([f"LASER_INTERVAL {pw_ms:.3f}mS"])

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
            device = int(laser_uid.split("_")[-1]) % 2
        except (ValueError, IndexError):
            device = sum(ord(c) for c in laser_uid) % 2

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
                "parsed laser_uid %% 2 for "
                "laser device selection"
            )

        if device == self.active_laser:
            return
        self.active_laser = device
        self._add_layer_action([f"LASER_DEVICE_{device}"])

    # -- Structural handlers ------------------------------------------------

    def _handle_job_start(self) -> None:
        """Declare the job in GlueScript, which emits the job header."""
        label = (
            self.doc.name
            if self.doc is not None and self.doc.name
            else _DEFAULT_JOB_LABEL
        )
        self._gluescript.declare_job(label, "MACHINE", None, 1, 1, 0.0, 0.0)

    def _handle_layer_start(self, ops: Ops, idx: int) -> None:
        """Declare the layer with settings from its first workflow step."""
        layer_uid = ops.layer_uid(idx)
        layer = self._find_layer(layer_uid)
        self._layer = self._layer_index_by_uid.get(layer_uid, 0)
        self._layer_key += 1
        layer_key = self._layer_key

        speed_mms, frequency_khz, power_pct = self._layer_settings(layer)
        self._gluescript.declare_layer(
            label=(
                layer.name if layer is not None else f"Layer {layer_key - 1}"
            ),
            color=(layer.color if layer is not None else _DEFAULT_LAYER_COLOR),
            mode="VECTOR",
            overscan="NONE",
            speed=speed_mms,
            frequency=frequency_khz,
            min_power_1=power_pct,
            max_power_1=power_pct,
        )
        self._op_contributions.setdefault(idx, []).append(("attrs", layer_key))

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

    def _handle_ops_section_start(self) -> None:
        """Emit a comment for the ops section start."""
        self._gluescript.comment(
            [
                "# Ops Actions",
                "# Ops Section Start",
            ]
        )

    def _handle_ops_section_end(self) -> None:
        """Emit a comment for the ops section end."""
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
