"""
RPA Adapter — Main driver class for Ruida laser controllers via the
Ruida Protocol Analyzer (RPA) library.

Two modes:
- Direct mode: wraps ``RpaDirectDriver`` (in-process ``RdDriver``)
- TUI RPC mode: wraps ``RpaRpcClient`` (remote RPyC service)
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import random
from dataclasses import replace
from gettext import gettext as _
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

from rayforge.context import RayforgeContext
from rayforge.core.capability import PWMCapability
from rayforge.core.varset import BoolVar, HostnameVar, Var, VarSet
from rayforge.machine.driver.driver import (
    Axis,
    DeviceStatus,
    Driver,
    DriverMaturity,
    DriverPrecheckError,
    DriverSetupError,
    Pos,
)
from rayforge.machine.driver.ruidarpa.rpa_direct_driver import RpaDirectDriver
from rayforge.machine.driver.ruidarpa.rpa_rpc_client import RpaRpcClient
from rayforge.machine.transport import TransportStatus

try:
    from ruidadriver.rd_status import RdStatusEvent
except ImportError:
    RdStatusEvent = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from raygeo.ops import Ops

    from rayforge.core.doc import Doc
    from rayforge.machine.models.laser import Laser
    from rayforge.machine.models.machine import Machine
    from rayforge.pipeline.encoder.base import EncodedOutput, OpsEncoder

logger = logging.getLogger(__name__)

# Type alias for the two possible backends
_RpaBackend = Union[RpaDirectDriver, RpaRpcClient]


def _unwrap_um(value: object) -> Optional[int]:
    """Extract integer µm from a MEM_CURRENT_POSITION_* field.

    The RPA TUI service sends these as ``(int_um, str_description)``
    tuples. Accept both forms for forward compatibility.
    """
    if isinstance(value, (list, tuple)):
        return value[0]
    return value  # type: ignore[return-value]


class RuidaRPAAdapter(Driver):
    """
    Main driver class for connecting to Ruida laser controllers via the
    Ruida Protocol Analyzer (RPA) library.

    Supports two connection modes:
    * **Direct mode** — wraps ``RdDriver`` from ``ruida-protocol-analyzer``
      in-process over USB or UDP.
    * **TUI RPC mode** — connects to a remote RPyC service running the
      RPA TUI adapter.
    """

    label = _("Ruida RPA")
    subtitle = _("Connect via Ruida Protocol Analyzer")
    supports_settings = False
    reports_granular_progress = False
    uses_gcode = False
    maturity = DriverMaturity.KNOWN_BUGGY
    supports_probing = False
    native_overscan = True

    # --- Reconnect constants ---
    CONNECTION_POLL_INTERVAL = 0.5
    RECONNECT_BASE_DELAY = 1.0
    RECONNECT_MAX_DELAY = 30.0
    RECONNECT_JITTER = 0.2  # ±20%

    def __init__(self, context: RayforgeContext, machine: Machine) -> None:
        super().__init__(context, machine)
        self._config: Dict[str, Any] = {}
        self._tui_mode: bool = False
        self._backend: Optional[_RpaBackend] = None
        self._connection_task: Optional[asyncio.Task] = None
        self._keep_running: bool = False
        self._is_connected: bool = False
        self._shutting_down: bool = False

    # --- Properties ---

    @property
    def machine_space_wcs(self) -> str:
        return "MACHINE"

    @property
    def machine_space_wcs_display_name(self) -> str:
        return _("Ruida Coordinates")

    @property
    def supported_wcs(self) -> List[str]:
        if not self._is_connected:
            return ["MACHINE"]
        return ["MACHINE", "REF0", "REF1"]

    @property
    def resource_uri(self) -> Optional[str]:
        host = self._config.get("udp_host", "")
        usb = self._config.get("usb_device", "")
        if host:
            return f"ruidarpa://{host}"
        if usb:
            return f"ruidarpa://{usb}"
        return None

    # --- Classmethods ---

    @classmethod
    def precheck(cls, **kwargs: Any) -> None:
        udp_host = kwargs.get("udp_host", "")
        usb_device = kwargs.get("usb_device", "")
        if not udp_host and not usb_device:
            raise DriverPrecheckError(
                _(
                    "At least one of 'Hostname' or 'USB device' "
                    "must be configured."
                )
            )

    @classmethod
    def get_setup_vars(cls) -> VarSet:
        return VarSet(
            vars=[
                HostnameVar(
                    key="udp_host",
                    label=_("Hostname"),
                    description=_(
                        "The IP address or hostname of the Ruida controller"
                    ),
                ),
                Var(
                    key="usb_device",
                    label=_("USB"),
                    var_type=str,
                    description=_(
                        "USB device path "
                        "(e.g., /dev/ttyUSB0, "
                        "/dev/serial/by-id/usb-Ruida..., "
                        "or COM3)"
                    ),
                ),
                BoolVar(
                    key="tui",
                    label=_("TUI RPC"),
                    description=_(
                        "Enable TUI RPC connection to a remote "
                        "RPA TUI service"
                    ),
                    default=False,
                ),
            ]
        )

    @classmethod
    def create_encoder(cls, machine: Machine) -> "OpsEncoder":
        from rayforge.machine.driver.ruidarpa.rpa_encoder import (
            RuidaRPAEncoder,
        )

        return RuidaRPAEncoder()

    # --- Setup / Connect ---

    def _setup_implementation(self, **kwargs: Any) -> None:
        self._config = dict(kwargs)
        self._tui_mode = bool(kwargs.get("tui", False))

        if self._tui_mode:
            self._backend = RpaRpcClient()
            logger.debug(
                "RPA adapter configured for TUI RPC mode",
                extra=self._log_extra("TUI_RPC"),
            )
        else:
            self._backend = RpaDirectDriver()
            logger.debug(
                "RPA adapter configured for direct mode",
                extra=self._log_extra("RPA"),
            )

    async def _connect_implementation(self) -> None:
        if self._connection_task and not self._connection_task.done():
            logger.warning(
                "Connect called with active connection task",
                extra=self._log_extra("RPA"),
            )
            return

        self._keep_running = True
        self._connection_task = asyncio.create_task(
            self._connection_loop(),
            name="ruidarpa-connection-loop",
        )

    async def _connection_loop(self) -> None:
        """Background reconnection loop with exponential backoff."""
        loop = asyncio.get_running_loop()
        delay = self.RECONNECT_BASE_DELAY
        label = "TUI_RPC" if self._tui_mode else "RPA"
        log_extra = self._log_extra(label)

        while self._keep_running:
            self.connection_status_changed.send(
                self, status=TransportStatus.CONNECTING, message=""
            )

            connected: bool = False
            try:
                backend = self._backend
                if backend is None:
                    raise DriverSetupError("Backend not initialized")

                if self._tui_mode:
                    client: RpaRpcClient = backend  # type: ignore
                    rpc_ok = await loop.run_in_executor(None, client.connect)
                    if not rpc_ok:
                        raise ConnectionError(
                            "Failed to establish RPyC connection"
                        )
                    udp_host = self._config.get("udp_host")
                    usb_device = self._config.get("usb_device")
                    started = await loop.run_in_executor(
                        None, client.start, udp_host, usb_device
                    )
                    connected = started
                    if connected:
                        # Register RPC callbacks for status/error/reply
                        await loop.run_in_executor(
                            None, client.register_status_listener,
                            self._on_rpa_status,
                        )
                        await loop.run_in_executor(
                            None, client.register_error_listener,
                            self._on_rpa_error,
                        )
                else:
                    driver: RpaDirectDriver = backend  # type: ignore
                    udp_host = self._config.get("udp_host")
                    usb_device = self._config.get("usb_device")
                    connected = await loop.run_in_executor(
                        None, driver.start, udp_host, usb_device
                    )
                    if connected:
                        # Register direct driver callbacks
                        driver.register_status_listener(
                            self._on_rpa_status
                        )
                        driver.register_error_listener(
                            self._on_rpa_error
                        )
                        driver.register_reply_listener(
                            self._on_rpa_reply
                        )

                if not connected:
                    raise ConnectionError(
                        "Failed to connect to Ruida controller"
                    )

                # --- Connected successfully ---
                delay = self.RECONNECT_BASE_DELAY
                self._is_connected = True
                self.state.status = DeviceStatus.IDLE
                self.state_changed.send(self, state=self.state)

                logger.info(
                    "Connected to Ruida controller via RPA",
                    extra=log_extra,
                )

                # Poll connection health
                while self._keep_running:
                    await asyncio.sleep(self.CONNECTION_POLL_INTERVAL)
                    assert backend is not None
                    _backend = backend
                    is_alive = await loop.run_in_executor(
                        None, lambda: _backend.is_connected
                    )
                    if not is_alive:
                        logger.warning(
                            "RPA connection lost",
                            extra=log_extra,
                        )
                        self._is_connected = False
                        break

            except asyncio.CancelledError:
                logger.debug("Connection loop cancelled", extra=log_extra)
                self._is_connected = False
                break
            except Exception as e:
                logger.warning(
                    "RPA reconnect attempt failed: %s", e,
                    extra=log_extra,
                )
                self.connection_status_changed.send(
                    self,
                    status=TransportStatus.ERROR,
                    message=str(e),
                )
                self._is_connected = False
                await self._stop_backend()

            # --- Reconnect delay with exponential backoff ---
            if self._keep_running:
                jitter = 1.0 + random.uniform(
                    -self.RECONNECT_JITTER, self.RECONNECT_JITTER
                )
                sleep_time = delay * jitter
                logger.debug(
                    "Reconnecting in %.1f seconds (base=%.1f, jitter=%.2f)",
                    sleep_time, delay, jitter,
                    extra=log_extra,
                )
                await asyncio.sleep(sleep_time)
                delay = min(delay * 2, self.RECONNECT_MAX_DELAY)

        logger.debug("Exiting RPA connection loop", extra=log_extra)

    # --- RPC / Direct driver callbacks ---

    def _on_rpa_status(self, event: Any) -> None:
        """Handle status events from the Ruida controller via RPC/direct mode.

        Called from the backend's background thread. Bridges to the adapter's
        state tracking.

        Args:
            event: A status string (e.g. 'CONNECTED', 'DISCONNECTED'), an
                RdStatusEvent enum member (direct mode), or a StatusDict dict
                for machine status updates.
        """
        if self._shutting_down:
            return
        # RdStatusEvent enum → string value (direct mode)
        if RdStatusEvent is not None and isinstance(event, RdStatusEvent):
            event = event.value
        if isinstance(event, str):
            if event == "CONNECTED":
                self._is_connected = True
                self.state.status = DeviceStatus.IDLE
                self.state_changed.send(self, state=self.state)
                self.connection_status_changed.send(
                    self, status=TransportStatus.CONNECTED, message=""
                )
                logger.info("RPA connected via %s",
                            "RPC" if self._tui_mode else "direct",
                            extra=self._log_extra(
                                "TUI_RPC" if self._tui_mode else "RPA"))
            elif event == "DISCONNECTED":
                self._is_connected = False
                self.state.status = DeviceStatus.UNKNOWN
                self.state_changed.send(self, state=self.state)
                self.connection_status_changed.send(
                    self, status=TransportStatus.DISCONNECTED, message=""
                )
                logger.warning("RPA disconnected",
                               extra=self._log_extra(
                                   "TUI_RPC" if self._tui_mode else "RPA"))
            elif event == "TERMINATED":
                self._is_connected = False
                self.state.status = DeviceStatus.UNKNOWN
                self.state_changed.send(self, state=self.state)
                self.connection_status_changed.send(
                    self, status=TransportStatus.DISCONNECTED, message=""
                )
        elif isinstance(event, dict):
            # StatusDict or RPyC netref — convert to local dict for reliable
            # type handling
            event = {k: event[k] for k in event}  # type: ignore
            status_value = event.get("status") or event.get(
                "MEM_MACHINE_STATUS"
            )
            if status_value is not None:
                logger.debug("RPA status update: %s", status_value,
                             extra=self._log_extra(
                                 "TUI_RPC" if self._tui_mode else "RPA"))

            # Extract current position (values in µm → convert to mm)
            # MEM_CURRENT_POSITION_* values are (int_um, str_description)
            pos_x_um = _unwrap_um(event.get("MEM_CURRENT_POSITION_X"))
            pos_y_um = _unwrap_um(event.get("MEM_CURRENT_POSITION_Y"))
            pos_z_um = _unwrap_um(event.get("MEM_CURRENT_POSITION_Z"))

            if any(v is not None for v in (pos_x_um, pos_y_um, pos_z_um)):
                current = self.state.machine_pos
                new_x = (
                    (current[0] or 0.0)
                    if pos_x_um is None
                    else pos_x_um / 1000.0
                )
                new_y = (
                    (current[1] or 0.0)
                    if pos_y_um is None
                    else pos_y_um / 1000.0
                )
                new_z = (
                    (current[2] or 0.0)
                    if pos_z_um is None
                    else pos_z_um / 1000.0
                )
                new_pos = (new_x, new_y, new_z)

                if new_pos != current:
                    self.state = replace(self.state, machine_pos=new_pos)
                    logger.debug(
                        "RPA position update: x=%.3f y=%.3f z=%.3f",
                        new_x, new_y, new_z,
                        extra=self._log_extra(
                            "TUI_RPC" if self._tui_mode else "RPA"),
                    )
                    self.state_changed.send(self, state=self.state)

    def _on_rpa_error(self, msg: str) -> None:
        """Handle error events from the Ruida controller."""
        if self._shutting_down:
            return
        logger.warning("RPA error: %s", msg,
                       extra=self._log_extra(
                           "TUI_RPC" if self._tui_mode else "RPA"))

    def _on_rpa_reply(self, replies: tuple[str, ...]) -> None:
        """Handle reply data from the Ruida controller."""
        if self._shutting_down:
            return
        logger.debug("RPA reply: %d lines", len(replies),
                     extra=self._log_extra(
                         "TUI_RPC" if self._tui_mode else "RPA"))

    async def _stop_backend(self) -> None:
        """Stop and release the current backend driver."""
        if self._backend is None:
            return

        loop = asyncio.get_running_loop()
        try:
            if self._tui_mode:
                client: RpaRpcClient = self._backend  # type: ignore
                if client.is_connected:
                    # Unregister listeners before stopping to prevent
                    # stale-callback warnings on the server.
                    await loop.run_in_executor(
                        None, client.unregister_status_listener,
                        self._on_rpa_status,
                    )
                    await loop.run_in_executor(
                        None, client.unregister_error_listener,
                        self._on_rpa_error,
                    )
                    await loop.run_in_executor(None, client.stop)
                    await loop.run_in_executor(None, client.disconnect)
            else:
                await loop.run_in_executor(None, self._backend.stop)
        except Exception:
            logger.exception("Error stopping RPA backend")

    # --- Script execution ---

    async def _run_script(self, script_lines: List[str],
                          auto_checksum: bool = False) -> None:
        """Run an rpascript via the backend in a thread pool executor.

        Args:
            script_lines: Rpascript command lines to execute.
            auto_checksum: Whether to auto-calculate END_JOB checksum.
                Only set True for full job scripts that contain END_JOB.
        """
        if not script_lines:
            return
        if self._backend is None:
            raise DriverSetupError("Backend not initialized")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._backend.run, script_lines, auto_checksum
        )

    # --- Job control ---

    async def run(
        self,
        encoded: EncodedOutput,
        doc: Doc,
        ops: Ops,
        on_command_done: Optional[
            Callable[[int], Union[None, Awaitable[None]]]
        ] = None,
    ) -> None:
        text_lines = [
            line.strip()
            for line in encoded.text.splitlines()
            if line.strip()
        ]
        op_map = encoded.op_map

        if on_command_done is not None:
            num_ops = 0
            if op_map and op_map.op_to_machine_code:
                num_ops = max(op_map.op_to_machine_code.keys()) + 1
            for op_index in range(num_ops):
                result = on_command_done(op_index)
                if inspect.isawaitable(result):
                    await result

        logger.info(
            "Executing %d rpascript commands",
            len(text_lines),
            extra=self._log_extra("TUI_RPC" if self._tui_mode else "RPA"),
        )

        if text_lines:
            await self._run_script(text_lines, auto_checksum=True)

        self.job_finished.send(self)

    async def run_raw(self, machine_code: str) -> None:
        lines = [
            line.strip()
            for line in machine_code.splitlines()
            if line.strip()
        ]
        if lines:
            logger.info(
                "Executing %d raw rpascript lines",
                len(lines),
                extra=self._log_extra("TUI_RPC" if self._tui_mode else "RPA"),
            )
            await self._run_script(lines, auto_checksum=True)
        self.job_finished.send(self)

    async def set_hold(self, hold: bool = True) -> None:
        if hold:
            await self._run_script(["PAUSE_JOB"])
        else:
            await self._run_script(["RESTORE_JOB"])

    async def cancel(self) -> None:
        await self._run_script(["STOP_JOB"])

    async def clear_alarm(self) -> None:
        await self._run_script(["STOP_JOB"])

    # --- Movement ---

    async def home(self, axes: Optional[Axis] = None) -> None:
        # TODO: The Speed:600 needs to be configurable.
        # The RPA TUI service defaults to 600 mm/s.
        cmds: List[str] = []
        if axes is None:
            cmds.append("MOVE_RAPID_XY Option=RAPID_ORIGIN X=-5 Y=-5")
        else:
            if axes & (Axis.X & Axis.Y):
                cmds.append("MOVE_RAPID_XY Option=RAPID_ORIGIN X=-5 Y=-5")
            elif axes & Axis.X:
                cmds.append("MOVE_RAPID_X Option=RAPID_ORIGIN X=0.0mm")
            elif axes & Axis.Y:
                cmds.append("MOVE_RAPID_Y Option=RAPID_ORIGIN Y=0.0mm")
            if axes & Axis.Z:
                cmds.append("HOME_Z")
        if cmds:
            cmds.insert(0, "SPEED_LASER_1 Speed:600")
            await self._run_script(cmds)

    async def move_to(self, pos_x: float, pos_y: float) -> None:
        # TODO: The coordinates coming from the UI are inverted. Why?
        # For now, invert them here to match user expectations.
        pos_x = -pos_x
        pos_y = -pos_y
        logger.info(
            "move_to x=%.3f y=%.3f", pos_x, pos_y,
            extra=self._log_extra("TUI_RPC" if self._tui_mode else "RPA"),
        )
        cmds: List[str] = []
        cmds.append("SPEED_LASER_1 Speed:600")
        cmds.append(
            f"MOVE_RAPID_XY Option=RAPID_ORIGIN"
            f" X={pos_x:.3f}mm Y={pos_y:.3f}mm")
        await self._run_script(cmds)

    async def select_tool(self, tool_number: int) -> None:
        pass

    async def jog(self, speed: int, **deltas: float) -> None:
        # TODO: Jog speed is in mm/min, but the RPA TUI service expects mm/s.
        # Convert for now.
        speed_mm_per_s = speed / 60.0
        cmds: List[str] = []
        cmds.append(f"SPEED_LASER_1 Speed:{speed_mm_per_s:.1f}")
        _move_x = False
        _move_y = False
        for axis_name, delta in deltas.items():
            axis_lower = axis_name.lower()
            if axis_lower == "x":
                _move_x = True
            elif axis_lower == "y":
                _move_y = True
        if _move_x and _move_y:
            _delta_x = deltas.get("x", 0.0)
            _delta_y = deltas.get("y", 0.0)
            cmds.append(
                f"MOVE_RAPID_XY Option=RAPID_NONE X={_delta_x:.3f}mm "
                f"Y={_delta_y:.3f}mm")
        else:
            if _move_x:
                _delta_x = deltas.get("x", 0.0)
                cmds.append(
                    f"MOVE_RAPID_X Option=RAPID_NONE"
                    f" X={_delta_x:.3f}mm")
            if _move_y:
                _delta_y = deltas.get("y", 0.0)
                cmds.append(
                    f"MOVE_RAPID_Y Option=RAPID_NONE"
                    f" Y={_delta_y:.3f}mm")
        if cmds:
            logger.debug(
                "Jogging axes: %s",
                ", ".join(
                    f"{k}={v:.3f}" for k, v in deltas.items()
                ),
                extra=self._log_extra("TUI_RPC" if self._tui_mode else "RPA"),
            )
            await self._run_script(cmds)

    # --- Power / Laser ---

    async def set_power(self, head: Laser, percent: float) -> None:
        power_pct = percent * 100.0
        laser_num = head.tool_number + 1
        cmd = f"IMD_POWER_{laser_num} Power={power_pct:.1f}%"
        await self._run_script([cmd])

    async def set_focus_power(self, head: Laser, percent: float) -> None:
        await self.set_power(head, percent)

    # --- WCS ---

    async def select_wcs(self, wcs: str) -> None:
        if wcs == "REF0":
            await self._run_script(["REF_POINT_1"])
        elif wcs == "REF1":
            await self._run_script(["REF_POINT_2"])

    async def set_wcs_offset(
        self, wcs_slot: str, x: float, y: float, z: float
    ) -> None:
        if wcs_slot == "MACHINE":
            return

        if wcs_slot not in ("REF0", "REF1"):
            logger.warning("Unknown WCS slot: %s", wcs_slot)
            return

        # Rpascript uses mm natively; the SET_SETTING command stores
        # user origin values as integer micrometers in the controller.
        await self._run_script([
            "SET_SETTING "
            f"MEM_USER_ORIGIN_X={int(x * 1000)} "
            f"MEM_USER_ORIGIN_Y={int(y * 1000)}",
        ])
        self.wcs_updated.send(self, offsets={wcs_slot: (x, y, z)})

    async def read_wcs_offsets(self) -> Dict[str, Pos]:
        offsets: Dict[str, Pos] = {"MACHINE": (0.0, 0.0, 0.0)}
        self.wcs_updated.send(self, offsets=offsets)
        return offsets

    async def read_parser_state(self) -> Optional[str]:
        return None

    # --- Settings ---

    async def read_settings(self) -> None:
        await asyncio.sleep(0)
        self.settings_read.send(self, settings=[])

    async def write_setting(self, key: str, value: Any) -> None:
        pass

    def get_setting_vars(self) -> List[VarSet]:
        return [VarSet(title=_("No settings"))]

    # --- Probing ---

    async def run_probe_cycle(
        self, axis: Axis, max_travel: float, feed_rate: int
    ) -> Optional[Pos]:
        self.probe_status_changed.send(
            self, message=_("Probe not supported by RPA driver")
        )
        return None

    # --- Capabilities ---

    def can_jog(self, axis: Optional[Axis] = None) -> bool:
        return True

    def get_laser_capabilities(self, laser: Laser):
        if laser.laser_type.supports_pwm:
            return (
                PWMCapability(
                    frequency=laser.pwm_frequency,
                    max_frequency=laser.max_pwm_frequency,
                    pulse_width=laser.pulse_width,
                    min_pulse_width=laser.min_pulse_width,
                    max_pulse_width=laser.max_pulse_width,
                ),
            )
        return ()

    # --- Cleanup ---

    async def cleanup(self):
        self._shutting_down = True
        self._keep_running = False
        self._is_connected = False

        if self._connection_task:
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass
            self._connection_task = None

        await self._stop_backend()
        self._backend = None

        self.connection_status_changed.send(
            self, status=TransportStatus.DISCONNECTED, message=""
        )
        await super().cleanup()
