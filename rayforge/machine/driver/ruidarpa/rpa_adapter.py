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
    Tuple,
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

# Default speed for move_to() absolute jogs (mm/s), used until the first
# jog() records the GUI-configured speed. After that, move_to() reuses the
# last jog speed so absolute moves do not fight the jog speed.
DEFAULT_MOVE_TO_JOG_SPEED_MM_S = 600.0


def _unwrap_mm(value: object) -> Optional[float]:
    """Extract the mm value from a POSITION_* field.

    StatusDict positions arrive as ``(float_mm, str_description)``
    tuples in both direct and TUI RPC modes. Accept both forms for
    forward compatibility.
    """
    if isinstance(value, (list, tuple)):
        return value[0]  # type: ignore[return-value]
    return value  # type: ignore[return-value]


def _render_transcript_lines(name: str, args: Tuple) -> List[str]:
    """Render one recorded GlueScript call as server transcript line(s).

    Mirrors the driver's own transcript mirroring: ``comment`` and
    ``inline`` expand to one line per item (the driver appends a mirror
    line per element), so the replayed server transcript length stays
    contiguous for the delta flush guard.
    """
    if name == "comment" and args and isinstance(args[0], (list, tuple)):
        return [f"comment({[line]!r})" for line in args[0]]
    if name == "inline" and args and isinstance(args[0], (list, tuple)):
        return [f"inline({[command]!r})" for command in args[0]]
    args_str = ", ".join(repr(arg) for arg in args)
    return [f"{name}({args_str})"]


def _stage_plan(client: RpaRpcClient, plan: List[Tuple[str, Tuple]]) -> None:
    """Replay a recorded GlueScript plan on the server via RPC.

    The server's replay registry has no ``add_layer_action`` case (raw
    lines are forwarded-only upstream), so the plan is walked in
    recorded order: structural calls are flushed as
    ``stage_gluescript_delta`` batches, and each ``add_layer_action`` is
    forwarded at its exact interleaved position. Every delta
    re-assembles the rpascript, so the final ``end_job`` flush stages
    the complete job. A failed stage is torn down so a stale job
    cannot be run afterwards.

    Args:
        client: The connected RPC client.
        plan: Recorded ``(method_name, args)`` pairs from the encoder.

    Raises:
        RuntimeError: If the server rejects a replayed batch or the
            plan never reached ``end_job()``.
    """
    try:
        root = client.root
        root.exposed_new_gluescript()
        flushed = 0
        buffer: List[str] = []
        for name, args in plan:
            if name == "add_layer_action":
                if buffer:
                    root.exposed_stage_gluescript_delta(
                        flushed, buffer, require_complete=False
                    )
                    flushed += len(buffer)
                    buffer = []
                root.exposed_add_layer_action(*args)
            else:
                buffer.extend(_render_transcript_lines(name, args))
        if buffer:
            root.exposed_stage_gluescript_delta(
                flushed, buffer, require_complete=True
            )
    except Exception:
        try:
            client._reset_staged()
        except Exception:
            logger.exception("Failed to reset staged state after stage error")
        raise


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
        self._jog_speed_mm_s: Optional[float] = None

    # --- Properties ---

    @property
    def machine_space_wcs(self) -> str:
        return "MACHINE"

    @property
    def machine_space_wcs_display_name(self) -> str:
        return _("Ruida Coordinates")

    @property
    def supported_wcs(self) -> List[str]:
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

    # --- Protect ---

    async def get_protect(self) -> bool:
        """Return whether protect mode is enabled."""
        if self._backend is None:
            return False
        if isinstance(self._backend, RpaDirectDriver):
            return self._backend.protect_enabled
        return False

    async def set_protect(self, enabled: bool) -> None:
        """Enable or disable protect mode."""
        if self._backend is None:
            return
        if isinstance(self._backend, RpaDirectDriver):
            self._backend.set_protect(enabled)

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
                        "Enable TUI RPC connection to a remote RPA TUI service"
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
                        # Register RPC callbacks on the fresh RPyC
                        # connection; the server clears them on
                        # disconnect, so a reconnect registers at most
                        # once per connection.
                        await loop.run_in_executor(
                            None,
                            client.register_status_listener,
                            self._on_rpa_status,
                        )
                        await loop.run_in_executor(
                            None,
                            client.register_error_listener,
                            self._on_rpa_error,
                        )
                        await loop.run_in_executor(
                            None,
                            client.register_reply_listener,
                            self._on_rpa_reply,
                        )
                        # Neutralize the server driver's default head/tail
                        # composition: staged jobs are fully self-framed by
                        # the encoder, and run_job(None) prepends the driver's
                        # default head script otherwise.
                        await loop.run_in_executor(
                            None, client.set_head_script, []
                        )
                        await loop.run_in_executor(
                            None, client.set_tail_script, []
                        )
                else:
                    driver: RpaDirectDriver = backend  # type: ignore
                    udp_host = self._config.get("udp_host")
                    usb_device = self._config.get("usb_device")
                    connected = await loop.run_in_executor(
                        None, driver.start, udp_host, usb_device
                    )
                    if connected:
                        # Direct mode retains callbacks across stop/start,
                        # so drop any stale registration before
                        # re-registering — repeated registers would
                        # double-fire every status event.
                        driver.unregister_status_listener(self._on_rpa_status)
                        driver.unregister_error_listener(self._on_rpa_error)
                        driver.unregister_reply_listener(self._on_rpa_reply)
                        driver.register_status_listener(self._on_rpa_status)
                        driver.register_error_listener(self._on_rpa_error)
                        driver.register_reply_listener(self._on_rpa_reply)

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
                    if self._tui_mode:
                        client: RpaRpcClient = _backend  # type: ignore
                        is_alive = await loop.run_in_executor(
                            None, client.is_alive
                        )
                    else:
                        driver: RpaDirectDriver = _backend  # type: ignore
                        is_alive = await loop.run_in_executor(
                            None, lambda: driver.is_connected
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
                    "RPA reconnect attempt failed: %s",
                    e,
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
                    sleep_time,
                    delay,
                    jitter,
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
                logger.info(
                    "RPA connected via %s",
                    "RPC" if self._tui_mode else "direct",
                    extra=self._log_extra(
                        "TUI_RPC" if self._tui_mode else "RPA"
                    ),
                )
            elif event == "DISCONNECTED":
                self._is_connected = False
                self.state.status = DeviceStatus.UNKNOWN
                self.state_changed.send(self, state=self.state)
                self.connection_status_changed.send(
                    self, status=TransportStatus.DISCONNECTED, message=""
                )
                logger.warning(
                    "RPA disconnected",
                    extra=self._log_extra(
                        "TUI_RPC" if self._tui_mode else "RPA"
                    ),
                )
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
                "MACHINE_STATUS"
            )
            if status_value is not None:
                logger.debug(
                    "RPA status update: %s",
                    status_value,
                    extra=self._log_extra(
                        "TUI_RPC" if self._tui_mode else "RPA"
                    ),
                )

            # Extract current position (values in mm)
            # POSITION_* values are (float_mm, str_description)
            pos_x = _unwrap_mm(event.get("POSITION_X"))
            pos_y = _unwrap_mm(event.get("POSITION_Y"))
            pos_z = _unwrap_mm(event.get("POSITION_Z"))

            if any(v is not None for v in (pos_x, pos_y, pos_z)):
                current = self.state.machine_pos
                new_x = (current[0] or 0.0) if pos_x is None else pos_x
                new_y = (current[1] or 0.0) if pos_y is None else pos_y
                new_z = (current[2] or 0.0) if pos_z is None else pos_z
                new_pos = (new_x, new_y, new_z)

                if new_pos != current:
                    self.state = replace(self.state, machine_pos=new_pos)
                    logger.debug(
                        "RPA position update: x=%.3f y=%.3f z=%.3f",
                        new_x,
                        new_y,
                        new_z,
                        extra=self._log_extra(
                            "TUI_RPC" if self._tui_mode else "RPA"
                        ),
                    )
                    self.state_changed.send(self, state=self.state)

    def _on_rpa_error(self, msg: str) -> None:
        """Handle error events from the Ruida controller."""
        if self._shutting_down:
            return
        logger.warning(
            "RPA error: %s",
            msg,
            extra=self._log_extra("TUI_RPC" if self._tui_mode else "RPA"),
        )

    def _on_rpa_reply(self, replies: list[str]) -> None:
        """Handle reply data from the Ruida controller."""
        if self._shutting_down:
            return
        logger.debug(
            "RPA reply: %d lines",
            len(replies),
            extra=self._log_extra("TUI_RPC" if self._tui_mode else "RPA"),
        )

    async def _stop_backend(self) -> None:
        """Stop and release the current backend driver."""
        if self._backend is None:
            return

        loop = asyncio.get_running_loop()
        logger.debug(
            "Stopping RPA backend (%s)",
            "TUI RPC" if self._tui_mode else "direct",
            extra=self._log_extra("TUI_RPC" if self._tui_mode else "RPA"),
        )
        try:
            if self._tui_mode:
                client: RpaRpcClient = self._backend  # type: ignore
                # is_connected is a blocking RPyC round trip; evaluate
                # it off the event loop thread like the poll loop does
                # so a hung-but-alive server cannot freeze the UI.
                try:
                    is_alive = await loop.run_in_executor(
                        None, lambda: client.is_connected
                    )
                    if is_alive:
                        # Closing the connection is the cleanup: the
                        # server unregisters this client's callbacks on
                        # disconnect.
                        await loop.run_in_executor(None, client.stop)
                finally:
                    # Disconnect must always run, even when stop() raised
                    # on an already-dead transport.
                    await loop.run_in_executor(None, client.disconnect)
            else:
                driver: RpaDirectDriver = self._backend  # type: ignore
                if driver.is_connected:
                    try:
                        driver.unregister_status_listener(self._on_rpa_status)
                    except Exception:
                        logger.exception("Error unregistering status listener")
                    try:
                        driver.unregister_error_listener(self._on_rpa_error)
                    except Exception:
                        logger.exception("Error unregistering error listener")
                    try:
                        driver.unregister_reply_listener(self._on_rpa_reply)
                    except Exception:
                        logger.exception("Error unregistering reply listener")
                await loop.run_in_executor(None, driver.stop)
        except Exception:
            logger.exception("Error stopping RPA backend")

    # --- Script execution ---

    async def _run_script(
        self, script_lines: List[str], auto_checksum: bool = False
    ) -> None:
        """Run raw rpascript via the backend's ``run``.

        All scripts — jobs and runtime commands — are sent raw through
        ``backend.run()``. The encoder output is already self-framed
        (REF_POINT/SET_ABSOLUTE/START_JOB…END_JOB), so head/tail
        composition is never used.

        Args:
            script_lines: Rpascript command lines to execute.
            auto_checksum: Whether to auto-calculate checksums. Passed
                for full job scripts (the current ruida-pa runner ignores
                this flag and patches END_JOB unconditionally).
        """
        if not script_lines:
            return
        if self._backend is None:
            raise DriverSetupError("Backend not initialized")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._backend.run, script_lines, auto_checksum
        )

    async def _run_staged_job(self, encoded: EncodedOutput) -> None:
        """Run a job by re-staging its GlueScript plan server-side.

        TUI RPC mode only: the encoded output carries the recorded
        GlueScript call plan (``rpa_plan``). The plan is replayed on
        the server via the RPC GlueScript sink (``_stage_plan``), then
        the staged rpascript is queued with ``run_staged_job``. A
        failed stage is torn down so a stale job cannot be run
        afterwards.

        Args:
            encoded: The encoder output; must carry ``rpa_plan``.

        Raises:
            DriverSetupError: If the backend is not initialized or the
                plan is missing.
        """
        if self._backend is None:
            raise DriverSetupError("Backend not initialized")
        if encoded.rpa_plan is None:
            raise DriverSetupError(
                "TUI RPC mode requires a GlueScript plan (rpa_plan) — "
                "re-encode the job with the ruidarpa encoder"
            )
        client: RpaRpcClient = self._backend  # type: ignore
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _stage_plan, client, encoded.rpa_plan)
        await loop.run_in_executor(None, client.run_staged_job)

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
            line.strip() for line in encoded.text.splitlines() if line.strip()
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

        if self._tui_mode:
            if text_lines:
                # TUI RPC re-stages the GlueScript plan server-side
                # instead of shipping the assembled rpascript text.
                await self._run_staged_job(encoded)
        elif text_lines:
            await self._run_script(text_lines, auto_checksum=True)

        self.job_finished.send(self)

    async def run_raw(self, machine_code: str) -> None:
        lines = [
            line.strip() for line in machine_code.splitlines() if line.strip()
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
        if self._backend is None:
            raise DriverSetupError("Backend not initialized")
        loop = asyncio.get_running_loop()
        # Live homing commands auto-send server-side (and the direct
        # backend auto-sends via the wrapped RdDriver); the returned
        # lines must not be run() again.
        if axes is None or (axes & (Axis.X | Axis.Y)):
            await loop.run_in_executor(None, self._backend.home)
        if axes is not None and (axes & Axis.Z):
            await loop.run_in_executor(None, self._backend.home_z)

    async def move_to(self, pos_x: float, pos_y: float) -> None:
        """Move to an absolute position in machine-frame mm.

        Coordinates are machine-frame (same frame as POSITION_* status
        reporting: +X left of home, +Y down from home) and are passed
        through unchanged to the backend jog_xy_to.
        """
        logger.info(
            "move_to x=%.3f y=%.3f",
            pos_x,
            pos_y,
            extra=self._log_extra("TUI_RPC" if self._tui_mode else "RPA"),
        )
        if self._backend is None:
            raise DriverSetupError("Backend not initialized")
        loop = asyncio.get_running_loop()
        speed_mm_s = (
            self._jog_speed_mm_s
            if self._jog_speed_mm_s is not None
            else DEFAULT_MOVE_TO_JOG_SPEED_MM_S
        )
        await loop.run_in_executor(
            None, self._backend.jog_set_xy_speed, speed_mm_s
        )
        # Live jog commands auto-send server-side (and the direct backend
        # auto-sends via the wrapped RdDriver); the returned lines must
        # not be run() again.
        await loop.run_in_executor(None, self._backend.jog_xy_to, pos_x, pos_y)

    async def select_tool(self, tool_number: int) -> None:
        pass

    async def jog(self, speed: int, **deltas: float) -> None:
        # TODO: Jog speed is in mm/min, but the RPA TUI service expects mm/s.
        # Convert for now.
        speed_mm_per_s = speed / 60.0
        if self._backend is None:
            raise DriverSetupError("Backend not initialized")
        self._jog_speed_mm_s = speed_mm_per_s
        # The speed is recorded before the early return so speed-only
        # jogs update the stored move_to() speed without touching the
        # backend.
        if not deltas:
            return
        loop = asyncio.get_running_loop()
        # Re-assert the XY jog speed on every jog: move_to() overrides
        # the backend speed, so a stale value would otherwise persist
        # until the next move.
        await loop.run_in_executor(
            None, self._backend.jog_set_xy_speed, speed_mm_per_s
        )
        if "z" in deltas:
            await loop.run_in_executor(
                None, self._backend.jog_set_z_speed, speed_mm_per_s
            )
        if "u" in deltas:
            await loop.run_in_executor(
                None, self._backend.jog_set_u_speed, speed_mm_per_s
            )
        # Live jog commands auto-send server-side (and the direct backend
        # auto-sends via the wrapped RdDriver); the returned lines must
        # not be run() again.
        if "x" in deltas and "y" in deltas:
            await loop.run_in_executor(
                None, self._backend.jog_xy_rel, deltas["x"], deltas["y"]
            )
        else:
            if "x" in deltas:
                await loop.run_in_executor(
                    None, self._backend.jog_x_rel, deltas["x"]
                )
            if "y" in deltas:
                await loop.run_in_executor(
                    None, self._backend.jog_y_rel, deltas["y"]
                )
        if "z" in deltas:
            await loop.run_in_executor(
                None, self._backend.jog_z_rel, deltas["z"]
            )
        if "u" in deltas:
            await loop.run_in_executor(
                None, self._backend.jog_u_rel, deltas["u"]
            )

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
        raise NotImplementedError(
            _(
                "set_wcs_offset is not supported: protect mode blocks "
                "SET_SETTING, and the feature was dropped by user "
                "decision. Use select_wcs to pick a reference point."
            )
        )

    async def read_wcs_offsets(self) -> Dict[str, Pos]:
        offsets: Dict[str, Pos] = {
            "MACHINE": (0.0, 0.0, 0.0),
            "REF0": (0.0, 0.0, 0.0),
            "REF1": (0.0, 0.0, 0.0),
        }
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
