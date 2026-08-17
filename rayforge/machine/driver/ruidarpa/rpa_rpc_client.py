"""RPyC client for remote Ruida laser controller access.

Connects to a remote RPA TUI service via RPyC, allowing the driver
to communicate with a Ruida controller on another machine.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import rpyc
    from rpyc.utils.helpers import BgServingThread
else:
    try:
        import rpyc
        from rpyc.utils.helpers import BgServingThread
    except ImportError:
        rpyc = None  # type: ignore
        BgServingThread = None  # type: ignore

# RPyC's default sync_request_timeout is 30s; 5s detects a
# hung-but-alive server in ~5s instead of blocking the caller.
SYNC_REQUEST_TIMEOUT = 5.0


class RpaRpcClient:
    """RPyC client wrapper for remote Ruida controller access.

    Connects to an RPyC service exposing the RPA TUI adapter on
    127.0.0.1:18812 by default.

    Uses a ``BgServingThread`` to ensure server-initiated callbacks
    (status, error, reply) are processed reliably on a daemon thread.
    """

    DEFAULT_HOST = "127.0.0.1"
    DEFAULT_PORT = 18812

    def __init__(
        self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
    ) -> None:
        self._host = host
        self._port = port
        self._conn: Any = None
        self._bg_thread: Any = None

    # --- Lifecycle ---

    def connect(self) -> bool:
        """Connect to the RPyC service.

        Starts a background serving thread to ensure server-initiated
        callbacks (status, error, reply) are processed reliably.

        Returns:
            True if connection succeeded.
        """
        self._ensure_imported()
        if self._conn is not None:
            # Defensive re-entrancy guard: never leak a previous
            # connection when connect() is called while already connected.
            self.disconnect()
        try:
            self._conn = rpyc.connect(
                self._host,
                self._port,
                config={
                    "sync_request_timeout": SYNC_REQUEST_TIMEOUT,
                    # Server-raised GlueScriptDeltaMismatchError must
                    # arrive typed so the delta path can distinguish a
                    # contiguity break from other failures; rpyc 6.0.2
                    # defaults both flags to False (GenericException).
                    "import_custom_exceptions": True,
                    "instantiate_custom_exceptions": True,
                },
            )
            self._bg_thread = BgServingThread(self._conn)
            _logger.info(
                "RPA RPC client connected to %s:%d", self._host, self._port
            )
            return True
        except ConnectionRefusedError:
            _logger.error(
                "RPA RPC connection refused to %s:%d", self._host, self._port
            )
            return False
        except Exception:
            _logger.exception("RPA RPC connection failed")
            return False

    def disconnect(self) -> None:
        """Disconnect from the RPyC service.

        Closing the connection is the cleanup path: the server
        unregisters any callbacks registered by this client.
        """
        # Close connection FIRST to unblock serve() in the bg thread
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                _logger.exception("Error disconnecting RPA RPC")
            self._conn = None

        if self._bg_thread is not None:
            try:
                self._bg_thread.stop()
            except AssertionError:
                # BgServingThread already shut down due to connection
                # close — expected.
                pass
            except Exception:
                _logger.exception("Error stopping BgServingThread")
            self._bg_thread = None

    @property
    def is_connected(self) -> bool:
        """Whether the remote Ruida controller is connected."""
        if self._conn is None:
            return False
        try:
            return self._conn.root.is_connected()
        except Exception:
            # Dead transport (EOFError/AssertionError/socket errors):
            # treat as disconnected so the caller can clean up instead
            # of aborting mid-shutdown.
            _logger.debug("RPA RPC connection is dead", exc_info=True)
            return False

    def is_alive(self) -> bool:
        """Whether the RPyC transport itself is alive.

        Probes the remote service; only call success matters. The
        remote ``is_connected()`` VALUE is deliberately NOT interpreted
        — controller up/down transitions are handled by the status
        listener, not by the health probe.
        """
        if self._conn is None:
            return False
        try:
            self._conn.root.is_connected()
            return True
        except Exception:
            # The probe failed: transport dead or sync-request timeout.
            # Treat as not alive so the caller tears down and reconnects.
            _logger.debug("RPA RPC probe failed", exc_info=True)
            return False

    # --- Delegated RPC calls ---

    def start(
        self, udp_host: Optional[str] = None, usb_device: Optional[str] = None
    ) -> bool:
        """Start the remote TUI adapter.

        Args:
            udp_host: UDP hostname/IP for the remote machine.
            usb_device: USB device path on the remote machine.

        Returns:
            True if started successfully.
        """
        return self._call("start", udp_host=udp_host, usb_device=usb_device)

    def stop(self) -> None:
        """Stop the remote TUI adapter."""
        self._call("stop")

    def run(self, script: list[str], auto_checksum: bool = False) -> None:
        """Run an Rpascript on the remote machine.

        Queues the raw script without head/tail composition.

        Args:
            script: List of Rpascript command strings.
            auto_checksum: Whether to auto-calculate checksums.
        """
        if not script:
            return
        self._call("run", script, auto_checksum=auto_checksum)

    def run_staged_job(self, auto_checksum: bool = False) -> None:
        """Run the rpascript staged server-side via ``stage_gluescript``.

        Runs the staged rpascript. Raises remotely when nothing has
        been staged.

        Args:
            auto_checksum: Whether to auto-calculate checksums.
        """
        self._call("run_job", None, auto_checksum=auto_checksum)

    def set_head_script(self, script: list[str]) -> None:
        """Set the server-side head script composed into staged jobs.

        The server's RdDriver starts with a non-empty default head
        (REF_POINT_ABSOLUTE/SET_ABSOLUTE/REF_POINT_SET/
        ENABLE_BLOCK_CUTTING State:OFF) that ``run_job(None)`` prepends
        to every staged job. Rayforge's encoder output is fully
        self-framed, so the head must be cleared to [] after connect to
        prevent duplicate framing plus the block-cutting toggle reaching
        the controller.

        Args:
            script: Rpascript command lines; pass [] to neutralize.
        """
        self._call("set_head_script", script)

    def set_tail_script(self, script: list[str]) -> None:
        """Set the server-side tail script composed into staged jobs.

        Cleared to [] at connect alongside the head so the staged-job
        path carries no server-side framing; the encoder's output is
        fully self-framed already.

        Args:
            script: Rpascript command lines; pass [] to neutralize.
        """
        self._call("set_tail_script", script)

    def cancel_script(self) -> None:
        """Cancel the currently running script remotely."""
        self._call("cancel_script")

    def _reset_staged(self) -> None:
        """Reset the server-side staged GlueScript state.

        Drops any staged rpascript (RPC ``new_gluescript``) so a stale
        job cannot be run by ``run_staged_job()`` after a failed or
        aborted stage. Private: only the driver's staged pipeline needs
        it.
        """
        self._call("new_gluescript")

    # --- Live commands ---

    # Jog, home, and job-control commands execute server-side (TuiAdapter
    # ``_gluescript_live_command``) and return the sent rpascript lines.
    # The client must not run those lines again — double execution.

    def jog_xy_to(self, x: float, y: float) -> None:
        """Jog the remote XY axes to an absolute position."""
        self._call("jog_xy_to", x, y)

    def jog_x_to(self, x: float) -> None:
        """Jog the remote X axis to an absolute position."""
        self._call("jog_x_to", x)

    def jog_y_to(self, y: float) -> None:
        """Jog the remote Y axis to an absolute position."""
        self._call("jog_y_to", y)

    def jog_z_to(self, z: float) -> None:
        """Jog the remote Z axis to an absolute position."""
        self._call("jog_z_to", z)

    def jog_u_to(self, u: float) -> None:
        """Jog the remote U axis to an absolute position."""
        self._call("jog_u_to", u)

    def jog_xy_rel(
        self, x: Optional[float] = None, y: Optional[float] = None
    ) -> None:
        """Jog the remote XY axes relative to the current position."""
        self._call("jog_xy_rel", x, y)

    def jog_x_rel(self, x: Optional[float] = None) -> None:
        """Jog the remote X axis relative to the current position."""
        self._call("jog_x_rel", x)

    def jog_y_rel(self, y: Optional[float] = None) -> None:
        """Jog the remote Y axis relative to the current position."""
        self._call("jog_y_rel", y)

    def jog_z_rel(self, z: Optional[float] = None) -> None:
        """Jog the remote Z axis relative to the current position."""
        self._call("jog_z_rel", z)

    def jog_u_rel(self, u: Optional[float] = None) -> None:
        """Jog the remote U axis relative to the current position."""
        self._call("jog_u_rel", u)

    def jog_set_xy_speed(self, speed: float) -> None:
        """Set the remote XY jog speed in mm/s."""
        self._call("jog_set_xy_speed", speed)

    def jog_set_z_speed(self, speed: float) -> None:
        """Set the remote Z jog speed in mm/s."""
        self._call("jog_set_z_speed", speed)

    def jog_set_u_speed(self, speed: float) -> None:
        """Set the remote U jog speed in mm/s."""
        self._call("jog_set_u_speed", speed)

    def jog_set_xy_rel(self, delta: float) -> None:
        """Set the remote relative XY jog distance in mm."""
        self._call("jog_set_xy_rel", delta)

    def jog_set_z_rel(self, delta: float) -> None:
        """Set the remote relative Z jog distance in mm."""
        self._call("jog_set_z_rel", delta)

    def jog_set_u_rel(self, delta: float) -> None:
        """Set the remote relative U jog distance in mm."""
        self._call("jog_set_u_rel", delta)

    def home(self) -> None:
        """Home the remote X and Y axes."""
        self._call("home")

    def home_z(self) -> None:
        """Home the remote Z axis."""
        self._call("home_z")

    def home_u(self) -> None:
        """Home the remote U axis (rotary)."""
        self._call("home_u")

    def pause(self) -> None:
        """Pause the running job on the remote controller.

        The returned sent lines are deliberately discarded so the pause
        is sent exactly once, never run again.
        """
        self._call("pause")

    def resume(self) -> None:
        """Resume the paused job on the remote controller.

        The returned sent lines are deliberately discarded so the resume
        is sent exactly once, never run again.
        """
        self._call("resume")

    def stop_job(self) -> None:
        """Stop the running job on the remote controller.

        The returned sent lines are deliberately discarded so the stop
        is sent exactly once, never run again.
        """
        self._call("stop_job")

    def reset(self) -> None:
        """Reset the remote controller.

        Stops the current job and homes the X/Y axes
        (reset = ["STOP_JOB", "HOME_XY"]). The returned sent lines are
        deliberately discarded so the reset is sent exactly once, never
        run again.
        """
        self._call("reset")

    # --- Listeners ---

    def register_status_listener(self, callback: Callable) -> None:
        """Register a status listener.

        Register at most once per connection; cleanup is connection
        close — the server clears listeners via on_disconnect (weakref).

        Args:
            callback: Callable accepting a status string.
        """
        self._require_connected()
        self._conn.root.exposed_register_status_listener(callback)

    def register_error_listener(self, callback: Callable) -> None:
        """Register an error listener.

        Register at most once per connection; cleanup is connection
        close — the server clears listeners via on_disconnect (weakref).

        Args:
            callback: Callable accepting an error string.
        """
        self._require_connected()
        self._conn.root.exposed_register_error_listener(callback)

    def register_reply_listener(self, callback: Callable) -> None:
        """Register a reply listener.

        Register at most once per connection; cleanup is connection
        close — the server clears listeners via on_disconnect (weakref).

        Args:
            callback: Callable accepting a tuple of reply strings.
        """
        self._require_connected()
        self._conn.root.exposed_register_reply_listener(callback)

    # --- Properties ---

    @property
    def root(self) -> Any:
        """The RPyC service root, connected-guarded.

        Exposes typed netref access to the remote service so the staged
        pipeline can call ``exposed_stage_gluescript`` /
        ``exposed_stage_gluescript_delta`` directly with plain Python
        values (lists and ints must cross as real values, not the
        string-dispatched ``_call`` helper).
        """
        self._require_connected()
        return self._conn.root

    @property
    def machine_status(self) -> dict:
        """Current machine status from the remote controller."""
        if self._conn is None:
            return {}
        try:
            return self._conn.root.machine_status()
        except Exception:
            # Dead transport (EOFError/AssertionError/socket errors):
            # treat as disconnected so the caller can clean up instead
            # of aborting mid-shutdown.
            _logger.debug("RPA RPC machine status read failed", exc_info=True)
            return {}

    # --- Internal helpers ---

    def _call(self, method: str, *args, **kwargs) -> Any:
        """Call an exposed method on the RPyC service.

        Args:
            method: Method name without 'exposed_' prefix.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Method return value.
        """
        self._require_connected()
        exposed = getattr(self._conn.root, f"exposed_{method}")
        return exposed(*args, **kwargs)

    def _ensure_imported(self) -> None:
        """Raise ImportError if rpyc or BgServingThread is not available."""
        if rpyc is None or BgServingThread is None:
            raise ImportError(
                "rpyc is not installed. Run: pixi run -e ruidarpa ..."
            )

    def _require_connected(self) -> None:
        """Raise RuntimeError if not connected."""
        if self._conn is None:
            raise RuntimeError("RPA RPC client is not connected")
