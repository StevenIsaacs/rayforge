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


class RpaRpcClient:
    """RPyC client wrapper for remote Ruida controller access.

    Connects to an RPyC service exposing the RPA TUI adapter on
    127.0.0.1:18812 by default.

    Uses a ``BgServingThread`` to ensure server-initiated callbacks
    (status, error, reply) are processed reliably on a daemon thread.
    """

    DEFAULT_HOST = "127.0.0.1"
    DEFAULT_PORT = 18812

    def __init__(self, host: str = DEFAULT_HOST,
                 port: int = DEFAULT_PORT) -> None:
        self._host = host
        self._port = port
        self._conn: Any = None
        self._bg_thread: Any = None
        self._status_listener: Optional[Callable] = None
        self._error_listener: Optional[Callable] = None
        self._reply_listener: Optional[Callable] = None

    # --- Lifecycle ---

    def connect(self) -> bool:
        """Connect to the RPyC service.

        Starts a background serving thread to ensure server-initiated
        callbacks (status, error, reply) are processed reliably.

        Returns:
            True if connection succeeded.
        """
        self._ensure_imported()
        try:
            self._conn = rpyc.connect(self._host, self._port)
            self._bg_thread = BgServingThread(self._conn)
            _logger.info("RPA RPC client connected to %s:%d",
                         self._host, self._port)
            return True
        except ConnectionRefusedError:
            _logger.error("RPA RPC connection refused to %s:%d",
                          self._host, self._port)
            return False
        except Exception:
            _logger.exception("RPA RPC connection failed")
            return False

    def disconnect(self) -> None:
        """Disconnect from the RPyC service."""
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
        """Whether the RPyC connection is active."""
        return (
            self._conn is not None
            and self._bg_thread is not None
            and self._bg_thread._thread.is_alive()
        )

    # --- Delegated RPC calls ---

    def start(self, udp_host: Optional[str] = None,
              usb_device: Optional[str] = None) -> bool:
        """Start the remote TUI adapter.

        Args:
            udp_host: UDP hostname/IP for the remote machine.
            usb_device: USB device path on the remote machine.

        Returns:
            True if started successfully.
        """
        return self._call("start", udp_host=udp_host,
                          usb_device=usb_device)

    def stop(self) -> None:
        """Stop the remote TUI adapter."""
        self._call("stop")

    def run(self, script: list[str],
            auto_checksum: bool = False) -> None:
        """Run an Rpascript on the remote machine.

        Args:
            script: List of Rpascript command strings.
            auto_checksum: Whether to auto-calculate checksums.
        """
        self._call("run", script, auto_checksum=auto_checksum)

    def cancel_script(self) -> None:
        """Cancel the currently running script remotely."""
        self._call("cancel_script")

    @property
    def machine_status(self) -> dict:
        """Current machine status from remote."""
        if self._conn is None:
            return {}
        return self._conn.root.exposed_machine_status()

    def register_status_listener(self, callback: Callable) -> None:
        """Register a status listener.

        The callback reference is stored internally so that the same
        Python object (same ``id()``) is used for later unregistration.
        This is required because RPyC identifies remote callable
        references by ``id()``, and bound methods create a new object
        on each access.

        Args:
            callback: Callable accepting a status string.
        """
        self._require_connected()
        self._status_listener = callback
        self._conn.root.exposed_register_status_listener(callback)

    def register_error_listener(self, callback: Callable) -> None:
        """Register an error listener.

        The callback reference is stored internally for reliable
        unregistration (see :meth:`register_status_listener`).
        """
        self._require_connected()
        self._error_listener = callback
        self._conn.root.exposed_register_error_listener(callback)

    def register_reply_listener(self, callback: Callable) -> None:
        """Register a reply listener.

        The callback reference is stored internally for reliable
        unregistration (see :meth:`register_status_listener`).
        """
        self._require_connected()
        self._reply_listener = callback
        self._conn.root.exposed_register_reply_listener(callback)

    def unregister_status_listener(self, callback: Callable) -> None:
        """Unregister a status listener.

        Uses the internally stored callback reference (set by
        :meth:`register_status_listener`) to ensure the same Python
        object is passed — required for RPyC callable identity matching.

        Args:
            callback: Ignored; the stored reference is used instead.
                Included for API consistency.
        """
        self._require_connected()
        if self._status_listener is not None:
            try:
                self._conn.root.exposed_unregister_status_listener(
                    self._status_listener,
                )
            except AttributeError:
                pass  # Server may not support unregister (pre-v0.8.0)
            self._status_listener = None

    def unregister_error_listener(self, callback: Callable) -> None:
        """Unregister an error listener.

        Uses the internally stored callback reference (set by
        :meth:`register_error_listener`) for reliable RPyC unregister.

        Args:
            callback: Ignored; the stored reference is used instead.
        """
        self._require_connected()
        if self._error_listener is not None:
            try:
                self._conn.root.exposed_unregister_error_listener(
                    self._error_listener,
                )
            except AttributeError:
                pass  # Server may not support unregister (pre-v0.8.0)
            self._error_listener = None

    def unregister_reply_listener(self, callback: Callable) -> None:
        """Unregister a reply listener.

        Uses the internally stored callback reference (set by
        :meth:`register_reply_listener`) for reliable RPyC unregister.

        Args:
            callback: Ignored; the stored reference is used instead.
        """
        self._require_connected()
        if self._reply_listener is not None:
            try:
                self._conn.root.exposed_unregister_reply_listener(
                    self._reply_listener,
                )
            except AttributeError:
                pass  # Server may not support unregister (pre-v0.8.0)
            self._reply_listener = None

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
                "rpyc is not installed. "
                "Run: pixi run -e ruidarpa ..."
            )

    def _require_connected(self) -> None:
        """Raise RuntimeError if not connected."""
        if self._conn is None:
            raise RuntimeError("RPA RPC client is not connected")
