"""RPyC client for remote Ruida laser controller access.

Connects to a remote RPA TUI service via RPyC, allowing the driver
to communicate with a Ruida controller on another machine.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

_logger = logging.getLogger(__name__)

try:
    import rpyc
except ImportError:
    rpyc = None  # type: ignore


class RpaRpcClient:
    """RPyC client wrapper for remote Ruida controller access.

    Connects to an RPyC service exposing the RPA TUI adapter on
    127.0.0.1:18812 by default.
    """

    DEFAULT_HOST = "127.0.0.1"
    DEFAULT_PORT = 18812

    def __init__(self, host: str = DEFAULT_HOST,
                 port: int = DEFAULT_PORT) -> None:
        self._host = host
        self._port = port
        self._conn: Any = None

    # --- Lifecycle ---

    def connect(self) -> bool:
        """Connect to the RPyC service.

        Returns:
            True if connection succeeded.
        """
        self._ensure_imported()
        try:
            self._conn = rpyc.connect(self._host, self._port)
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
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                _logger.exception("Error disconnecting RPA RPC")
            self._conn = None

    @property
    def is_connected(self) -> bool:
        """Whether the RPyC connection is active."""
        return self._conn is not None

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
        """Register a status listener (local).

        Note: For RPC, the callback fires locally from a polling
        mechanism; actual status events are received by polling.

        Args:
            callback: Callable accepting a status string.
        """
        self._require_connected()
        self._conn.root.exposed_register_status_listener(callback)

    def register_error_listener(self, callback: Callable) -> None:
        """Register an error listener."""
        self._require_connected()
        self._conn.root.exposed_register_error_listener(callback)

    def register_reply_listener(self, callback: Callable) -> None:
        """Register a reply listener."""
        self._require_connected()
        self._conn.root.exposed_register_reply_listener(callback)

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
        """Raise ImportError if rpyc is not available."""
        if rpyc is None:
            raise ImportError(
                "rpyc is not installed. "
                "Run: pixi run -e ruidarpa ..."
            )

    def _require_connected(self) -> None:
        """Raise RuntimeError if not connected."""
        if self._conn is None:
            raise RuntimeError("RPA RPC client is not connected")
