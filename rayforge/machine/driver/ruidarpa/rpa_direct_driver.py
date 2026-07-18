"""Direct USB/UDP driver for Ruida laser controllers.

Wraps the external RuidaDriver from ruida-protocol-analyzer for direct
communication over USB or UDP.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

from rayforge.pipeline.encoder.base import EncodedOutput

if TYPE_CHECKING:
    from ruidadriver.ruida_driver import RdDriver
    from rayforge.core.doc import Doc
    from rayforge.machine.driver.ruidarpa.rpa_encoder import (
        RuidaRPAEncoder,
    )
    from rayforge.machine.models.machine import Machine
    from raygeo.ops import Ops
else:
    try:
        from ruidadriver.ruida_driver import RdDriver  # noqa: E402
    except ImportError:
        RdDriver = None  # type: ignore[assignment]

_logger = logging.getLogger(__name__)


class RpaDirectDriver:
    """Direct connection to a Ruida laser controller via USB or UDP.

    Wraps RdDriver lifecycle and listener management. All callbacks
    fire from background daemon threads.
    """

    def __init__(self) -> None:
        self._driver: Optional[RdDriver] = None

    # --- Lifecycle ---

    def start(self, udp_host: Optional[str] = None,
              usb_device: Optional[str] = None) -> bool:
        """Start connection to the Ruida controller.

        Args:
            udp_host: UDP hostname/IP (e.g. '192.168.1.100').
            usb_device: USB device path.

        Returns:
            True if connection succeeded.
        """
        self._ensure_imported()
        self._driver = RdDriver()
        result = self._driver.start(udp_host=udp_host,
                                    usb_device=usb_device)
        if result:
            _logger.info("RPA direct driver connected; udp=%s, usb=%s",
                         udp_host, usb_device)
        else:
            _logger.warning("RPA direct driver failed to connect")
        return result

    def stop(self) -> None:
        """Disconnect and clean up."""
        if self._driver is not None:
            try:
                self._driver.stop()
            except Exception:
                _logger.exception("Error stopping RPA direct driver")
            self._driver = None

    @property
    def is_connected(self) -> bool:
        """Whether the underlying driver is connected."""
        return self._driver is not None and self._driver.is_connected

    # --- Run control ---

    def run(self, script: list[str],
            auto_checksum: bool = False) -> None:
        """Run an Rpascript.

        Args:
            script: List of Rpascript command strings.
            auto_checksum: Whether to auto-calculate checksums.
        """
        if not script:
            return
        driver = self._require_connected()
        driver.run(script, auto_checksum=auto_checksum)

    # --- Encoder integration ---

    @staticmethod
    def create_encoder() -> "RuidaRPAEncoder":
        """Create an RPA encoder for converting Ops to rpascript."""
        from rayforge.machine.driver.ruidarpa.rpa_encoder import (
            RuidaRPAEncoder,
        )

        return RuidaRPAEncoder()

    def run_encoded(
        self, encoded: EncodedOutput, auto_checksum: bool = False
    ) -> None:
        """Run an EncodedOutput by extracting its text lines.

        Args:
            encoded: The encoder output containing rpascript text.
            auto_checksum: Whether to auto-calculate checksums.
        """
        text_lines = [
            line.strip()
            for line in encoded.text.splitlines()
            if line.strip()
        ]
        self.run(text_lines, auto_checksum=auto_checksum)

    def encode_and_run(
        self,
        ops: "Ops",
        machine: "Machine",
        doc: "Doc",
        auto_checksum: bool = False,
    ) -> EncodedOutput:
        """Encode Ops to rpascript and run it on the controller.

        Args:
            ops: Ops object from raygeo containing commands to encode.
            machine: The machine configuration.
            doc: The document being processed.
            auto_checksum: Whether to auto-calculate checksums.

        Returns:
            The EncodedOutput produced by the encoder.
        """
        encoder = self.create_encoder()
        encoded = encoder.encode(ops, machine, doc)
        self.run_encoded(encoded, auto_checksum=auto_checksum)
        return encoded

    def cancel_script(self) -> None:
        """Cancel the currently running script."""
        if self._driver is not None:
            self._driver.cancel_script()

    # --- Head/Tail scripts ---

    def set_head_script(self, script: list[str]) -> None:
        """Set the head script executed before every job.

        The head script runs automatically by ``run_job`` before the
        job-specific commands.  Pass an empty list to clear.

        Args:
            script: List of rpascript command strings.
        """
        driver = self._require_connected()
        driver.set_head_script(script)

    def get_head_script(self) -> list[str]:
        """Return the current head script."""
        driver = self._require_connected()
        return driver.get_head_script()

    def set_tail_script(self, script: list[str]) -> None:
        """Set the tail script executed after every job.

        The tail script runs automatically by ``run_job`` after the
        job-specific commands.  Pass an empty list to clear.

        Args:
            script: List of rpascript command strings.
        """
        driver = self._require_connected()
        driver.set_tail_script(script)

    def get_tail_script(self) -> list[str]:
        """Return the current tail script."""
        driver = self._require_connected()
        return driver.get_tail_script()

    def run_job(self, script: list[str],
                auto_checksum: bool = False) -> None:
        """Run a job with automatic head and tail composition.

        Executes: head_script + *script* + tail_script.  For a raw
        script without head/tail, use ``run()`` instead.

        Args:
            script: Job-specific rpascript command strings.
            auto_checksum: Whether to auto-calculate checksums.
        """
        if not script:
            return
        driver = self._require_connected()
        driver.run_job(script, auto_checksum=auto_checksum)

    def set_protect(self, enabled: bool) -> None:
        """Enable or disable protect mode.

        When enabled, the machine will not execute move/cut commands,
        allowing safe dry-run testing.
        """
        driver = self._require_connected()
        driver.set_protect(enabled)

    @property
    def protect_enabled(self) -> bool:
        """Whether protect mode is currently enabled."""
        if self._driver is None:
            return False
        return self._driver.protect_enabled

    # --- Status ---

    @property
    def machine_status(self) -> dict:
        """Current machine status dict."""
        if self._driver is None:
            return {}
        return self._driver.machine_status

    # --- Listeners ---

    def register_status_listener(self, callback: Callable) -> None:
        """Register a status listener.

        Callback signature: callable(status_event: str)
        Status events: CONNECTED, DISCONNECTED, SCRIPT_ERROR, etc.
        """
        driver = self._require_connected()
        driver.register_status_listener(callback)

    def register_error_listener(self, callback: Callable) -> None:
        """Register an error listener.

        Callback signature: callable(error_message: str)
        """
        driver = self._require_connected()
        driver.register_error_listener(callback)

    def register_reply_listener(self, callback: Callable) -> None:
        """Register a reply listener.

        Callback signature: callable(reply_data: bytes)
        """
        driver = self._require_connected()
        driver.register_reply_listener(callback)

    def unregister_status_listener(self) -> None:
        """Unregister the status listener."""
        driver = self._require_connected()
        driver.unregister_status_listener()

    def unregister_error_listener(self) -> None:
        """Unregister the error listener."""
        driver = self._require_connected()
        driver.unregister_error_listener()

    def unregister_reply_listener(self) -> None:
        """Unregister the reply listener."""
        driver = self._require_connected()
        driver.unregister_reply_listener()

    # --- Internal helpers ---

    def _ensure_imported(self) -> None:
        """Raise ImportError if ruidadriver is not available."""
        if RdDriver is None:
            raise ImportError(
                "ruidadriver is not installed. "
                "Run: pixi run -e ruidarpa ..."
            )

    def _require_connected(self) -> RdDriver:
        """Raise RuntimeError if not connected, otherwise return the driver."""
        if self._driver is None or not self._driver.is_connected:
            raise RuntimeError("RPA direct driver is not connected")
        return self._driver
