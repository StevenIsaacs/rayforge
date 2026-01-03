from typing import Optional, Any
import struct
import logging
import asyncio

from ...transport.transport import Transport, TransportStatus
from ...transport.udp import UdpTransport
from ...transport.serial import SerialTransport

from .ruida_transcoder import RtcEncoder, RtcDecoder

logger = logging.getLogger(__name__)

class RuidaTransport(Transport):
    """ Automatically connect using either UDP or Serial.

    Handles swizzling and packetizing of data, ACK handshaking, and automatic
    connect and dispatch to an underlying transport which can be either UDP or
    USB/Serial (tty).

    NOTE: A key difference between the UDP and USB/Serial transports when using
    a Ruida controller is that with UDP the controller replies to every command
    with and ACK. On the other hand, there are no ACKs when using a USB/Serial
    transport. Another difference is that UDP packets are preceded with a 16 bit
    unsigned integer checksum and USB/Serial packets have no checksum.

    Attributes:
        host    The IP address or host name of the Ruida controller for a UDP
                connection.
        tty     The device name for a USB/Serial connection. This can be the
                path to a symbolic link. The path must match the pattern
                `/dev/tty*`. NOTE: `/dev/ttyS*` devices are excluded.
        magic   The magic number to use for swizzling and unswizzling data.

        writer  The active send transport.
        reader  The active receive transport.

    """
    def __init__(self, **kwargs: Any):
        super().__init__()
        self.host: str = kwargs.get("host", "")
        self.tty: str = kwargs.get("tty", "")
        self.baud: int = kwargs.get("baud", 115200)
        self.magic: int = kwargs.get("magic", 0x88)

        self.writer: Optional[Transport] = None
        self.reader: Optional[Transport] = None

        self._urn = '' # host or tty depending upon connected transport.

        self._udp_transport: Optional[UdpTransport] = None
        if self.host:
            self._udp_transport = UdpTransport(self.host, 50200)

        self._serial_transport: Optional[SerialTransport] = None
        if self.tty:
            self._serial_transport = SerialTransport(self.tty, self.baud)

        if not self._udp_transport and not self._serial_transport:
            self._udp_transport = UdpTransport(self.tty, self.baud)

        self._swizzle_lut = ()
        self._unswizzle_lut = ()
        self._gen_swizzle_luts()

        self._reconnect_interval = 5
        self._connection_task: Optional[asyncio.Task] = None

    def _swizzle_byte(self, b: int) -> int:
        b ^= (b >> 7) & 0xFF
        b ^= (b << 7) & 0xFF
        b ^= (b >> 7) & 0xFF
        b ^= self.magic
        b = (b + 1) & 0xFF
        return b

    def _unswizzle_byte(self, b: int) -> int:
        b = (b - 1) & 0xFF
        b ^= self.magic
        b ^= (b >> 7) & 0xFF
        b ^= (b << 7) & 0xFF
        b ^= (b >> 7) & 0xFF
        return b

    def _gen_swizzle_luts(self):
        self._swizzle_lut = [self._swizzle_byte(s) for s in range(256)]
        self._unswizzle_lut = [self._unswizzle_byte(s) for s in range(256)]

    def _swizzle(self, data: bytes) -> bytes:
        _r = list()
        for b in data:
            _r.append(self._swizzle_lut[b])
        return bytes(_r)

    def _unswizzle(self, data: bytes) -> bytes:
        _r = list()
        for b in data:
            _r.append(self._unswizzle_lut[b])
        return bytes(_r)

    def _package(self, data: bytes) -> bytes:
        _data = self._swizzle(data)
        # When using UDP all packets are preceded with an unswizzled 16 bit
        # checksum.
        if isinstance(self.writer, UdpTransport):
            return struct.pack(">H", sum(_data) & 0xFFFF) + _data
        else:
            return _data

    @property
    def is_connected(self) -> bool:
        """Check if the transport is actively connected."""
        return self.writer is not None and self.writer.is_connected

    async def connect(self) -> None:
        """Connect to a Ruida controller using either the UDP or Serial
        transports.

        Which transport is used depends upon which interfaces are available
        giving preference to USB if both are available. USB has slightly better
        performance because of the lack of ACK handshake.
        """
        if self.is_connected:
            return

        for _transport in [self._serial_transport, self._udp_transport]:
            try:
                if _transport:
                    await _transport.connect()
                    if _transport.is_connected():
                        self.writer = self.reader = _transport
                        break
            except Exception:
                # Interface not connected or not configured correctly
                pass
        if not self.is_connected:
            raise

    async def disconnect(self) -> None:
        logger.info(
            f"Disconnecting from controller at {self._urn}..."
        )

    async def send(self, data: bytes) -> None:
        if not self.is_connected:
            raise ConnectionError("Not connected")
        _pack = self._package(data)
        await self.writer.send(_pack) # pylance complains but this is OK.
