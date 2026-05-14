from typing import Optional, Any
import struct
import logging
import asyncio

from ...transport.transport import Transport, TransportStatus
from ...transport.udp import UdpTransport
from ...transport.serial import SerialTransport

from .ruida_transcoder import ACK

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

    Key features:
    - Automatic connection using either UDP or Serial transports.
    - Swizzling and unswizzling of data according to the Ruida protocol.
    - Packetizing data with checksums for UDP transport.
    - ACK handshaking for UDP transport.
    - Queued sending of data to optimize throughput.

    When using UDP transport, each packet sent to the Ruida controller is expected
    to be acknowledged with an ACK byte. It is possible to send multiple packets
    without waiting for an ACK.

    It is also possible to package multiple commands in a single packet
    regardless of the transport used.

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

        self.transport: Optional[Transport] = None

        self._urn = '' # host or tty depending upon connected transport.

        self._udp_transport: Optional[UdpTransport] = None
        if self.host:
            self._udp_transport = UdpTransport(self.host, 50200)

        self._serial_transport: Optional[SerialTransport] = None
        if self.tty:
            self._serial_transport = SerialTransport(self.tty, self.baud)

        if not self._udp_transport and not self._serial_transport:
            logger.critical("No valid transport available.")

        self._swizzle_lut = ()
        self._unswizzle_lut = ()
        self._gen_swizzle_luts()

        self._reconnect_interval = 5
        self._connection_task: Optional[asyncio.Task] = None

        self._send_task: Optional[asyncio.Task] = None
        self._send_queue: asyncio.Queue = asyncio.Queue()
        self._acks_expected: int = 0 # Number of ACKs expected

        self._receive_task: Optional[asyncio.Task] = None
        self._expecting_ack = False

        self.keep_running = False

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
        self._swizzle_lut = [self._swizzle_byte(b) for b in range(256)]
        self._unswizzle_lut = [self._unswizzle_byte(b) for b in range(256)]

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

        try:
            # USB first because it has slightly better performance due to
            # the lack of ACK handshakes.
            if _transport is not None:
                await _transport.connect()
                if _transport.is_connected():
                    self.writer = self.reader = _transport
                    self.reader.received.connect(self._on_receive)
                    self._send_task = asyncio.create_task(self._send_loop())
                    self._urn = self.tty if _transport == self._serial_transport else self.host
                    logger.info(f"Successfully connected to {self._urn}.")
                    break
        except Exception:
            # Interface not connected or not configured correctly
            pass
        if not self.is_connected:
            raise

    async def disconnect(self) -> None:
        """Disconnects from the Ruida controller."""
        if self.writer:
            logger.info(f"Disconnecting from controller at {self._urn}...")
            await self.writer.disconnect()
            self.writer = None
            self.reader = None
        else:
            raise ConnectionError("No active transport to disconnect.")

    async def send(self, data: bytes) -> None:
        """Sends data to the connected Ruida controller.

        NOTE: Each call to send() is assumed to be a discrete Ruida command."""
        if not self.is_connected:
            raise ConnectionError("Not connected")
        _pack = self._package(data)
        if self.writer:
            await self._send_queue.put(_pack)
        else:
            raise ConnectionError("No active transport to send data")

    async def send_list(self, commands: list[bytearray]) -> None:
        """Sends a list of commands to the connected Ruida controller.

        Each item in the list is a assumed to be a discrete Ruida command and is
        sent in order.

        Commands are accumulated into a buffer until the either the list is
        empty or the buffer exceeds 1024 bytes in size, at which point the
        buffer is added to the send queue. This repeats until the list has been
        consumed.

        NOTE: Although this method can be used to send discrete commands, it is
        primarily intended to be used for sending Ruida file data in a streaming
        manner.
        """
        if not self.is_connected:
            raise ConnectionError("Not connected")

        while True:
            _buffer = bytearray()
            _i = 0
            while _i < len(commands) and len(_buffer) <= 1024:
                _buffer.extend(commands[_i])
                _i += 1
            if _buffer:
                _pack = self._package(bytes(_buffer))
                _buffer = bytearray()
                if self.writer:
                    await self._send_queue.put(_pack)
                else:
                    raise ConnectionError("No active transport to send data")
            else:
                break

    async def _send_loop(self) -> None:
        """ Send queued packages to the Ruida controller and handle ACKs.
        """
        logger.debug("RuidaTransport send loop started.")
        while self.keep_running:
            try:
                _packet = self._send_queue.get()
                await self.writer.send(_packet)
                if isinstance(self.writer, UdpTransport):
                    self._acks_expected += 1
            except Exception as e:
                logger.error(f"RuidaTransport send loop error: {e}")

    def _on_receive(self, data: bytes) -> None:
        """ Receive packets from the Ruida controller and handle ACKs.

        Only two types of packets are expected from the Ruida controller:
        - ACK packets in response to sent commands.
        - Data packets in response to read memory commands.

        Received data is unswizzled and emitted via the `received` signal.
        """
        logger.debug("RuidaTransport receive loop started.")
        while self.keep_running:
            try:
                data = await self.reader.recvfrom()
                if not data:
                    continue
                if isinstance(self.reader, UdpTransport):
                    if self._acks_expected > 0 and data == ACK:
                        self._acks_expected -= 1
                        continue
                _unswizzled = self._unswizzle(data)
                self.received.send(self, data=_unswizzled)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"RuidaTransport receive loop error: {e}")
