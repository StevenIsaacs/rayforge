from typing import Callable
from dataclasses import dataclass

from ....pipeline.encoder.base import OpsEncoder, MachineCodeOpMap

# NOTE: Much of the information here is distilled and derived from the
# Ruida Protocol Analyzer and MeerK40t (see README).

# Single byte handshaking.
ACK = 0xCC  # Controller received data and is valid.
ERR = 0xCD  # Controller detected a problem with the message.
ENQ = 0xCE  # Keep alive. This should be replied to with a corresponding
            # ENQ or ACK.
NAK = 0xCF  # Message checksum mismatch. Resend packet.

CMD_MASK = 0x80 # Only the first byte of a command has the top bit set.
                # This, I'm guessing, is the primary reason a 7 bit
                # protocol is used.
                # This can be used to re-sync or to check the length of
                # data associated with a command for validity.
EOF = 0xD7      # Indicates the end of the Ruida file and the checksum will
                # follow.

MEM_CARD_ID = b"\x05\x7E"
MEM_BED_SIZE_X = b"\x00\x26"
MEM_BED_SIZE_Y = b"\x00\x36"
MEM_MACHINE_STATUS = b"\x04\x00"
MEM_CURRENT_X = b"\x04\x21"
MEM_CURRENT_Y = b"\x04\x31"

@dataclass
class RtcMem:
    decoder: Callable
    var_name: str
    desc: str

MACHINE_STATUS_MOVING = 0x01000000
MACHINE_STATUS_PART_END = 0x00000002
MACHINE_STATUS_JOB_RUNNING = 0x00000001

# Command table.
CT = {
    "CMD_READ_MEM": b"\xDA\x00",
}

# Reply table.
RT = {
    "RPY_READ_MEM": b"\xDA\x01",
}

class RtcEncoder(OpsEncoder):
    """Encode commands and their parameters into bytes."""
    def __init__(self):
        pass

    def int_to_i35(self, v: int) -> bytes:
        return bytes(
            [
                (v >> 28) & 0x7F,
                (v >> 21) & 0x7F,
                (v >> 14) & 0x7F,
                (v >> 7) & 0x7F,
                v & 0x7F,
            ]
        )

    def int_to_i14(self, v: int) -> bytes:
        return bytes(
            [
                (v >> 7) & 0x7F,
                v & 0x7F,
            ]
        )

class RtcDecoder:
    """Decode bytes into values.

    This is used primarily for decoding memory read data.
    """
    # Known card IDs. This is indexed using the reply from a MEM_CARD_ID read.
    CID_LUT = {
        0x65106510: "RDC6442S",
    }

    def __init__(self):
        self.card_id = 0
        self.m_status = 0
        self.bed_x = 0
        self.bed_y = 0
        self.x_pos = 0
        self.y_pos = 0

    def _to_int(self, data: bytes, n_bytes=0) -> int:
        if not n_bytes:
            _n = len(data)
        else:
            _n = n_bytes
        _v = 0
        _m = 0 # For 2's complement later.

        for _i in range(_n):
            _b = data[_i]
            if _i == 0:
                _b &= 0x3F
            _v = (_v << 7) + _b
            _m = (_m << 7) + 0x7F
        if data[0] & 0x40:
            # Two's complement -- sorta.
            _v = ((~_v & (_m >> 1)) + 1) * -1

        return _v

    def _to_uint(self, data: bytes, n_bytes=0) -> int:
        if not n_bytes:
            _n = len(data)
        else:
            _n = n_bytes
        _v = 0
        for _i in range(_n):
            _v = (_v << 7) + data[_i]
        return _v

    def int35_to_int(self, data: bytes) -> int:
        return self._to_int(data, 5)

    def uint35_to_uint(self, data: bytes) -> int:
        return self._to_uint(data, 5)

    MT = {
        MEM_CARD_ID: RtcMem(
            uint35_to_uint, "card_id", "Ruida card ID: 0x{:08X} {}"),
        MEM_BED_SIZE_X: RtcMem(
            int35_to_int, "bed_x", "Bed X: {:.3f}mm"),
        MEM_BED_SIZE_Y: RtcMem(
            int35_to_int, "bed_y", "Bed Y: {:.3f}mm"),
        MEM_CURRENT_X: RtcMem(
            int35_to_int, "x_pos", "X Pos: {:.3f}mm"),
        MEM_CURRENT_Y: RtcMem(
            int35_to_int, "y_pos", "Y Pos: {:.3f}mm"),
        MEM_MACHINE_STATUS: RtcMem(
            uint35_to_uint, "m_status", "Machine: 0x{:08X}"),
    }

    @property
    def is_moving(self) -> bool:
        """True when the laser head is in motion following a jog or absolute
        position or physical home command."""
        if self.m_status & MACHINE_STATUS_MOVING:
            return True
        return False

    @property
    def is_part_end(self) -> bool:
        """True when running a job and a part (layer) has been completed."""
        if self.m_status & MACHINE_STATUS_PART_END:
            return True
        return False

    @property
    def is_job_running(self):
        """True when a job is running."""
        if self.m_status & MACHINE_STATUS_JOB_RUNNING:
            return True
        return False

    def decode_reply(self, data: bytes) -> tuple[int, str]:
        """Decode a memory read reply packet.

        The packet must be at least 9 bytes in length and is assumed to have the
        format:
          2 bytes: Indicating a reply to a memory read (0xDA, 0x01).
          2 bytes: The controller address of the data.
          5 bytes: A 35 (5 * 7) bit value read at the address.
        """
        if len(data) < 9:
            raise ValueError("Insufficient data for memory reply.")
        _msb = data[0]
        _lsb = data[1]
        if _msb == 0xDA and _lsb == 0x01:
            # Is a reply to a memory read.
            _addr = data[2:4] # Memory address.
            if _addr in self.MT:
                _v = self.MT[_addr].decoder(data[4:9])
                setattr(self.__class__, self.MT[_addr].var_name, _v)
                if _addr == MEM_CARD_ID:
                    if _v in self.CID_LUT:
                        _s = self.MT[_addr].desc.format(_v, self.CID_LUT[_v])
                        return _v, _s
                    else:
                        return _v, f"Unknown card: {_v:08X}"
                return _v, self.MT[_addr].desc.format(_v)
            _v = self.uint35_to_uint(data[4:9])
            return _v, f"Unknown memory address: 0x{_v:08X}"
        raise ValueError(f"0x{_msb:02X}{_lsb:02X} is not a reply.")