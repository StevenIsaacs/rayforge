"""
RPA Encoder - Produces rpascript output for Ruida Protocol Analyzer driver.

Rpascript is the native command format for RdDriver.
Coordinates use mm natively (no unit conversion needed).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

from raygeo.ops.types import CommandType

from rayforge.pipeline.encoder.base import (
    EncodedOutput,
    MachineCodeOpMap,
    OpsEncoder,
)

if TYPE_CHECKING:
    from rayforge.core.doc import Doc
    from rayforge.machine.models.machine import Machine

logger = logging.getLogger(__name__)


class RuidaRPAEncoder(OpsEncoder):
    """Encodes Ops commands into rpascript text format.

    Rpascript is human-readable and uses mm natively — no unit conversion
    is needed. State is tracked to avoid emitting redundant config commands.
    """

    def __init__(self) -> None:
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset all encoder state for a new encoding session."""
        self.power: Optional[float] = None
        self.cut_speed: Optional[float] = None
        self.travel_speed: Optional[float] = None
        self.current_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.active_laser: int = 1
        self.air_assist: bool = False
        self.lines: List[str] = []
        self.op_map: Optional[MachineCodeOpMap] = None

    def encode(
        self, ops, machine: "Machine", doc: "Doc"
    ) -> EncodedOutput:
        """Encode Ops commands into rpascript format.

        Args:
            ops: Ops object from raygeo containing commands to encode.
            machine: The machine configuration (used for laser head lookup).
            doc: The document being processed.

        Returns:
            EncodedOutput with rpascript text, op_map, and no driver_data.
        """
        self._reset_state()
        self.op_map = MachineCodeOpMap()

        for i in range(ops.len()):
            start_line = len(self.lines)
            self._handle_command(ops, i, machine)
            end_line = len(self.lines)

            if end_line > start_line:
                line_indices = list(range(start_line, end_line))
                self.op_map.op_to_machine_code[i] = line_indices
                for line_num in line_indices:
                    self.op_map.machine_code_to_op[line_num] = i
            else:
                self.op_map.op_to_machine_code[i] = []

        text = "\n".join(self.lines)
        return EncodedOutput(text=text, op_map=self.op_map)

    # -- Command dispatch ---------------------------------------------------

    def _handle_command(
        self, ops, idx: int, machine: "Machine"
    ) -> None:
        """Dispatch a single Ops command to the appropriate handler.

        Args:
            ops: The Ops object.
            idx: Index of the command within ops.
            machine: Machine configuration for laser head resolution.
        """
        ct = ops.command_type(idx)

        if ct == CommandType.SET_POWER:
            self._handle_set_power(ops, idx)
        elif ct == CommandType.SET_CUT_SPEED:
            self._handle_set_cut_speed(ops, idx)
        elif ct == CommandType.SET_TRAVEL_SPEED:
            self._handle_set_travel_speed(ops, idx)
        elif ct == CommandType.SET_FREQUENCY:
            self._handle_set_frequency(ops, idx)
        elif ct == CommandType.SET_PULSE_WIDTH:
            self._handle_set_pulse_width(ops, idx)
        elif ct == CommandType.ENABLE_AIR_ASSIST:
            self._handle_enable_air_assist()
        elif ct == CommandType.DISABLE_AIR_ASSIST:
            self._handle_disable_air_assist()
        elif ct == CommandType.SET_LASER:
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
            self._handle_job_end()
        elif ct == CommandType.LAYER_START:
            self._handle_layer_start(ops, idx)
        elif ct == CommandType.LAYER_END:
            self._handle_layer_end()
        elif ct == CommandType.WORKPIECE_START:
            self._handle_workpiece_start(ops, idx)
        elif ct == CommandType.WORKPIECE_END:
            self._handle_workpiece_end()
        elif ct == CommandType.OPS_SECTION_START:
            self._handle_ops_section_start()
        elif ct == CommandType.OPS_SECTION_END:
            self._handle_ops_section_end()
        else:
            raise ValueError(f"Unknown command type: {ct}")

    # -- Helpers ------------------------------------------------------------

    def _emit(self, lines: List[str]) -> None:
        """Append one or more lines to the output."""
        self.lines.extend(lines)

    # -- Movement handlers --------------------------------------------------

    def _handle_move_to(self, ops, idx: int) -> None:
        """Rapid move (laser off) to an absolute position."""
        x, y, z = ops.endpoint(idx)
        self.current_pos = (x, y, z)
        self._emit([f"MOVE_ABS_XY X={x:.3f}mm Y={y:.3f}mm"])

    def _handle_line_to(self, ops, idx: int) -> None:
        """Cutting move (laser on) to an absolute position."""
        x, y, z = ops.endpoint(idx)
        self.current_pos = (x, y, z)
        self._emit([f"CUT_ABS_XY X={x:.3f}mm Y={y:.3f}mm"])

    def _handle_arc_to(self, ops, idx: int) -> None:
        """Linearize arc to cut segments.

        Rpascript does not have a native arc command, so arcs are
        decomposed into CUT_ABS_XY segments via linearize().
        """
        start_pos = self.current_pos
        end = ops.endpoint(idx)

        sub_ops = ops.linearize(idx, start_pos)
        for j in range(sub_ops.len()):
            sub_ct = sub_ops.command_type(j)
            if sub_ct == CommandType.LINE_TO:
                sx, sy, sz = sub_ops.endpoint(j)
                self._emit([f"CUT_ABS_XY X={sx:.3f}mm Y={sy:.3f}mm"])
            elif sub_ct == CommandType.SET_POWER:
                power_norm = sub_ops.power(j)
                self._emit_set_power_line(power_norm)

        self.current_pos = end

    def _handle_scan_line(self, ops, idx: int) -> None:
        """Linearize scan line to power and cut segments."""
        start_pos = self.current_pos
        end = ops.endpoint(idx)

        sub_ops = ops.linearize(idx, start_pos)
        for j in range(sub_ops.len()):
            sub_ct = sub_ops.command_type(j)
            if sub_ct == CommandType.LINE_TO:
                sx, sy, sz = sub_ops.endpoint(j)
                self._emit([f"CUT_ABS_XY X={sx:.3f}mm Y={sy:.3f}mm"])
            elif sub_ct == CommandType.SET_POWER:
                power_norm = sub_ops.power(j)
                self._emit_set_power_line(power_norm)

        self.current_pos = end

    def _handle_dwell(self, ops, idx: int) -> None:
        """Emit a dwell (pause) command.

        Rpascript DELAY accepts time in seconds or milliseconds.
        """
        duration_ms = ops.dwell_duration(idx)
        self._emit([f"DELAY {duration_ms:.3f}ms"])

    def _handle_bezier_to(self, ops, idx: int) -> None:
        """Linearize cubic bezier curve to cut segments."""
        start_pos = self.current_pos
        end = ops.endpoint(idx)

        sub_ops = ops.linearize(idx, start_pos)
        for j in range(sub_ops.len()):
            sub_ct = sub_ops.command_type(j)
            if sub_ct == CommandType.LINE_TO:
                sx, sy, sz = sub_ops.endpoint(j)
                self._emit([f"CUT_ABS_XY X={sx:.3f}mm Y={sy:.3f}mm"])
            elif sub_ct == CommandType.SET_POWER:
                power_norm = sub_ops.power(j)
                self._emit_set_power_line(power_norm)

        self.current_pos = end

    def _handle_quadratic_bezier_to(self, ops, idx: int) -> None:
        """Linearize quadratic bezier curve to cut segments."""
        start_pos = self.current_pos
        end = ops.endpoint(idx)

        sub_ops = ops.linearize(idx, start_pos)
        for j in range(sub_ops.len()):
            sub_ct = sub_ops.command_type(j)
            if sub_ct == CommandType.LINE_TO:
                sx, sy, sz = sub_ops.endpoint(j)
                self._emit([f"CUT_ABS_XY X={sx:.3f}mm Y={sy:.3f}mm"])
            elif sub_ct == CommandType.SET_POWER:
                power_norm = sub_ops.power(j)
                self._emit_set_power_line(power_norm)

        self.current_pos = end

    # -- Configuration handlers ---------------------------------------------

    def _emit_set_power_line(self, power_norm: float) -> None:
        """Emit a power command line and update tracked state."""
        power_pct = power_norm * 100.0
        self.power = power_pct
        self._emit([f"IMD_POWER_1 Power={power_pct:.1f}%"])

    def _handle_set_power(self, ops, idx: int) -> None:
        """Set laser power, skipping redundant values."""
        power_norm = ops.power(idx)
        power_pct = power_norm * 100.0
        if self.power is not None and abs(power_pct - self.power) < 0.01:
            return
        self._emit_set_power_line(power_norm)

    def _handle_set_cut_speed(self, ops, idx: int) -> None:
        """Set cutting speed, skipping redundant values."""
        speed = float(ops.speed(idx))
        if self.cut_speed is not None and abs(speed - self.cut_speed) < 0.001:
            return
        self.cut_speed = speed
        self._emit([f"SPEED_LASER_1 Speed={speed:.3f}mm/S"])

    def _handle_set_travel_speed(self, ops, idx: int) -> None:
        """Set travel (rapid move) speed, skipping redundant values."""
        speed = float(ops.speed(idx))
        if (
            self.travel_speed is not None
            and abs(speed - self.travel_speed) < 0.001
        ):
            return
        self.travel_speed = speed
        self._emit([f"SPEED_AXIS Speed={speed:.3f}mm/S"])

    def _handle_set_frequency(self, ops, idx: int) -> None:
        """Set laser frequency (Hz → KHz)."""
        freq_hz = ops.frequency(idx)
        freq_khz = freq_hz / 1000.0
        self._emit([
            f"FREQUENCY_PART Laser={self.active_laser}"
            f" Part=1 Freq={freq_khz:.3f}KHz"
        ])

    def _handle_set_pulse_width(self, ops, idx: int) -> None:
        """Set laser pulse width (µs → mS)."""
        pw_us = ops.pulse_width(idx)
        pw_ms = pw_us / 1000.0
        self._emit([f"LASER_INTERVAL {pw_ms:.3f}mS"])

    def _handle_enable_air_assist(self) -> None:
        """Enable air assist, skipping if already on."""
        if self.air_assist:
            return
        self.air_assist = True
        self._emit(["AIR_ASSIST_ON"])

    def _handle_disable_air_assist(self) -> None:
        """Disable air assist, skipping if already off."""
        if not self.air_assist:
            return
        self.air_assist = False
        self._emit(["AIR_ASSIST_OFF"])

    def _handle_set_laser(
        self, ops, idx: int, machine: "Machine"
    ) -> None:
        """Select laser device by resolving laser_uid to a tool number.

        Attempts to find the laser head in machine.heads. Falls back to
        extracting the trailing numeric suffix from laser_uid and modding
        by 2 if the head is not found or heads are unavailable.
        """
        laser_uid = ops.laser_uid(idx)
        # laser_uid is a string like "laser_42" or a UUID — extract a
        # deterministic device number (0 or 1) for dual-laser setups
        try:
            device = int(laser_uid.split("_")[-1]) % 2
        except (ValueError, IndexError):
            # Non-numeric UID (e.g., UUID) — use simple hash
            device = sum(ord(c) for c in laser_uid) % 2

        try:
            laser_head = next(
                (
                    head
                    for head in machine.heads
                    if head.uid == laser_uid
                ),
                None,
            )
            if laser_head is not None:
                device = laser_head.tool_number - 1
        except (AttributeError, TypeError):
            logger.debug(
                "machine.heads not available, falling back to "
                "parsed laser_uid %% 2 for "
                "laser device selection"
            )

        if device == self.active_laser:
            return
        self.active_laser = device
        self._emit([f"LASER_DEVICE_{device}"])

    # -- Structural handlers ------------------------------------------------

    def _handle_job_start(self) -> None:
        """Emit job start framing.

        SET_ABSOLUTE establishes absolute coordinate mode.
        START_PROCESS begins the processing block.
        """
        self._emit(["SET_ABSOLUTE", "START_PROCESS"])

    def _handle_job_end(self) -> None:
        """Emit job end framing.

        BLOCK_END closes the processing block.
        SET_FILE_SUM marks the file for checksum calculation by the runner.
        """
        self._emit(["BLOCK_END", "SET_FILE_SUM"])

    def _handle_layer_start(self, ops, idx: int) -> None:
        """Emit a layer start marker."""
        layer_uid = ops.layer_uid(idx)
        self._emit([f"LAYER_START uid={layer_uid}"])

    def _handle_layer_end(self) -> None:
        """Emit a layer end marker."""
        self._emit(["LAYER_END"])

    def _handle_workpiece_start(self, ops, idx: int) -> None:
        """Emit a workpiece start marker."""
        wp_uid = ops.workpiece_uid(idx)
        self._emit([f"WORKPIECE_START uid={wp_uid}"])

    def _handle_workpiece_end(self) -> None:
        """Emit a workpiece end marker."""
        self._emit(["WORKPIECE_END"])

    def _handle_ops_section_start(self) -> None:
        """Emit nothing for ops section start.

        Section framing is handled by JOB_START/JOB_END.
        """
        pass

    def _handle_ops_section_end(self) -> None:
        """Emit nothing for ops section end.

        Section framing is handled by JOB_START/JOB_END.
        """
        pass
