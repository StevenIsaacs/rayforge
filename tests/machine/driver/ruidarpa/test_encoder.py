"""
Test suite for the RuidaRPAEncoder.

The encoder drives the ruida-pa GlueScript API to produce a GlueScript
transcript (the source); the backend compiles it to rpascript via
stage_gluescript when the job runs. Tests cover:
- Job framing (declare_job / end_job)
- Layer declaration from workflow steps (settings, defaults, raw power)
- Move and cut transcript lines
- Configuration actions (power, speed, frequency, pulse width, air assist)
- Curve linearization (arcs, scan lines)
- Bidirectional op_map generation
- Error handling (missing JOB_END, unknown commands, missing layer)
"""

import ast
import logging
from unittest.mock import Mock

import pytest
from raygeo.ops import Ops
from raygeo.ops.state import AirAssistMode, CoolantMode
from raygeo.ops.types import RasterMode, SectionType
from ruidadriver.rd_gluescript import GlueScript

from rayforge.core.doc import Doc
from rayforge.core.step import Step
from rayforge.machine.driver.ruidarpa import rpa_encoder
from rayforge.machine.driver.ruidarpa.rpa_encoder import RuidaRPAEncoder
from rayforge.machine.models.laser import Laser
from rayforge.pipeline.encoder.base import EncodedOutput, MachineCodeOpMap


class CutStep(Step):
    """Minimal concrete Step for encoder layer-settings tests.

    Mirrors the laser step's process attributes (power, cut_speed,
    frequency) that the encoder reads from the first workflow step.
    """

    def __init__(self):
        super().__init__(typelabel="cut")
        self.power: float = 0.0
        self.frequency: int = 0


@pytest.fixture
def encoder():
    """Provides a fresh RuidaRPAEncoder instance."""
    return RuidaRPAEncoder()


@pytest.fixture
def mock_machine(isolated_machine):
    """Provides a machine with two laser heads for testing."""
    laser1 = Laser()
    laser1.uid = "laser-1"
    laser1.tool_number = 1

    laser2 = Laser()
    laser2.uid = "laser-2"
    laser2.tool_number = 2

    isolated_machine.heads.clear()
    isolated_machine.add_head(laser1)
    isolated_machine.add_head(laser2)
    return isolated_machine


@pytest.fixture
def doc():
    """Provides a fresh Doc instance (3 default layers)."""
    return Doc()


def _declare_layer_mode(line: str) -> str:
    """Extract the mode argument from a declare_layer transcript line."""
    args = ast.literal_eval(line[len("declare_layer(") : -1])
    return args[2]


def _declare_layer_overscan(line: str) -> str:
    """Extract the overscan argument from a declare_layer transcript line."""
    args = ast.literal_eval(line[len("declare_layer(") : -1])
    return args[3]


class TestRuidaRPAEncoderBasics:
    """Basic encoder functionality tests."""

    def test_encode_returns_encoded_output(self, encoder, mock_machine, doc):
        """Verify encode() returns an EncodedOutput instance."""
        ops = Ops()
        result = encoder.encode(ops, mock_machine, doc)

        assert isinstance(result, EncodedOutput)
        assert isinstance(result.text, str)
        assert isinstance(result.op_map, MachineCodeOpMap)

    def test_empty_ops_produces_empty_output(self, encoder, mock_machine, doc):
        """Empty Ops should produce empty text and op_map."""
        ops = Ops()
        result = encoder.encode(ops, mock_machine, doc)

        assert result.text == ""
        assert result.op_map.op_count == 0
        assert result.op_map.line_count == 0

    def test_encoder_state_resets_between_encodes(
        self, encoder, mock_machine, doc
    ):
        """Each encode() call should reset internal state."""
        ops1 = Ops()
        ops1.job_start()
        ops1.layer_start(layer_uid=doc.layers[0].uid)
        ops1.workpiece_start("wp-0")
        ops1.set_power(0.5)
        ops1.move_to(0.0, 0.0, 0.0)
        ops1.workpiece_end("wp-0")
        ops1.layer_end(layer_uid=doc.layers[0].uid)
        ops1.job_end()
        encoder.encode(ops1, mock_machine, doc)

        ops2 = Ops()
        ops2.job_start()
        ops2.layer_start(layer_uid=doc.layers[0].uid)
        ops2.workpiece_start("wp-0")
        ops2.move_to(0.0, 0.0, 0.0)
        ops2.workpiece_end("wp-0")
        ops2.layer_end(layer_uid=doc.layers[0].uid)
        ops2.job_end()
        result2 = encoder.encode(ops2, mock_machine, doc)

        assert encoder.active_laser == 1
        # Second job: 0=job_start, 1=layer_start, 2=workpiece_start,
        # 3=move_to, 4=workpiece_end, 5=layer_end, 6=job_end
        assert result2.op_map.op_count == 7
        assert result2.op_map.span_for_op(5) == (0, 0)


class TestJobStructure:
    """Tests for job framing."""

    def _simple_job(self, doc):
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.move_to(5.0, 5.0, 0.0)
        ops.line_to(10.0, 8.0, 0.0)
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        return ops

    def test_job_framing(self, encoder, mock_machine, doc):
        """The transcript is framed by declare_job and end_job."""
        result = encoder.encode(self._simple_job(doc), mock_machine, doc)
        lines = result.text.split("\n")

        assert lines[0].startswith("declare_job(")
        assert lines[-1] == "end_job()"


class TestLayerDeclaration:
    """Tests for layer attribute declaration."""

    def test_default_layer_settings(self, encoder, mock_machine, doc):
        """Layers without workflow steps should use safe defaults."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        assert (
            "declare_layer('Layer 1', '#00ccff', 'VECTOR', 'NONE', "
            "100.0, 20.0, 20.0, 20.0)" in result.text
        )

    def test_layer_settings_from_step(self, encoder, mock_machine, doc):
        """Layer attributes should come from the first workflow step."""
        step = CutStep()
        step.power = 0.5
        step.cut_speed = 300
        doc.layers[0].workflow.add_step(step)

        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        assert (
            "declare_layer('Layer 1', '#00ccff', 'VECTOR', 'NONE', "
            "5.0, 20.0, 50.0, 50.0)" in result.text
        )

    def test_power_below_minimum_raises_from_gluescript(
        self, encoder, mock_machine, doc
    ):
        """Power below the 8% controller minimum must raise in GlueScript.

        The encoder passes the raw power through; GlueScript rejects it.
        """
        step = CutStep()
        step.power = 0.05
        doc.layers[0].workflow.add_step(step)

        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()

        with pytest.raises(ValueError):
            encoder.encode(ops, mock_machine, doc)

    def test_unknown_layer_uses_defaults(self, encoder, mock_machine, doc):
        """Layers absent from the document should still stage cleanly."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid="missing-layer-uid")
        ops.workpiece_start("wp-0")
        ops.move_to(1.0, 1.0, 0.0)
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid="missing-layer-uid")
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        assert any(
            line.startswith("declare_layer(")
            for line in result.text.split("\n")
        )

    def test_multi_workpiece_layer_declares_each_workpiece(
        self, encoder, mock_machine, doc
    ):
        """One layer with two workpieces emits two declare_layer lines."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.move_to(1.0, 1.0, 0.0)
        ops.workpiece_end("wp-0")
        ops.workpiece_start("wp-1")
        ops.move_to(2.0, 2.0, 0.0)
        ops.workpiece_end("wp-1")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        declared = [
            line
            for line in result.text.split("\n")
            if line.startswith("declare_layer(")
        ]
        assert len(declared) == 2
        assert all("'Layer 1'" in line for line in declared)
        assert all("'#00ccff'" in line for line in declared)

    def test_per_workpiece_mode_from_sections(
        self, encoder, mock_machine, doc
    ):
        """Two workpieces in one layer derive their own declare_layer modes."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.ops_section_start(
            SectionType.RASTER_FILL,
            "wp-0",
            raster_mode=RasterMode.VARIABLE_POWER,
        )
        ops.set_power(0.5)
        ops.ops_section_end(
            SectionType.RASTER_FILL, raster_mode=RasterMode.VARIABLE_POWER
        )
        ops.workpiece_end("wp-0")
        ops.workpiece_start("wp-1")
        ops.set_power(0.5)
        ops.workpiece_end("wp-1")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        declared = [
            line
            for line in result.text.split("\n")
            if line.startswith("declare_layer(")
        ]
        assert _declare_layer_mode(declared[0]) == "IMAGE"
        assert _declare_layer_mode(declared[1]) == "VECTOR"


class TestMoveCutForms:
    """Tests for move and cut transcript lines."""

    def test_near_move_uses_near_form(self, encoder, mock_machine, doc):
        """Small moves should use the near form."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.move_to(5.0, 5.0, 0.0)
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        assert "move_xy_to(5.0, 5.0)" in result.text

    def test_far_move_uses_far_form(self, encoder, mock_machine, doc):
        """Moves beyond the 8.192mm threshold must use the far form."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.move_to(0.0, 0.0, 0.0)
        ops.move_to(150.0, 0.0, 0.0)
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        assert "move_xy_to(150.0, 0.0)" in result.text

    def test_near_cut_uses_near_form(self, encoder, mock_machine, doc):
        """Small cuts should use the near form."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.move_to(0.0, 0.0, 0.0)
        ops.line_to(5.0, 5.0, 0.0)
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        assert "cut_xy_to(5.0, 5.0)" in result.text

    def test_far_cut_uses_far_form(self, encoder, mock_machine, doc):
        """Cuts beyond the 8.192mm threshold must use the far form."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.move_to(0.0, 0.0, 0.0)
        ops.line_to(20.0, 0.0, 0.0)
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        assert "cut_xy_to(20.0, 0.0)" in result.text


class TestSettingsCommands:
    """Tests for configuration action commands."""

    def test_power_emits_min_max_lines(self, encoder, mock_machine, doc):
        """SET_POWER should emit a power_range transcript line."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.set_power(0.5)
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        assert "power_range(50.0, 50.0)" in result.text

    def test_power_action_below_minimum_raises_from_gluescript(
        self, encoder, mock_machine, doc
    ):
        """Per-op SET_POWER below 8% must raise in GlueScript.

        The encoder passes the raw power through; GlueScript rejects it.
        """
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.set_power(0.05)
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()

        with pytest.raises(ValueError):
            encoder.encode(ops, mock_machine, doc)

    def test_legacy_coolant_non_off_logs_warning(
        self, encoder, mock_machine, doc, caplog
    ):
        """Legacy SET_COOLANT must warn instead of silently dropping."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.set_coolant(CoolantMode.FLOOD)
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        assert "air_assist_on()" not in result.text
        assert any(
            "SET_COOLANT" in record.message for record in caplog.records
        )

    def test_feed_rate_emits_speed_line(self, encoder, mock_machine, doc):
        """SET_FEED_RATE should emit a cut_speed transcript line."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.set_feed_rate(200)
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        assert "cut_speed(3.3333333333333335)" in result.text

    def test_rapid_rate_emits_axis_speed(self, encoder, mock_machine, doc):
        """SET_RAPID_RATE should emit a move_speed transcript line."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.set_rapid_rate(500)
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        assert "move_speed(8.333333333333334)" in result.text

    def test_frequency_emits_khz_line(self, encoder, mock_machine, doc):
        """SET_FREQUENCY should convert Hz to KHz for gluescript."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.set_frequency(20000)
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        assert "frequency(20.0)" in result.text

    def test_pulse_width_emits_interval_line(self, encoder, mock_machine, doc):
        """SET_PULSE_WIDTH should pass microseconds to gluescript pwm."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.set_pulse_width(50)
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        assert "pwm(50.0)" in result.text

    def test_dwell_warns_and_emits_no_delay(
        self, encoder, mock_machine, doc, caplog
    ):
        """DWELL is unsupported — warn and emit no delay line."""
        caplog.set_level(logging.WARNING, logger=rpa_encoder.logger.name)
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.dwell(250)
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        assert "delay" not in result.text
        assert any("DWELL" in record.message for record in caplog.records)

    def test_air_assist_on_off(self, encoder, mock_machine, doc):
        """Air assist toggle should emit ON then OFF lines."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.set_air_assist(AirAssistMode.ON)
        ops.move_to(0.0, 0.0, 0.0)
        ops.set_air_assist(AirAssistMode.OFF)
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        lines = result.text.split("\n")
        assert "air_assist_on()" in lines
        assert "air_assist_off()" in lines
        assert lines.index("air_assist_on()") < lines.index("air_assist_off()")

    def test_set_head_selects_laser_device(
        self, encoder, mock_machine, doc, caplog
    ):
        """SET_HEAD resolves the laser_uid to a device via select_laser."""
        caplog.set_level(logging.WARNING, logger=rpa_encoder.logger.name)
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.set_head("laser-2")
        ops.set_power(0.5)
        ops.set_head("laser-1")
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        lines = result.text.split("\n")
        # laser-2 resolves to device 2: select_laser(2) is recorded into
        # the transcript, but no raw LASER_DEVICE_2 (only laser 1 is wired
        # in ruida-pa), and a warning is logged.
        assert "select_laser(2)" in lines
        assert any(
            "select_laser" in record.message for record in caplog.records
        )
        # laser-1 resolves to device 1: select_laser(1) is recorded.
        assert "select_laser(1)" in lines

    def test_set_head_numeric_suffix_fallback_selects_device(
        self, encoder, mock_machine, doc
    ):
        """SET_HEAD with a numeric suffix derives the device number."""
        mock_machine.heads.clear()

        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.set_head("laser_2")
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        # ((2 - 1) % 2) + 1 = 2; active_laser defaults to 1
        assert "select_laser(2)" in result.text

        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.set_head("laser_1")
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        # ((1 - 1) % 2) + 1 = 1, no switch needed
        assert "select_laser(1)" not in result.text

    def test_set_head_char_sum_fallback_selects_device(
        self, encoder, mock_machine, doc
    ):
        """SET_HEAD without a numeric suffix falls back to char sums."""
        mock_machine.heads.clear()

        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.set_head("laser-3")
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        # sum(ord(c) for c in "laser-3") = 631, odd -> device 2
        assert "select_laser(2)" in result.text


class TestSectionPowerRouting:
    """Section-aware SET_POWER routing to GlueScript power/power_range."""

    def _raster_job(self, doc, raster_mode, power):
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.ops_section_start(
            SectionType.RASTER_FILL, "wp-0", raster_mode=raster_mode
        )
        ops.set_power(power)
        ops.ops_section_end(SectionType.RASTER_FILL, raster_mode=raster_mode)
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        return ops

    def test_variable_power_section_uses_power_gluescript(
        self, encoder, mock_machine, doc
    ):
        """RASTER_FILL + VARIABLE_POWER must route through power()."""
        ops = self._raster_job(doc, RasterMode.VARIABLE_POWER, 0.5)
        result = encoder.encode(ops, mock_machine, doc)

        assert "power(50.0)" in result.text

    def test_variable_power_section_passes_low_power_through(
        self, encoder, mock_machine, doc, caplog
    ):
        """Image power below 8% must pass through without clamping."""
        ops = self._raster_job(doc, RasterMode.VARIABLE_POWER, 0.05)
        result = encoder.encode(ops, mock_machine, doc)

        assert "power(5.0)" in result.text
        assert not any(
            "clamping" in record.message for record in caplog.records
        )

    def test_depth_map_section_uses_power_gluescript(
        self, encoder, mock_machine, doc
    ):
        """RASTER_FILL + DEPTH_MAP must route through power()."""
        ops = self._raster_job(doc, RasterMode.DEPTH_MAP, 0.5)
        result = encoder.encode(ops, mock_machine, doc)

        assert "power(50.0)" in result.text

    def test_constant_power_section_uses_power_range(
        self, encoder, mock_machine, doc
    ):
        """RASTER_FILL + CONSTANT_POWER must use power_range()."""
        ops = self._raster_job(doc, RasterMode.CONSTANT_POWER, 0.5)
        result = encoder.encode(ops, mock_machine, doc)

        assert "power_range(50.0, 50.0)" in result.text

    def test_layer_mode_derived_from_sections(
        self, encoder, mock_machine, doc
    ):
        """declare_layer mode follows the layer's ops sections."""
        cases = [
            (RasterMode.VARIABLE_POWER, "IMAGE"),
            (RasterMode.DEPTH_MAP, "DEPTHMAP"),
            (RasterMode.CONSTANT_POWER, "RASTER"),
            (None, "VECTOR"),
        ]
        for raster_mode, expected in cases:
            ops = Ops()
            ops.job_start()
            ops.layer_start(layer_uid=doc.layers[0].uid)
            ops.workpiece_start("wp-0")
            if raster_mode is None:
                ops.set_power(0.5)
            else:
                ops.ops_section_start(
                    SectionType.RASTER_FILL,
                    "wp-0",
                    raster_mode=raster_mode,
                )
                ops.set_power(0.5)
                ops.ops_section_end(
                    SectionType.RASTER_FILL, raster_mode=raster_mode
                )
            ops.workpiece_end("wp-0")
            ops.layer_end(layer_uid=doc.layers[0].uid)
            ops.job_end()
            result = encoder.encode(ops, mock_machine, doc)
            declared = [
                line
                for line in result.text.split("\n")
                if line.startswith("declare_layer(")
            ]
            assert _declare_layer_mode(declared[0]) == expected

    def test_layer_mode_priority_depth_map_wins_regardless_of_order(
        self, encoder, mock_machine, doc
    ):
        """DEPTH_MAP wins over VARIABLE_POWER even when it comes last."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.ops_section_start(
            SectionType.RASTER_FILL,
            "wp-0",
            raster_mode=RasterMode.VARIABLE_POWER,
        )
        ops.set_power(0.5)
        ops.ops_section_end(
            SectionType.RASTER_FILL, raster_mode=RasterMode.VARIABLE_POWER
        )
        ops.ops_section_start(
            SectionType.RASTER_FILL,
            "wp-0",
            raster_mode=RasterMode.DEPTH_MAP,
        )
        ops.set_power(0.5)
        ops.ops_section_end(
            SectionType.RASTER_FILL, raster_mode=RasterMode.DEPTH_MAP
        )
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        declared = [
            line
            for line in result.text.split("\n")
            if line.startswith("declare_layer(")
        ]
        assert _declare_layer_mode(declared[0]) == "DEPTHMAP"

    def test_section_state_resets_after_section_end(
        self, encoder, mock_machine, doc
    ):
        """SET_POWER after OPS_SECTION_END must fall back to power_range()."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.ops_section_start(
            SectionType.RASTER_FILL,
            "wp-0",
            raster_mode=RasterMode.VARIABLE_POWER,
        )
        ops.ops_section_end(
            SectionType.RASTER_FILL, raster_mode=RasterMode.VARIABLE_POWER
        )
        ops.set_power(0.5)
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        lines = result.text.split("\n")
        assert "power_range(50.0, 50.0)" in lines
        assert "power(50.0)" not in lines

    def test_image_section_op_before_workpiece_start_raises(
        self, encoder, mock_machine, doc
    ):
        """An image-section op before WORKPIECE_START must fail loudly."""
        ops = Ops()
        ops.job_start()
        ops.ops_section_start(
            SectionType.RASTER_FILL,
            "wp-0",
            raster_mode=RasterMode.VARIABLE_POWER,
        )
        ops.set_power(0.5)

        with pytest.raises(ValueError, match="WORKPIECE_START"):
            encoder.encode(ops, mock_machine, doc)


class TestLayerOverscan:
    """declare_layer overscan follows the layer's raster scan lines."""

    def _overscan_job(self, doc, scan_end):
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.ops_section_start(
            SectionType.RASTER_FILL,
            "wp-0",
            raster_mode=RasterMode.VARIABLE_POWER,
        )
        ops.move_to(0.0, 0.0, 0.0)
        ops.scan_to(scan_end[0], scan_end[1], 0.0, bytearray([128, 128]))
        ops.ops_section_end(
            SectionType.RASTER_FILL, raster_mode=RasterMode.VARIABLE_POWER
        )
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        return ops

    def _declared_overscan(self, encoder, mock_machine, doc, ops):
        result = encoder.encode(ops, mock_machine, doc)
        declared = [
            line
            for line in result.text.split("\n")
            if line.startswith("declare_layer(")
        ]
        return _declare_layer_overscan(declared[0])

    def test_horizontal_scan_uses_x_bi(self, encoder, mock_machine, doc):
        """A horizontal scan line must yield X_BI overscan."""
        ops = self._overscan_job(doc, (5.0, 0.0))
        assert self._declared_overscan(encoder, mock_machine, doc, ops) == (
            "X_BI"
        )

    def test_vertical_scan_uses_y_bi(self, encoder, mock_machine, doc):
        """A vertical scan line must yield Y_BI overscan."""
        ops = self._overscan_job(doc, (0.0, 5.0))
        assert self._declared_overscan(encoder, mock_machine, doc, ops) == (
            "Y_BI"
        )

    def test_diagonal_scan_uses_none(self, encoder, mock_machine, doc):
        """A diagonal scan line must yield NONE overscan."""
        ops = self._overscan_job(doc, (5.0, 5.0))
        assert self._declared_overscan(encoder, mock_machine, doc, ops) == (
            "NONE"
        )

    def test_vector_layer_uses_none(self, encoder, mock_machine, doc):
        """A vector layer (no sections) must yield NONE overscan."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.set_power(0.5)
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        assert self._declared_overscan(encoder, mock_machine, doc, ops) == (
            "NONE"
        )

    def test_raster_layer_overscan_uses_angle(
        self, encoder, mock_machine, doc
    ):
        """RASTER layers (CONSTANT_POWER) derive overscan from angle."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.ops_section_start(
            SectionType.RASTER_FILL,
            "wp-0",
            raster_mode=RasterMode.CONSTANT_POWER,
        )
        ops.move_to(0.0, 0.0, 0.0)
        ops.scan_to(5.0, 0.0, 0.0, bytearray([255, 255]))
        ops.ops_section_end(
            SectionType.RASTER_FILL, raster_mode=RasterMode.CONSTANT_POWER
        )
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        assert self._declared_overscan(encoder, mock_machine, doc, ops) == (
            "X_BI"
        )

    def test_raster_layer_without_scan_lines_uses_none(
        self, encoder, mock_machine, doc
    ):
        """RASTER layers without scan lines must yield NONE overscan."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.ops_section_start(
            SectionType.RASTER_FILL,
            "wp-0",
            raster_mode=RasterMode.VARIABLE_POWER,
        )
        ops.set_power(0.5)
        ops.ops_section_end(
            SectionType.RASTER_FILL, raster_mode=RasterMode.VARIABLE_POWER
        )
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        assert self._declared_overscan(encoder, mock_machine, doc, ops) == (
            "NONE"
        )


class TestCurveLinearization:
    """Tests for curve commands linearized into cut segments."""

    def test_arc_linearizes_to_cut_lines(self, encoder, mock_machine, doc):
        """ARC_TO should decompose into cut segments."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.move_to(0.0, 0.0, 0.0)
        ops.arc_to(10.0, 0.0, 5.0, 0.0, clockwise=True)
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        lines = result.text.split("\n")
        cut_lines = [line for line in lines if line.startswith("cut_xy_to(")]
        assert len(cut_lines) >= 3
        assert any(line.startswith("declare_layer(") for line in lines)

    def test_scan_line_linearizes(self, encoder, mock_machine, doc):
        """SCAN_LINE should decompose into power and cut segments."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.move_to(0.0, 0.0, 0.0)
        # All values map above the 8% controller minimum so the raw
        # power_range() pass-through does not raise in GlueScript.
        power_values = bytearray([64, 128, 255, 128, 64])
        ops.scan_to(5.0, 0.0, 0.0, power_values)
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        lines = result.text.split("\n")
        assert any(line.startswith("power_range(") for line in lines)
        assert any(line.startswith("cut_xy_to(") for line in lines)

    def test_constant_power_scan_with_zero_pixels_emits_cuts(
        self, mock_machine, doc
    ):
        """CONSTANT_POWER scan lines must always cut, never become moves.

        Off-pixels carry 0.0 power, which must still be emitted through
        power_range(0.0, 0.0) as a verification marker. A real GlueScript
        raises ValueError on that call, so a mock is injected.
        """
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.ops_section_start(
            SectionType.RASTER_FILL,
            "wp-0",
            raster_mode=RasterMode.CONSTANT_POWER,
        )
        ops.move_to(0.0, 0.0, 0.0)
        ops.scan_to(4.0, 0.0, 0.0, bytearray([0, 255, 255, 0]))
        ops.ops_section_end(
            SectionType.RASTER_FILL, raster_mode=RasterMode.CONSTANT_POWER
        )
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()

        mock_gluescript = Mock(spec=GlueScript)
        mock_gluescript.gluescript = []
        encoder = RuidaRPAEncoder(gluescript=mock_gluescript)
        encoder.encode(ops, mock_machine, doc)

        assert (0.0, 0.0) in [
            call.args for call in mock_gluescript.power_range.call_args_list
        ]
        assert mock_gluescript.cut_xy_to.call_args_list
        # The initial move_to(0.0, 0.0, 0.0) before the scan is a rapid
        # move; the scan segments themselves must never become moves.
        assert all(
            call.args[0] == 0.0
            for call in mock_gluescript.move_xy_to.call_args_list
        )

    def test_variable_power_scan_with_zero_pixels_emits_moves(
        self, encoder, mock_machine, doc
    ):
        """VARIABLE_POWER scans keep moves for 0-power pixels."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.ops_section_start(
            SectionType.RASTER_FILL,
            "wp-0",
            raster_mode=RasterMode.VARIABLE_POWER,
        )
        ops.move_to(0.0, 0.0, 0.0)
        ops.scan_to(4.0, 0.0, 0.0, bytearray([0, 255, 255, 0]))
        ops.ops_section_end(
            SectionType.RASTER_FILL, raster_mode=RasterMode.VARIABLE_POWER
        )
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        lines = result.text.split("\n")
        assert any(line.startswith("move_xy_to(") for line in lines)
        assert any(line.startswith("cut_xy_to(") for line in lines)
        assert "power_range(0.0, 0.0)" not in lines
        assert "power(0.0)" not in lines


class TestOpMapGeneration:
    """Tests for bidirectional op_map generation."""

    def _structured_job(self, doc):
        ops = Ops()
        ops.job_start()  # 0 -> declare_job line
        ops.layer_start(layer_uid=doc.layers[0].uid)  # 1 -> nothing
        ops.workpiece_start("wp-0")  # 2 -> comment + declare_layer
        ops.set_power(0.5)  # 3 -> power_range line
        ops.move_to(5.0, 5.0, 0.0)  # 4 -> move_xy_to line
        ops.line_to(10.0, 8.0, 0.0)  # 5 -> cut_xy_to line
        ops.workpiece_end("wp-0")  # 6 -> nothing
        ops.layer_end(layer_uid=doc.layers[0].uid)  # 7 -> nothing
        ops.job_end()  # 8 -> end_job line
        return ops

    def test_every_op_has_entry(self, encoder, mock_machine, doc):
        """Every op index must be present in the op_map."""
        result = encoder.encode(self._structured_job(doc), mock_machine, doc)

        assert result.op_map.op_count == 9

    def test_job_start_maps_to_header(self, encoder, mock_machine, doc):
        """JOB_START should map to the declare_job header line."""
        result = encoder.encode(self._structured_job(doc), mock_machine, doc)

        assert result.op_map.span_for_op(0) == (0, 1)
        assert result.op_map.op_for_line(0) == 0

    def test_workpiece_start_maps_to_attrs(self, encoder, mock_machine, doc):
        """WORKPIECE_START should map to the layer attribute block."""
        result = encoder.encode(self._structured_job(doc), mock_machine, doc)
        lines = result.text.split("\n")
        first_attr = next(
            i
            for i, line in enumerate(lines)
            if line.startswith("declare_layer(")
        )
        last_layer = next(
            i
            for i, line in enumerate(lines)
            if line.startswith("power_range(")
        )

        expected = list(range(first_attr - 1, last_layer))
        assert result.op_map.span_for_op(2) == (
            expected[0],
            expected[-1] - expected[0] + 1,
        )
        for line_num in expected:
            assert result.op_map.op_for_line(line_num) == 2

    def test_action_ops_map_to_action_lines(self, encoder, mock_machine, doc):
        """Set/move/cut ops should map to their action lines."""
        result = encoder.encode(self._structured_job(doc), mock_machine, doc)
        lines = result.text.split("\n")
        power_line = next(
            i
            for i, line in enumerate(lines)
            if line.startswith("power_range(")
        )
        move_line = next(
            i for i, line in enumerate(lines) if line.startswith("move_xy_to(")
        )
        cut_line = next(
            i for i, line in enumerate(lines) if line.startswith("cut_xy_to(")
        )

        assert result.op_map.span_for_op(3) == (power_line, 1)
        assert result.op_map.op_for_line(power_line) == 3
        assert result.op_map.span_for_op(4) == (move_line, 1)
        assert result.op_map.op_for_line(move_line) == 4
        assert result.op_map.span_for_op(5) == (cut_line, 1)
        assert result.op_map.op_for_line(cut_line) == 5

    def test_layer_end_maps_to_nothing(self, encoder, mock_machine, doc):
        """LAYER_END produces no transcript lines."""
        result = encoder.encode(self._structured_job(doc), mock_machine, doc)

        assert result.op_map.span_for_op(7) == (0, 0)

    def test_layer_start_emits_nothing(self, encoder, mock_machine, doc):
        """LAYER_START produces no transcript lines."""
        result = encoder.encode(self._structured_job(doc), mock_machine, doc)

        assert result.op_map.span_for_op(1) == (0, 0)

    def test_job_end_maps_to_tail(self, encoder, mock_machine, doc):
        """JOB_END should map to the end_job line."""
        result = encoder.encode(self._structured_job(doc), mock_machine, doc)
        lines = result.text.split("\n")
        end_job_line = lines.index("end_job()")

        assert result.op_map.span_for_op(8) == (end_job_line, 1)
        assert result.op_map.op_for_line(end_job_line) == 8

    def test_reverse_mapping_is_consistent(self, encoder, mock_machine, doc):
        """Every line must map back to its owning op."""
        result = encoder.encode(self._structured_job(doc), mock_machine, doc)
        lines = result.text.split("\n")

        for line_num in range(len(lines)):
            op_index = result.op_map.op_for_line(line_num)
            assert op_index is not None
            start, count = result.op_map.span_for_op(op_index)
            assert start <= line_num < start + count

    def _three_layer_job(self, doc):
        ops = Ops()
        ops.job_start()  # 0 -> declare_job line
        ops.layer_start(layer_uid=doc.layers[0].uid)  # 1 -> nothing
        ops.workpiece_start("wp-0")  # 2 -> comment + declare_layer
        ops.set_power(0.5)  # 3 -> power_range line
        ops.move_to(5.0, 5.0, 0.0)  # 4 -> move_xy_to line
        ops.workpiece_end("wp-0")  # 5 -> nothing
        ops.layer_end(layer_uid=doc.layers[0].uid)  # 6 -> nothing
        ops.layer_start(layer_uid=doc.layers[1].uid)  # 7 -> nothing
        ops.workpiece_start("wp-1")  # 8 -> comment + declare_layer
        ops.move_to(1.0, 1.0, 0.0)  # 9 -> move_xy_to line
        ops.workpiece_end("wp-1")  # 10 -> nothing
        ops.layer_end(layer_uid=doc.layers[1].uid)  # 11 -> nothing
        ops.layer_start(layer_uid=doc.layers[2].uid)  # 12 -> nothing
        ops.workpiece_start("wp-2")  # 13 -> comment + declare_layer
        ops.line_to(9.0, 9.0, 0.0)  # 14 -> cut_xy_to line
        ops.workpiece_end("wp-2")  # 15 -> nothing
        ops.layer_end(layer_uid=doc.layers[2].uid)  # 16 -> nothing
        ops.job_end()  # 17 -> end_job line
        return ops

    def test_three_layer_op_map_positions(self, encoder, mock_machine, doc):
        """A 3-layer job must keep exact per-layer op_map positions."""
        result = encoder.encode(self._three_layer_job(doc), mock_machine, doc)
        lines = result.text.split("\n")
        op_map = result.op_map

        declare_lines = [
            i
            for i, line in enumerate(lines)
            if line.startswith("declare_layer(")
        ]
        attr0, attr1, attr2 = (
            declare_lines[0],
            declare_lines[1],
            declare_lines[2],
        )
        end_job = lines.index("end_job()")

        assert attr0 < attr1 < attr2 < end_job
        assert end_job == len(lines) - 1

        assert op_map.span_for_op(0) == (0, 1)
        assert op_map.span_for_op(2) == (attr0 - 1, 2)
        assert op_map.span_for_op(8) == (attr1 - 1, 2)
        assert op_map.span_for_op(13) == (attr2 - 1, 2)
        power_line = next(
            i
            for i, line in enumerate(lines)
            if line.startswith("power_range(")
        )
        assert op_map.span_for_op(3) == (power_line, 1)
        assert op_map.op_for_line(power_line) == 3
        move3 = next(
            i for i, line in enumerate(lines) if line == "move_xy_to(5.0, 5.0)"
        )
        assert op_map.span_for_op(4) == (move3, 1)
        assert op_map.op_for_line(move3) == 4
        move6 = next(
            i for i, line in enumerate(lines) if line == "move_xy_to(1.0, 1.0)"
        )
        assert op_map.span_for_op(9) == (move6, 1)
        assert op_map.op_for_line(move6) == 9
        cut9 = next(
            i for i, line in enumerate(lines) if line == "cut_xy_to(9.0, 9.0)"
        )
        assert op_map.span_for_op(14) == (cut9, 1)
        assert op_map.op_for_line(cut9) == 14
        assert op_map.span_for_op(17) == (end_job, 1)
        assert op_map.op_for_line(end_job) == 17


class TestOpMapLayoutPinning:
    """Exact op_map layout assertions across a 2-layer job."""

    def test_two_layer_layout_positions(self, encoder, mock_machine, doc):
        """Header/attrs/actions/end_job keep fixed positions."""
        ops = Ops()
        ops.job_start()  # 0 -> declare_job line
        ops.layer_start(layer_uid=doc.layers[0].uid)  # 1 -> nothing
        ops.workpiece_start("wp-0")  # 2 -> comment + declare_layer
        ops.move_to(5.0, 5.0, 0.0)  # 3 -> move_xy_to line
        ops.workpiece_end("wp-0")  # 4 -> nothing
        ops.layer_end(layer_uid=doc.layers[0].uid)  # 5 -> nothing
        ops.layer_start(layer_uid=doc.layers[1].uid)  # 6 -> nothing
        ops.workpiece_start("wp-1")  # 7 -> comment + declare_layer
        ops.line_to(10.0, 8.0, 0.0)  # 8 -> cut_xy_to line
        ops.workpiece_end("wp-1")  # 9 -> nothing
        ops.layer_end(layer_uid=doc.layers[1].uid)  # 10 -> nothing
        ops.job_end()  # 11 -> end_job line
        result = encoder.encode(ops, mock_machine, doc)
        lines = result.text.split("\n")
        op_map = result.op_map

        declare_lines = [
            i
            for i, line in enumerate(lines)
            if line.startswith("declare_layer(")
        ]
        attr0, attr1 = declare_lines[0], declare_lines[1]
        move2 = next(
            i for i, line in enumerate(lines) if line == "move_xy_to(5.0, 5.0)"
        )
        cut5 = next(
            i for i, line in enumerate(lines) if line == "cut_xy_to(10.0, 8.0)"
        )
        end_job = lines.index("end_job()")

        assert end_job == len(lines) - 1
        assert attr0 < move2 < attr1 < cut5 < end_job

        assert op_map.span_for_op(0) == (0, 1)
        assert op_map.span_for_op(2) == (attr0 - 1, 2)
        assert op_map.span_for_op(3) == (move2, 1)
        assert op_map.op_for_line(move2) == 3
        assert op_map.span_for_op(7) == (attr1 - 1, 2)
        assert op_map.span_for_op(8) == (cut5, 1)
        assert op_map.op_for_line(cut5) == 8
        assert op_map.span_for_op(11) == (end_job, 1)

        for line_num in range(len(lines)):
            op_index = op_map.op_for_line(line_num)
            assert op_index is not None
            start, count = op_map.span_for_op(op_index)
            assert start <= line_num < start + count


class TestErrorHandling:
    """Tests for encoder error handling."""

    def test_missing_job_end_raises(self, encoder, mock_machine, doc):
        """An incomplete job (no JOB_END) must fail loudly."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.move_to(0.0, 0.0, 0.0)

        with pytest.raises(RuntimeError, match="JOB_END"):
            encoder.encode(ops, mock_machine, doc)

    def test_layer_scoped_op_before_layer_start_raises(
        self, encoder, mock_machine, doc
    ):
        """Layer-scoped ops before LAYER_START must fail loudly."""
        ops = Ops()
        ops.job_start()
        ops.set_power(0.5)

        with pytest.raises(ValueError, match="LAYER_START"):
            encoder.encode(ops, mock_machine, doc)

    def test_move_outside_layer_raises(self, encoder, mock_machine, doc):
        """Moves without an active layer must fail loudly."""
        ops = Ops()
        ops.job_start()
        ops.move_to(1.0, 1.0, 0.0)

        with pytest.raises(ValueError, match="LAYER_START"):
            encoder.encode(ops, mock_machine, doc)

    def test_workpiece_start_before_layer_start_raises(
        self, encoder, mock_machine, doc
    ):
        """WORKPIECE_START before LAYER_START must fail loudly."""
        ops = Ops()
        ops.job_start()
        ops.workpiece_start("wp-0")

        with pytest.raises(
            ValueError, match="WORKPIECE_START encountered before LAYER_START"
        ):
            encoder.encode(ops, mock_machine, doc)

    def test_layer_end_without_workpiece_raises(
        self, encoder, mock_machine, doc
    ):
        """LAYER_END without any WORKPIECE_START must fail loudly."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.layer_end(layer_uid=doc.layers[0].uid)

        with pytest.raises(ValueError, match="WORKPIECE_START"):
            encoder.encode(ops, mock_machine, doc)


def _plan_job(doc):
    """Return Ops for a small two-layer plan test job."""
    ops = Ops()
    ops.job_start()
    ops.layer_start(layer_uid=doc.layers[0].uid)
    ops.workpiece_start("wp-0")
    ops.set_power(0.5)
    ops.move_to(5.0, 5.0, 0.0)
    ops.line_to(10.0, 8.0, 0.0)
    ops.workpiece_end("wp-0")
    ops.layer_end(layer_uid=doc.layers[0].uid)
    ops.layer_start(layer_uid=doc.layers[1].uid)
    ops.workpiece_start("wp-1")
    ops.set_feed_rate(200)
    ops.move_to(20.0, 20.0, 0.0)
    ops.workpiece_end("wp-1")
    ops.layer_end(layer_uid=doc.layers[1].uid)
    ops.job_end()
    return ops


class TestTranscriptOutput:
    """The encoder's text output IS the GlueScript transcript."""

    def test_encode_populates_transcript(self, encoder, mock_machine, doc):
        """encode() must return the transcript as text."""
        result = encoder.encode(_plan_job(doc), mock_machine, doc)
        lines = result.text.split("\n")
        assert len(lines) > 0
        assert all(isinstance(line, str) for line in lines)

    def test_empty_ops_have_no_transcript(self, encoder, mock_machine, doc):
        """An empty job produces no transcript (no GlueScript calls)."""
        result = encoder.encode(Ops(), mock_machine, doc)
        assert result.text == ""

    def test_transcript_starts_with_declare_job_and_ends_with_end_job(
        self, encoder, mock_machine, doc
    ):
        """The transcript frames the job exactly like the driver transcript."""
        result = encoder.encode(_plan_job(doc), mock_machine, doc)
        lines = result.text.split("\n")
        assert lines[0].startswith("declare_job(")
        assert lines[-1] == "end_job()"

    def test_transcript_records_structural_and_raw_calls(
        self, encoder, mock_machine, doc
    ):
        """Structural calls and power_range raw lines are recorded."""
        result = encoder.encode(_plan_job(doc), mock_machine, doc)
        lines = result.text.split("\n")
        assert any(line.startswith("declare_layer(") for line in lines)
        assert any(line.startswith("move_xy_to(") for line in lines)
        assert any(line.startswith("cut_xy_to(") for line in lines)
        assert any(line.startswith("power_range(") for line in lines)

    def test_transcript_survives_encoder_reuse(
        self, encoder, mock_machine, doc
    ):
        """A new encode replaces the transcript; the old text stays valid."""
        result1 = encoder.encode(_plan_job(doc), mock_machine, doc)
        lines1 = result1.text
        result2 = encoder.encode(_plan_job(doc), mock_machine, doc)
        assert result2.text == lines1

    def test_transcript_records_per_op_settings_as_lines(
        self, encoder, mock_machine, doc
    ):
        """Per-op settings record as transcript lines."""
        result = encoder.encode(_plan_job(doc), mock_machine, doc)
        assert "cut_speed(3.3333333333333335)" in result.text

    def test_transcript_records_select_laser_power_and_power_range(
        self, encoder, mock_machine, doc
    ):
        """select_laser, power, and power_range all appear as lines."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.set_head("laser-2")
        ops.ops_section_start(
            SectionType.RASTER_FILL,
            "wp-0",
            raster_mode=RasterMode.VARIABLE_POWER,
        )
        ops.set_power(0.5)
        ops.ops_section_end(
            SectionType.RASTER_FILL, raster_mode=RasterMode.VARIABLE_POWER
        )
        ops.set_power(0.5)
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)
        lines = result.text.split("\n")
        assert any(line.startswith("select_laser(") for line in lines)
        assert any(line.startswith("power(") for line in lines)
        assert any(line.startswith("power_range(") for line in lines)


class TestWcsToRefPoint:
    """Active WCS names map to GlueScript declare_job reference points."""

    # The default doc has no name, so the encoder uses its default label.
    _JOB_LABEL = "Rayforge Job"

    @pytest.fixture(autouse=True)
    def _reset_fallback_wcs(self):
        """Reset the module-level G54 fallback dedup state."""
        rpa_encoder._last_fallback_wcs = None
        yield
        rpa_encoder._last_fallback_wcs = None

    @pytest.mark.parametrize(
        "wcs,expected",
        [
            ("MACHINE", "MACHINE"),
            ("ANCHOR", "ABSOLUTE"),
            ("CURRENT", "CURRENT"),
            ("SET_POINT", "SET_POINT"),
        ],
    )
    def test_declare_job_ref_point_maps_from_active_wcs(
        self, encoder, mock_machine, doc, wcs, expected
    ):
        """The declare_job ref point mirrors the active framework WCS."""
        mock_machine.active_wcs = wcs
        result = encoder.encode(_plan_job(doc), mock_machine, doc)
        assert result.text.split("\n")[0] == (
            f"declare_job({self._JOB_LABEL!r}, {expected!r}, "
            "[0.0, 0.0], 1, 1, 0.0, 0.0)"
        )
        if wcs == "ANCHOR":
            # ANCHOR maps to ABSOLUTE; the encoder passes abs_xy=None,
            # which GlueScript normalizes to [0.0, 0.0].
            args = ast.literal_eval(
                result.text.split("\n")[0][len("declare_job(") : -1]
            )
            assert args[2] == [0.0, 0.0]

    def test_g54_falls_back_to_machine_ref_point(
        self, encoder, mock_machine, doc
    ):
        """The framework default G54 must fall back to MACHINE."""
        mock_machine.active_wcs = "G54"
        result = encoder.encode(_plan_job(doc), mock_machine, doc)
        assert result.text.split("\n")[0] == (
            f"declare_job({self._JOB_LABEL!r}, 'MACHINE', "
            "[0.0, 0.0], 1, 1, 0.0, 0.0)"
        )

    def test_machine_none_defaults_to_machine_ref_point(self, encoder, doc):
        """machine=None must default to the MACHINE reference point."""
        result = encoder.encode(_plan_job(doc), None, doc)
        assert result.text.split("\n")[0] == (
            f"declare_job({self._JOB_LABEL!r}, 'MACHINE', "
            "[0.0, 0.0], 1, 1, 0.0, 0.0)"
        )

    def test_unknown_wcs_raises_value_error(self, encoder, mock_machine, doc):
        """An unrecognized WCS must fail loudly."""
        mock_machine.active_wcs = "G55"
        with pytest.raises(ValueError, match="G55"):
            encoder.encode(_plan_job(doc), mock_machine, doc)

    def test_g54_warning_fires_only_when_changed(
        self, encoder, mock_machine, doc, caplog
    ):
        """The G54 fallback warning dedups until a real WCS is used."""
        caplog.set_level(logging.WARNING, logger=rpa_encoder.logger.name)
        mock_machine.active_wcs = "G54"
        encoder.encode(_plan_job(doc), mock_machine, doc)
        encoder.encode(_plan_job(doc), mock_machine, doc)
        mock_machine.active_wcs = "MACHINE"
        encoder.encode(_plan_job(doc), mock_machine, doc)
        mock_machine.active_wcs = "G54"
        encoder.encode(_plan_job(doc), mock_machine, doc)
        warnings = [
            record
            for record in caplog.records
            if "framework default" in record.message
        ]
        assert len(warnings) == 2


class TestInjectedGluescript:
    """The encoder authors into an injected GlueScript backend."""

    @staticmethod
    def _job(doc):
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.workpiece_start("wp-0")
        ops.move_to(5.0, 5.0, 0.0)
        ops.line_to(10.0, 8.0, 0.0)
        ops.workpiece_end("wp-0")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        return ops

    def test_authors_into_injected_gluescript(self, mock_machine, doc):
        """encode() must author into the injected instance, not a new one."""
        injected = GlueScript()
        encoder = RuidaRPAEncoder(gluescript=injected)

        result = encoder.encode(self._job(doc), mock_machine, doc)

        assert encoder._gluescript is injected
        assert any(
            line.startswith("declare_job(") for line in injected.gluescript
        )
        assert result.text != ""

    def test_calls_new_gluescript_to_reset(self, mock_machine, doc):
        """Each encode must reset the injected backend via new_gluescript."""
        injected = GlueScript()
        encoder = RuidaRPAEncoder(gluescript=injected)

        encoder.encode(self._job(doc), mock_machine, doc)
        encoder.encode(self._job(doc), mock_machine, doc)

        # The transcript must not accumulate across encodes.
        declared = sum(
            1
            for line in injected.gluescript
            if line.startswith("declare_job(")
        )
        assert declared == 1

    def test_reset_state_preserves_injected_instance(self, mock_machine, doc):
        """_reset_state() must keep the injected instance across encodes."""
        injected = GlueScript()
        encoder = RuidaRPAEncoder(gluescript=injected)

        encoder.encode(self._job(doc), mock_machine, doc)
        encoder._reset_state()

        assert encoder._gluescript is injected
