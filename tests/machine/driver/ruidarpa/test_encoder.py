"""
Test suite for the RuidaRPAEncoder.

The encoder drives the ruida-pa GlueScript API to produce rpascript text.
Tests cover:
- Job framing (START_JOB / END_JOB / EOF, auto bounding boxes)
- Layer declaration from workflow steps (settings, defaults, power clamp)
- Near/far move and cut form selection
- Configuration actions (power, speed, frequency, pulse width, air assist)
- Curve linearization (arcs, scan lines)
- Bidirectional op_map generation
- Error handling (missing JOB_END, unknown commands, missing layer)
"""

import logging

import pytest
from raygeo.ops import Ops
from raygeo.ops.state import AirAssistMode, CoolantMode
from raygeo.ops.types import RasterMode, SectionType

from rayforge.core.doc import Doc
from rayforge.core.step import Step
from rayforge.machine.driver.ruidarpa import rpa_encoder
from rayforge.machine.driver.ruidarpa.rpa_encoder import RuidaRPAEncoder
from rayforge.machine.models.laser import Laser
from rayforge.pipeline.encoder.base import EncodedOutput, MachineCodeOpMap


class CutStep(Step):
    """Minimal concrete Step for encoder layer-settings tests."""

    def __init__(self):
        super().__init__(typelabel="cut")


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
        assert result.op_map.op_to_machine_code == {}
        assert result.op_map.machine_code_to_op == {}

    def test_encoder_state_resets_between_encodes(
        self, encoder, mock_machine, doc
    ):
        """Each encode() call should reset internal state."""
        ops1 = Ops()
        ops1.job_start()
        ops1.layer_start(layer_uid=doc.layers[0].uid)
        ops1.set_power(0.5)
        ops1.move_to(0.0, 0.0, 0.0)
        ops1.layer_end(layer_uid=doc.layers[0].uid)
        ops1.job_end()
        encoder.encode(ops1, mock_machine, doc)

        ops2 = Ops()
        ops2.job_start()
        ops2.layer_start(layer_uid=doc.layers[0].uid)
        ops2.move_to(0.0, 0.0, 0.0)
        ops2.layer_end(layer_uid=doc.layers[0].uid)
        ops2.job_end()
        result2 = encoder.encode(ops2, mock_machine, doc)

        assert encoder.active_laser == 1
        # Second job: 0=job_start, 1=layer_start, 2=move_to,
        # 3=layer_end, 4=job_end
        assert set(result2.op_map.op_to_machine_code.keys()) == set(range(5))
        assert result2.op_map.op_to_machine_code[3] == []


class TestJobStructure:
    """Tests for job framing and bounding boxes."""

    def _simple_job(self, doc):
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.move_to(5.0, 5.0, 0.0)
        ops.line_to(10.0, 8.0, 0.0)
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        return ops

    def test_job_framing(self, encoder, mock_machine, doc):
        """Job should be framed by START_JOB and END_JOB/EOF."""
        result = encoder.encode(self._simple_job(doc), mock_machine, doc)
        lines = result.text.split("\n")

        assert lines[0] == "# Job: Rayforge Job"
        assert "START_JOB" in lines
        assert lines[-2] == "END_JOB"
        assert lines[-1] == "EOF"

    def test_auto_bounding_box(self, encoder, mock_machine, doc):
        """Bounding box should be derived from actual cut extents."""
        result = encoder.encode(self._simple_job(doc), mock_machine, doc)
        lines = result.text.split("\n")

        assert "JOB_TOP_RIGHT X=5.000mm Y=5.000mm" in lines
        assert "JOB_BOTTOM_LEFT X=10.000mm Y=8.000mm" in lines
        assert "LAYER_TOP_RIGHT Layer:0 X=5.000mm Y=5.000mm" in lines
        assert "LAYER_BOTTOM_LEFT Layer:0 X=10.000mm Y=8.000mm" in lines

    def test_no_manual_bbox_artifacts(self, encoder, mock_machine, doc):
        """The encoder must not emit manual bbox command blocks."""
        result = encoder.encode(self._simple_job(doc), mock_machine, doc)
        lines = result.text.split("\n")

        assert "ARRAY_END" not in lines
        assert "BLOCK_END" not in lines
        assert "SET_BBOX" not in lines

    def test_layer_selection_markers(self, encoder, mock_machine, doc):
        """Layer selection markers should be emitted per layer."""
        result = encoder.encode(self._simple_job(doc), mock_machine, doc)
        lines = result.text.split("\n")

        assert "LAST_LAYER Layer:0" in lines
        assert "SELECT_LAYER Layer:0" in lines


class TestLayerDeclaration:
    """Tests for layer attribute declaration."""

    def test_default_layer_settings(self, encoder, mock_machine, doc):
        """Layers without workflow steps should use safe defaults."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        lines = result.text.split("\n")
        assert "CUT_SPEED_LASER_1 Layer:0 Speed:100.0mm/S" in lines
        assert "LAYER_MIN_POWER_1 Layer:0 Power:20.0%" in lines
        assert "LAYER_MAX_POWER_1 Layer:0 Power:20.0%" in lines

    def test_layer_settings_from_step(self, encoder, mock_machine, doc):
        """Layer attributes should come from the first workflow step."""
        step = CutStep()
        step.power = 0.5
        step.cut_speed = 300
        doc.layers[0].workflow.add_step(step)

        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        lines = result.text.split("\n")
        assert "CUT_SPEED_LASER_1 Layer:0 Speed:5.0mm/S" in lines
        assert "LAYER_MIN_POWER_1 Layer:0 Power:50.0%" in lines
        assert "LAYER_MAX_POWER_1 Layer:0 Power:50.0%" in lines

    def test_power_below_minimum_is_clamped(
        self, encoder, mock_machine, doc, caplog
    ):
        """Power below the 8% controller minimum must clamp up."""
        step = CutStep()
        step.power = 0.05
        doc.layers[0].workflow.add_step(step)

        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        lines = result.text.split("\n")
        assert "LAYER_MIN_POWER_1 Layer:0 Power:8.0%" in lines
        assert "LAYER_MAX_POWER_1 Layer:0 Power:8.0%" in lines
        assert any("clamping" in record.message for record in caplog.records)

    def test_unknown_layer_uses_defaults(self, encoder, mock_machine, doc):
        """Layers absent from the document should still stage cleanly."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid="missing-layer-uid")
        ops.move_to(1.0, 1.0, 0.0)
        ops.layer_end(layer_uid="missing-layer-uid")
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        assert "LAST_LAYER Layer:0" in result.text


class TestMoveCutForms:
    """Tests for near/far move form auto-selection."""

    def test_near_move_uses_near_form(self, encoder, mock_machine, doc):
        """Small moves should use the near form."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.move_to(5.0, 5.0, 0.0)
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        assert "MOVE_NEAR_XY nearX=5.000mm nearY=5.000mm" in result.text
        assert "MOVE_FAR_XY" not in result.text

    def test_far_move_uses_far_form(self, encoder, mock_machine, doc):
        """Moves beyond the 8.192mm threshold must use the far form."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.move_to(0.0, 0.0, 0.0)
        ops.move_to(150.0, 0.0, 0.0)
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        assert "MOVE_FAR_XY X=150.000mm Y=0.000mm" in result.text

    def test_near_cut_uses_near_form(self, encoder, mock_machine, doc):
        """Small cuts should use the near form."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.move_to(0.0, 0.0, 0.0)
        ops.line_to(5.0, 5.0, 0.0)
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        assert "CUT_NEAR_XY nearX=5.000mm nearY=5.000mm" in result.text

    def test_far_cut_uses_far_form(self, encoder, mock_machine, doc):
        """Cuts beyond the 8.192mm threshold must use the far form."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.move_to(0.0, 0.0, 0.0)
        ops.line_to(20.0, 0.0, 0.0)
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        assert "CUT_FAR_XY X=20.000mm Y=0.000mm" in result.text


class TestSettingsCommands:
    """Tests for configuration action commands."""

    def test_power_emits_min_max_lines(self, encoder, mock_machine, doc):
        """SET_POWER should emit MIN and MAX power action lines."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.set_power(0.5)
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        lines = result.text.split("\n")
        assert "MIN_POWER_1 Power:50.0%" in lines
        assert "MAX_POWER_1 Power:50.0%" in lines

    def test_power_action_below_minimum_clamps(
        self, encoder, mock_machine, doc, caplog
    ):
        """Per-op SET_POWER below 8% must clamp with a warning."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.set_power(0.05)
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        lines = result.text.split("\n")
        assert "MIN_POWER_1 Power:8.0%" in lines
        assert "MAX_POWER_1 Power:8.0%" in lines
        assert any("clamping" in record.message for record in caplog.records)

    def test_legacy_coolant_non_off_logs_warning(
        self, encoder, mock_machine, doc, caplog
    ):
        """Legacy SET_COOLANT must warn instead of silently dropping."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.set_coolant(CoolantMode.FLOOD)
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        assert "AIR_ASSIST_ON" not in result.text
        assert any(
            "SET_COOLANT" in record.message for record in caplog.records
        )

    def test_feed_rate_emits_speed_line(self, encoder, mock_machine, doc):
        """SET_FEED_RATE should emit a comment-only cut_speed line."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.set_feed_rate(200)
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        assert (
            "CUT_SPEED_LASER_1 Layer:0 Speed=3.3333333333333335"
            in result.text
        )

    def test_rapid_rate_emits_axis_speed(self, encoder, mock_machine, doc):
        """SET_RAPID_RATE should emit a comment-only move_speed line."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.set_rapid_rate(500)
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        assert "# move_speed(8.333333333333334)" in result.text

    def test_frequency_emits_khz_line(self, encoder, mock_machine, doc):
        """SET_FREQUENCY should convert Hz to KHz for gluescript."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.set_frequency(20000)
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        assert "# frequency(20.0)" in result.text

    def test_pulse_width_emits_interval_line(self, encoder, mock_machine, doc):
        """SET_PULSE_WIDTH should pass microseconds to gluescript pwm."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.set_pulse_width(50)
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        assert "# pwm(50.0)" in result.text

    def test_dwell_warns_and_emits_no_delay(
        self, encoder, mock_machine, doc, caplog
    ):
        """DWELL is unsupported — warn and emit no delay line."""
        caplog.set_level(logging.WARNING, logger=rpa_encoder.logger.name)
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.dwell(250)
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
        ops.set_air_assist(AirAssistMode.ON)
        ops.move_to(0.0, 0.0, 0.0)
        ops.set_air_assist(AirAssistMode.OFF)
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        lines = result.text.split("\n")
        assert "AIR_ASSIST_ON" in lines
        assert "AIR_ASSIST_OFF" in lines
        assert lines.index("AIR_ASSIST_ON") < lines.index("AIR_ASSIST_OFF")

    def test_set_head_selects_laser_device(
        self, encoder, mock_machine, doc, caplog
    ):
        """SET_HEAD resolves the laser_uid to a device via select_laser."""
        caplog.set_level(logging.WARNING, logger=rpa_encoder.logger.name)
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.set_head("laser-2")
        ops.set_power(0.5)
        ops.set_head("laser-1")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        lines = result.text.split("\n")
        # laser-2 resolves to device 2: select_laser(2) is recorded into the
        # plan, but no raw LASER_DEVICE_2 (only laser 1 is wired in
        # ruida-pa), and a warning is logged.
        assert ("select_laser", (2,)) in result.rpa_plan
        assert "LASER_DEVICE_2" not in lines
        assert any(
            "select_laser" in record.message for record in caplog.records
        )
        # laser-1 resolves to device 1: select_laser(1) emits LASER_DEVICE_1.
        assert "LASER_DEVICE_1" in lines

    def test_set_head_numeric_suffix_fallback_selects_device(
        self, encoder, mock_machine, doc
    ):
        """SET_HEAD with a numeric suffix derives the device number."""
        mock_machine.heads.clear()

        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.set_head("laser_2")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        # ((2 - 1) % 2) + 1 = 2; active_laser defaults to 1
        assert ("select_laser", (2,)) in result.rpa_plan
        assert "LASER_DEVICE_2" not in result.text

        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.set_head("laser_1")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        # ((1 - 1) % 2) + 1 = 1, no switch needed
        assert "LASER_DEVICE_1" not in result.text

    def test_set_head_char_sum_fallback_selects_device(
        self, encoder, mock_machine, doc
    ):
        """SET_HEAD without a numeric suffix falls back to char sums."""
        mock_machine.heads.clear()

        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.set_head("laser-3")
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        lines = result.text.split("\n")
        # sum(ord(c) for c in "laser-3") = 631, odd -> device 2
        assert ("select_laser", (2,)) in result.rpa_plan
        assert "LASER_DEVICE_2" not in lines


class TestSectionPowerRouting:
    """Section-aware SET_POWER routing to GlueScript power/power_range."""

    def _raster_job(self, doc, raster_mode, power):
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.ops_section_start(
            SectionType.RASTER_FILL, "wp-0", raster_mode=raster_mode
        )
        ops.set_power(power)
        ops.ops_section_end(SectionType.RASTER_FILL, raster_mode=raster_mode)
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        return ops

    def test_variable_power_section_uses_power_gluescript(
        self, encoder, mock_machine, doc
    ):
        """RASTER_FILL + VARIABLE_POWER must route through power()."""
        ops = self._raster_job(doc, RasterMode.VARIABLE_POWER, 0.5)
        result = encoder.encode(ops, mock_machine, doc)

        assert "IMD_POWER_1 Power:50.0%" in result.text
        assert ("power", (50.0,)) in result.rpa_plan

    def test_variable_power_section_passes_low_power_through(
        self, encoder, mock_machine, doc, caplog
    ):
        """Image power below 8% must pass through without clamping."""
        ops = self._raster_job(doc, RasterMode.VARIABLE_POWER, 0.05)
        result = encoder.encode(ops, mock_machine, doc)

        assert "IMD_POWER_1 Power:5.0%" in result.text
        assert not any(
            "clamping" in record.message for record in caplog.records
        )

    def test_depth_map_section_uses_power_gluescript(
        self, encoder, mock_machine, doc
    ):
        """RASTER_FILL + DEPTH_MAP must route through power()."""
        ops = self._raster_job(doc, RasterMode.DEPTH_MAP, 0.5)
        result = encoder.encode(ops, mock_machine, doc)

        assert "IMD_POWER_1 Power:50.0%" in result.text
        assert ("power", (50.0,)) in result.rpa_plan

    def test_constant_power_section_uses_power_range(
        self, encoder, mock_machine, doc
    ):
        """RASTER_FILL + CONSTANT_POWER must use power_range()."""
        ops = self._raster_job(doc, RasterMode.CONSTANT_POWER, 0.5)
        result = encoder.encode(ops, mock_machine, doc)

        lines = result.text.split("\n")
        assert "MIN_POWER_1 Power:50.0%" in lines
        assert "MAX_POWER_1 Power:50.0%" in lines
        assert ("power_range", (50.0, 50.0)) in result.rpa_plan

    def test_layer_mode_derived_from_sections(
        self, encoder, mock_machine, doc
    ):
        """declare_layer mode follows the layer's ops sections."""
        cases = [
            (RasterMode.VARIABLE_POWER, "IMAGE"),
            (RasterMode.DEPTH_MAP, "DEPTHMAP"),
            (None, "VECTOR"),
        ]
        for raster_mode, expected in cases:
            ops = Ops()
            ops.job_start()
            ops.layer_start(layer_uid=doc.layers[0].uid)
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
            ops.layer_end(layer_uid=doc.layers[0].uid)
            ops.job_end()
            result = encoder.encode(ops, mock_machine, doc)
            declared = [
                args
                for name, args in result.rpa_plan
                if name == "declare_layer"
            ]
            assert declared[0][2] == expected

    def test_layer_mode_priority_depth_map_wins_regardless_of_order(
        self, encoder, mock_machine, doc
    ):
        """DEPTH_MAP wins over VARIABLE_POWER even when it comes last."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
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
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        declared = [
            args for name, args in result.rpa_plan if name == "declare_layer"
        ]
        assert declared[0][2] == "DEPTHMAP"

    def test_section_state_resets_after_section_end(
        self, encoder, mock_machine, doc
    ):
        """SET_POWER after OPS_SECTION_END must fall back to power_range()."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.ops_section_start(
            SectionType.RASTER_FILL,
            "wp-0",
            raster_mode=RasterMode.VARIABLE_POWER,
        )
        ops.ops_section_end(
            SectionType.RASTER_FILL, raster_mode=RasterMode.VARIABLE_POWER
        )
        ops.set_power(0.5)
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        lines = result.text.split("\n")
        assert "MIN_POWER_1 Power:50.0%" in lines
        assert "MAX_POWER_1 Power:50.0%" in lines
        assert "IMD_POWER_1" not in lines

    def test_image_section_op_before_layer_start_raises(
        self, encoder, mock_machine, doc
    ):
        """An image-section op before LAYER_START must fail loudly."""
        ops = Ops()
        ops.job_start()
        ops.ops_section_start(
            SectionType.RASTER_FILL,
            "wp-0",
            raster_mode=RasterMode.VARIABLE_POWER,
        )
        ops.set_power(0.5)

        with pytest.raises(ValueError, match="LAYER_START"):
            encoder.encode(ops, mock_machine, doc)


class TestCurveLinearization:
    """Tests for curve commands linearized into cut segments."""

    def test_arc_linearizes_to_cut_lines(self, encoder, mock_machine, doc):
        """ARC_TO should decompose into cut segments with power lines."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.move_to(0.0, 0.0, 0.0)
        ops.arc_to(10.0, 0.0, 5.0, 0.0, clockwise=True)
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        lines = result.text.split("\n")
        cut_lines = [line for line in lines if "CUT_" in line]
        assert len(cut_lines) >= 3
        assert any("MIN_POWER_1" in line for line in lines)

    def test_scan_line_linearizes(self, encoder, mock_machine, doc):
        """SCAN_LINE should decompose into power and cut segments."""
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.move_to(0.0, 0.0, 0.0)
        power_values = bytearray([0, 128, 255, 128, 0])
        ops.scan_to(5.0, 0.0, 0.0, power_values)
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        result = encoder.encode(ops, mock_machine, doc)

        lines = result.text.split("\n")
        assert any("MIN_POWER_1" in line for line in lines)
        assert any("CUT_" in line for line in lines)


class TestOpMapGeneration:
    """Tests for bidirectional op_map generation."""

    def _structured_job(self, doc):
        ops = Ops()
        ops.job_start()  # 0 -> header block
        ops.layer_start(layer_uid=doc.layers[0].uid)  # 1 -> attrs block
        ops.set_power(0.5)  # 2 -> MIN/MAX power actions
        ops.move_to(5.0, 5.0, 0.0)  # 3 -> MOVE action
        ops.line_to(10.0, 8.0, 0.0)  # 4 -> CUT action
        ops.layer_end(layer_uid=doc.layers[0].uid)  # 5 -> nothing
        ops.job_end()  # 6 -> LAST_LAYER/SELECT/END_JOB/EOF
        return ops

    def test_every_op_has_entry(self, encoder, mock_machine, doc):
        """Every op index must be present in op_to_machine_code."""
        result = encoder.encode(self._structured_job(doc), mock_machine, doc)

        assert set(result.op_map.op_to_machine_code.keys()) == set(range(7))

    def test_job_start_maps_to_header(self, encoder, mock_machine, doc):
        """JOB_START should map to every line before the first layer attr."""
        result = encoder.encode(self._structured_job(doc), mock_machine, doc)
        lines = result.text.split("\n")
        first_attr = lines.index("# Layer 0: Layer 1")

        assert result.op_map.op_to_machine_code[0] == list(range(first_attr))

    def test_layer_start_maps_to_attrs(self, encoder, mock_machine, doc):
        """LAYER_START should map to the layer attribute block."""
        result = encoder.encode(self._structured_job(doc), mock_machine, doc)
        lines = result.text.split("\n")
        first_attr = lines.index("# Layer 0: Layer 1")
        last_layer = lines.index("LAST_LAYER Layer:0")

        expected = list(range(first_attr, last_layer))
        assert result.op_map.op_to_machine_code[1] == expected

    def test_action_ops_map_to_action_lines(self, encoder, mock_machine, doc):
        """Set/move/cut ops should map to their action lines."""
        result = encoder.encode(self._structured_job(doc), mock_machine, doc)
        lines = result.text.split("\n")
        min_power = lines.index("MIN_POWER_1 Power:50.0%")
        max_power = lines.index("MAX_POWER_1 Power:50.0%")
        move_line = lines.index("MOVE_NEAR_XY nearX=5.000mm nearY=5.000mm")
        cut_line = lines.index("CUT_NEAR_XY nearX=5.000mm nearY=3.000mm")

        assert result.op_map.op_to_machine_code[2] == [
            min_power,
            max_power,
        ]
        assert result.op_map.op_to_machine_code[3] == [move_line]
        assert result.op_map.op_to_machine_code[4] == [cut_line]

    def test_layer_end_maps_to_nothing(self, encoder, mock_machine, doc):
        """LAYER_END produces no rpascript lines."""
        result = encoder.encode(self._structured_job(doc), mock_machine, doc)

        assert result.op_map.op_to_machine_code[5] == []

    def test_job_end_maps_to_tail(self, encoder, mock_machine, doc):
        """JOB_END should map to LAST_LAYER/SELECT/END_JOB/EOF."""
        result = encoder.encode(self._structured_job(doc), mock_machine, doc)
        lines = result.text.split("\n")
        tail = [
            lines.index("LAST_LAYER Layer:0"),
            lines.index("SELECT_LAYER Layer:0"),
            lines.index("END_JOB"),
            lines.index("EOF"),
        ]

        assert result.op_map.op_to_machine_code[6] == sorted(tail)

    def test_reverse_mapping_is_consistent(self, encoder, mock_machine, doc):
        """machine_code_to_op must agree with op_to_machine_code."""
        result = encoder.encode(self._structured_job(doc), mock_machine, doc)
        lines = result.text.split("\n")

        for op_index, block in result.op_map.op_to_machine_code.items():
            for line_num in block:
                assert result.op_map.machine_code_to_op[line_num] == op_index

        mapped = set(result.op_map.machine_code_to_op.keys())
        assert mapped == set(range(len(lines)))

    def _three_layer_job(self, doc):
        ops = Ops()
        ops.job_start()  # 0 -> header
        ops.layer_start(layer_uid=doc.layers[0].uid)  # 1 -> attrs block
        ops.set_power(0.5)  # 2 -> MIN/MAX power actions
        ops.move_to(5.0, 5.0, 0.0)  # 3 -> MOVE action
        ops.layer_end(layer_uid=doc.layers[0].uid)  # 4 -> nothing
        ops.layer_start(layer_uid=doc.layers[1].uid)  # 5 -> attrs block
        ops.move_to(1.0, 1.0, 0.0)  # 6 -> MOVE action
        ops.layer_end(layer_uid=doc.layers[1].uid)  # 7 -> nothing
        ops.layer_start(layer_uid=doc.layers[2].uid)  # 8 -> attrs block
        ops.line_to(9.0, 9.0, 0.0)  # 9 -> CUT action
        ops.layer_end(layer_uid=doc.layers[2].uid)  # 10 -> nothing
        ops.job_end()  # 11 -> LAST_LAYER/SELECTs/END_JOB/EOF
        return ops

    def test_three_layer_op_map_positions(self, encoder, mock_machine, doc):
        """A 3-layer job must keep exact per-layer op_map positions."""
        result = encoder.encode(self._three_layer_job(doc), mock_machine, doc)
        lines = result.text.split("\n")
        op_map = result.op_map.op_to_machine_code

        attr0 = lines.index("# Layer 0: Layer 1")
        attr1 = lines.index("# Layer 1: Layer 2")
        attr2 = lines.index("# Layer 2: Layer 3")
        last_layer = lines.index("LAST_LAYER Layer:2")
        select0 = lines.index("SELECT_LAYER Layer:0")
        select1 = lines.index("SELECT_LAYER Layer:1")
        select2 = lines.index("SELECT_LAYER Layer:2")
        end_job = lines.index("END_JOB")
        eof = lines.index("EOF")

        assert (
            attr0
            < attr1
            < attr2
            < last_layer
            < select0
            < select1
            < select2
            < end_job
            < eof
        )
        assert eof == len(lines) - 1

        assert op_map[0] == list(range(0, attr0))
        assert op_map[1] == list(range(attr0, attr1))
        assert op_map[5] == list(range(attr1, attr2))
        assert op_map[8] == list(range(attr2, last_layer))
        assert op_map[2] == [
            lines.index("MIN_POWER_1 Power:50.0%"),
            lines.index("MAX_POWER_1 Power:50.0%"),
        ]
        assert op_map[3] == [
            lines.index("MOVE_NEAR_XY nearX=5.000mm nearY=5.000mm")
        ]
        assert op_map[6] == [
            lines.index("MOVE_NEAR_XY nearX=-4.000mm nearY=-4.000mm")
        ]
        assert op_map[9] == [
            lines.index("CUT_NEAR_XY nearX=8.000mm nearY=8.000mm")
        ]
        assert op_map[11] == [
            last_layer,
            select0,
            select1,
            select2,
            end_job,
            eof,
        ]


class TestOpMapLayoutPinning:
    """Exact op_map layout assertions across a 2-layer job."""

    def test_two_layer_layout_positions(self, encoder, mock_machine, doc):
        """Header/attrs/actions/END_JOB/EOF keep fixed positions."""
        ops = Ops()
        ops.job_start()  # 0 -> header
        ops.layer_start(layer_uid=doc.layers[0].uid)  # 1 -> attrs block
        ops.move_to(5.0, 5.0, 0.0)  # 2 -> MOVE action
        ops.layer_end(layer_uid=doc.layers[0].uid)  # 3 -> nothing
        ops.layer_start(layer_uid=doc.layers[1].uid)  # 4 -> attrs block
        ops.line_to(10.0, 8.0, 0.0)  # 5 -> CUT action
        ops.layer_end(layer_uid=doc.layers[1].uid)  # 6 -> nothing
        ops.job_end()  # 7 -> LAST_LAYER/SELECTs/END_JOB/EOF
        result = encoder.encode(ops, mock_machine, doc)
        lines = result.text.split("\n")
        op_map = result.op_map.op_to_machine_code

        attr0 = lines.index("# Layer 0: Layer 1")
        attr1 = lines.index("# Layer 1: Layer 2")
        last_layer = lines.index("LAST_LAYER Layer:1")
        select0 = lines.index("SELECT_LAYER Layer:0")
        select1 = lines.index("SELECT_LAYER Layer:1")
        end_job = lines.index("END_JOB")
        eof = lines.index("EOF")

        assert eof == len(lines) - 1
        assert end_job == eof - 1
        assert last_layer < select0 < select1 < end_job

        assert op_map[0] == list(range(0, attr0))
        assert op_map[1] == list(range(attr0, attr1))
        assert op_map[2] == [
            lines.index("MOVE_NEAR_XY nearX=5.000mm nearY=5.000mm")
        ]
        assert op_map[4] == list(range(attr1, last_layer))
        assert op_map[5] == [
            lines.index("CUT_NEAR_XY nearX=5.000mm nearY=3.000mm")
        ]
        assert op_map[7] == [last_layer, select0, select1, end_job, eof]

        for op_index, block in op_map.items():
            for line_num in block:
                assert result.op_map.machine_code_to_op[line_num] == op_index

        mapped = set(result.op_map.machine_code_to_op.keys())
        assert mapped == set(range(len(lines)))


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


def _plan_job(doc):
    """Return Ops for a small two-layer plan test job."""
    ops = Ops()
    ops.job_start()
    ops.layer_start(layer_uid=doc.layers[0].uid)
    ops.set_power(0.5)
    ops.move_to(5.0, 5.0, 0.0)
    ops.line_to(10.0, 8.0, 0.0)
    ops.layer_end(layer_uid=doc.layers[0].uid)
    ops.layer_start(layer_uid=doc.layers[1].uid)
    ops.set_feed_rate(200)
    ops.move_to(20.0, 20.0, 0.0)
    ops.layer_end(layer_uid=doc.layers[1].uid)
    ops.job_end()
    return ops


class TestGluescriptPlan:
    """The encoder records its GlueScript call plan for re-staging."""

    def test_encode_populates_rpa_plan(self, encoder, mock_machine, doc):
        """encode() must attach the recorded plan to the output."""
        result = encoder.encode(_plan_job(doc), mock_machine, doc)
        assert result.rpa_plan is not None
        assert len(result.rpa_plan) > 0

    def test_empty_ops_have_no_plan(self, encoder, mock_machine, doc):
        """An empty job produces no plan (no GlueScript calls)."""
        result = encoder.encode(Ops(), mock_machine, doc)
        assert result.rpa_plan is None

    def test_plan_starts_with_declare_job_and_ends_with_end_job(
        self, encoder, mock_machine, doc
    ):
        """The plan frames the job exactly like the driver transcript."""
        result = encoder.encode(_plan_job(doc), mock_machine, doc)
        assert result.rpa_plan[0][0] == "declare_job"
        assert result.rpa_plan[-1] == ("end_job", ())

    def test_plan_records_structural_and_raw_calls(
        self, encoder, mock_machine, doc
    ):
        """Structural calls and power_range raw lines are recorded."""
        result = encoder.encode(_plan_job(doc), mock_machine, doc)
        names = [name for name, _ in result.rpa_plan]
        assert "declare_layer" in names
        assert "move_xy_to" in names
        assert "cut_xy_to" in names
        assert "power_range" in names

    def test_plan_args_are_positional_tuples(self, encoder, mock_machine, doc):
        """Recorded args are plain positional tuples (no kwargs)."""
        result = encoder.encode(_plan_job(doc), mock_machine, doc)
        for name, args in result.rpa_plan:
            assert isinstance(name, str)
            assert isinstance(args, tuple)

    def test_plan_replays_without_re_recording(
        self, encoder, mock_machine, doc
    ):
        """gluescript_plan() returns the same snapshot each call."""
        encoder.encode(_plan_job(doc), mock_machine, doc)
        first = encoder.gluescript_plan()
        second = encoder.gluescript_plan()
        assert first == second

    def test_plan_survives_encoder_reuse(self, encoder, mock_machine, doc):
        """A new encode replaces the plan; the old snapshot stays valid."""
        result1 = encoder.encode(_plan_job(doc), mock_machine, doc)
        plan1 = result1.rpa_plan
        result2 = encoder.encode(_plan_job(doc), mock_machine, doc)
        assert result2.rpa_plan is not None
        assert result2.rpa_plan == plan1

    def test_plan_records_per_op_settings_as_gluescript_calls(
        self, encoder, mock_machine, doc
    ):
        """Per-op settings record as gluescript calls in the plan."""
        result = encoder.encode(_plan_job(doc), mock_machine, doc)
        assert ("cut_speed", (3.3333333333333335,)) in result.rpa_plan


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
        assert result.rpa_plan[0] == (
            "declare_job",
            (self._JOB_LABEL, expected, None, 1, 1, 0.0, 0.0),
        )
        if wcs == "ANCHOR":
            # ANCHOR maps to ABSOLUTE with abs_xy pinned to None.
            assert result.rpa_plan[0][1][2] is None

    def test_g54_falls_back_to_machine_ref_point(
        self, encoder, mock_machine, doc
    ):
        """The framework default G54 must fall back to MACHINE."""
        mock_machine.active_wcs = "G54"
        result = encoder.encode(_plan_job(doc), mock_machine, doc)
        assert result.rpa_plan[0] == (
            "declare_job",
            (self._JOB_LABEL, "MACHINE", None, 1, 1, 0.0, 0.0),
        )

    def test_machine_none_defaults_to_machine_ref_point(self, encoder, doc):
        """machine=None must default to the MACHINE reference point."""
        result = encoder.encode(_plan_job(doc), None, doc)
        assert result.rpa_plan[0] == (
            "declare_job",
            (self._JOB_LABEL, "MACHINE", None, 1, 1, 0.0, 0.0),
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
