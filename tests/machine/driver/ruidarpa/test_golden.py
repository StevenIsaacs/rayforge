"""Golden-file regression tests for the ruidarpa encoder staging.

The committed fixture ``golden/staged_job.rpas`` locks the encoder's
staged rpascript byte-for-byte. The representative job covers layer
attribute blocks and per-op action lines across vector cuts and moves,
a linearized arc, a scan line, power clamping, and the air-assist path.

Regenerate the fixture with ``golden/regen_golden.py`` when the encoder
or upstream GlueScript staging legitimately changes the output; do not
weaken these tests to mask drift.
"""

import importlib.util
from pathlib import Path
from typing import Any

from rayforge.machine.driver.ruidarpa.rpa_encoder import RuidaRPAEncoder

GOLDEN_DIR = Path(__file__).parent / "golden"
GOLDEN_FILE = GOLDEN_DIR / "staged_job.rpas"


def _load_regen_builder():
    """Load build_representative_job from the regen script module."""
    module_path = GOLDEN_DIR / "regen_golden.py"
    spec = importlib.util.spec_from_file_location("regen_golden", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load regen script: {module_path}")
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_representative_job


build_representative_job = _load_regen_builder()


def _encode_representative_job():
    """Encode the representative job exactly as the fixture was built."""
    doc, ops, machine = build_representative_job()
    result = RuidaRPAEncoder().encode(ops, machine, doc)
    return result.text


class TestGoldenFixture:
    """Byte-level lock on the committed staged-job fixture."""

    def test_staged_output_matches_fixture(self):
        """Encoder output must equal the fixture, byte for byte."""
        expected = GOLDEN_FILE.read_text(encoding="utf-8")
        assert _encode_representative_job() + "\n" == expected

    def test_staged_output_is_deterministic(self):
        """Two encodes of the same job must produce identical bytes."""
        first = _encode_representative_job()
        second = _encode_representative_job()
        assert first == second


class TestGoldenStructure:
    """Structural invariants independent of exact fixture bytes."""

    def test_no_enable_block_cutting_leaks_into_output(self):
        """ENABLE_BLOCK_CUTTING must not leak into the staged output."""
        assert "ENABLE_BLOCK_CUTTING" not in _encode_representative_job()

    def test_single_job_framing(self):
        """The output is self-contained: one ref-point/start block."""
        lines = _encode_representative_job().split("\n")
        assert lines.count("REF_POINT_ABSOLUTE") == 1
        assert lines.count("REF_POINT_SET") == 1
        assert lines.count("START_JOB") == 1
        assert lines.count("LAST_LAYER Layer:2") == 1
        assert lines.count("END_JOB") == 1
        assert lines.count("EOF") == 1

    def test_ends_with_end_job_and_eof(self):
        """The staged script terminates with END_JOB then the EOF marker."""
        lines = _encode_representative_job().split("\n")
        assert lines[-2] == "END_JOB"
        assert lines[-1] == "EOF"

    def test_layer_attribute_blocks(self):
        """Layer attrs stage with workflow settings, clamps, and defaults."""
        text = _encode_representative_job()
        assert "# Layer 0: Cut" in text
        assert "# Layer 1: Engrave" in text
        assert "# Layer 2: Default" in text
        assert "LAYER_MIN_POWER_1 Layer:0 Power:50.0%" in text
        assert "LAYER_MIN_POWER_1 Layer:1 Power:8.0%" in text
        assert "LAYER_SPEED_LASER_1 Layer:2 Speed:100.0mm/S" in text
        assert "LAYER_MIN_POWER_1 Layer:2 Power:20.0%" in text

    def test_layer_action_blocks(self):
        """Per-op settings emit their action lines once per layer."""
        text = _encode_representative_job()
        assert text.count("SELECT_LAYER Layer:0") == 1
        assert text.count("SELECT_LAYER Layer:1") == 1
        assert text.count("SELECT_LAYER Layer:2") == 1
        assert "MIN_POWER_1 Power=8.0%" in text
        assert "SPEED_LASER_1 Speed=250.000mm/S" in text
        assert "LAYER_FREQUENCY Laser=1 Layer=0 Freq=25.000KHz" in text
        assert "LASER_INTERVAL 0.050mS" in text
        assert "AIR_ASSIST_ON" in text
        assert "AIR_ASSIST_OFF" in text
        assert "delay 250.000ms" in text

    def test_moves_cuts_arc_and_scan_present(self):
        """Near/far moves and cuts, an arc, and a scan line all stage."""
        text = _encode_representative_job()
        assert "MOVE_NEAR_XY X=0.000mm Y=0.000mm" in text
        assert "MOVE_FAR_XY X=20.000mm Y=20.000mm" in text
        assert "CUT_NEAR_XY X=5.000mm Y=5.000mm" in text
        assert "CUT_FAR_XY X=50.000mm Y=20.000mm" in text
        assert text.count("CUT_NEAR_XY") > 5
        assert "MIN_POWER_1 Power=50.2%" in text
        assert "MIN_POWER_1 Power=100.0%" in text
