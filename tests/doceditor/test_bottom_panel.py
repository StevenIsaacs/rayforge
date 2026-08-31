"""Tests for the BottomPanel move-to-position corner selection.

The physical lower-left corner of a selection is always world
(min_x, min_y) and the upper-right is always (max_x, max_y),
independent of the machine origin. The origin only affects the
world-to-machine conversion, which is handled separately by
CoordinateSpace.world_point_to_machine.
"""

import pytest

from rayforge.ui_gtk.doceditor.bottom_panel import BottomPanel

MIN_X, MIN_Y, MAX_X, MAX_Y = 1.0, 2.0, 10.0, 20.0


@pytest.mark.parametrize(
    "position, expected",
    [
        ("ll", (MIN_X, MIN_Y)),
        ("ur", (MAX_X, MAX_Y)),
        ("center", ((MIN_X + MAX_X) / 2, (MIN_Y + MAX_Y) / 2)),
    ],
)
def test_world_corner_for_position(position, expected):
    result = BottomPanel._world_corner_for_position(
        position, MIN_X, MIN_Y, MAX_X, MAX_Y
    )
    assert result == expected


def test_world_corner_for_position_unknown_returns_none():
    result = BottomPanel._world_corner_for_position(
        "bogus", MIN_X, MIN_Y, MAX_X, MAX_Y
    )
    assert result is None
