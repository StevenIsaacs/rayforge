"""
Unit tests for the RpaDirectDriver live-command wrappers.

The wrappers delegate to the wrapped RdDriver and discard the returned
lines, which the driver auto-sends when connected. Calling ``run()`` on
those lines would double-send every live command, so the tests pin
``run`` as never-called and the wrapper return value as None.
"""

from unittest.mock import Mock

import pytest

from ruidadriver.ruida_driver import RdDriver

from rayforge.machine.driver.ruidarpa.rpa_direct_driver import RpaDirectDriver

_LIVE_METHODS = [
    ("home", ()),
    ("home_z", ()),
    ("jog_xy_to", (10.0, 20.0)),
    ("jog_xy_rel", (5.0, 5.0)),
    ("jog_x_rel", (5.0,)),
    ("jog_y_rel", (5.0,)),
    ("jog_z_rel", (5.0,)),
    ("jog_u_rel", (5.0,)),
    ("pause", ()),
    ("resume", ()),
    ("stop_job", ()),
    ("reset", ()),
]

_SPEED_METHODS = [
    ("jog_set_xy_speed", (100.0,)),
    ("jog_set_z_speed", (100.0,)),
    ("jog_set_u_speed", (100.0,)),
]


def _direct_driver(connected: bool) -> tuple[RpaDirectDriver, Mock]:
    """Return an RpaDirectDriver wrapping a mock RdDriver."""
    driver = RpaDirectDriver()
    mock_driver = Mock(spec=RdDriver)
    mock_driver.is_connected = connected
    driver._driver = mock_driver
    return driver, mock_driver


class TestLiveWrapperDelegation:
    """Live-command wrappers delegate and never run the returned lines."""

    @pytest.mark.parametrize("method,args", _LIVE_METHODS)
    def test_delegates_and_never_runs_returned_lines(self, method, args):
        """The wrapper calls the RdDriver method and discards its lines."""
        driver, mock_driver = _direct_driver(connected=True)
        mock_method = getattr(mock_driver, method)
        mock_method.return_value = ["SPEED_LASER_1 100.0"]

        result = getattr(driver, method)(*args)

        mock_method.assert_called_once_with(*args)
        mock_driver.run.assert_not_called()
        assert result is None

    @pytest.mark.parametrize("method,args", _LIVE_METHODS)
    def test_raises_when_disconnected(self, method, args):
        """Jog, home, and job-control commands must fail loudly instead
        of silently no-oping.
        """
        driver, _mock_driver = _direct_driver(connected=False)

        with pytest.raises(RuntimeError, match="not connected"):
            getattr(driver, method)(*args)


class TestRequireConnected:
    """_require_connected gates jog/home on a live session."""

    def test_raises_when_disconnected(self):
        """A disconnected driver raises rather than returning None lines."""
        driver, _mock_driver = _direct_driver(connected=False)

        with pytest.raises(RuntimeError, match="not connected"):
            driver._require_connected()

    def test_returns_driver_when_connected(self):
        """A connected driver is returned for direct delegation."""
        driver, mock_driver = _direct_driver(connected=True)

        assert driver._require_connected() is mock_driver


class TestSpeedSetterDelegation:
    """jog_set_*_speed setters delegate without requiring a connection."""

    @pytest.mark.parametrize("method,args", _SPEED_METHODS)
    def test_speed_setter_delegates_while_disconnected(self, method, args):
        """Speed is session-less state, so the setter works disconnected."""
        driver, mock_driver = _direct_driver(connected=False)

        getattr(driver, method)(*args)

        getattr(mock_driver, method).assert_called_once_with(*args)

    @pytest.mark.parametrize("method,args", _SPEED_METHODS)
    def test_speed_setter_never_runs_lines(self, method, args):
        """A speed setter must never touch the script runner."""
        driver, mock_driver = _direct_driver(connected=True)

        getattr(driver, method)(*args)

        mock_driver.run.assert_not_called()
