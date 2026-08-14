"""
Tests for the MachineController class.

This module tests the MachineController which handles:
- Driver lifecycle management (connect/disconnect/shutdown)
- Command execution (jog, home, run_raw, etc.)
- Signal emissions for state changes

The MachineController is the logic layer that owns and manages the driver.
"""

import pytest

from rayforge.machine.driver.dummy import NoDeviceDriver
from rayforge.machine.models.controller import MachineController
from rayforge.machine.models.machine import Machine
from rayforge.machine.transport import TransportStatus
from rayforge.shared.tasker import task_mgr


class _SelectWcsRecordingDriver(NoDeviceDriver):
    """Fake driver that overrides select_wcs and records routing calls."""

    @property
    def supported_wcs(self):
        return ["MACHINE", "ANCHOR", "CURRENT", "SET_POINT"]

    def __init__(self, context, machine):
        super().__init__(context, machine)
        self.select_wcs_calls = []
        self.run_raw_calls = []
        self.offset_reads = 0

    async def select_wcs(self, wcs):
        self.select_wcs_calls.append(wcs)

    async def run_raw(self, machine_code):
        self.run_raw_calls.append(machine_code)

    async def read_parser_state(self):
        if self.select_wcs_calls:
            return self.select_wcs_calls[-1]
        return self._machine.active_wcs

    async def read_wcs_offsets(self):
        self.offset_reads += 1
        return {}


class _RunRawOnlyDriver(NoDeviceDriver):
    """Fake driver that leaves select_wcs as the inherited base no-op."""

    @property
    def supported_wcs(self):
        return ["MACHINE", "ANCHOR", "CURRENT", "SET_POINT"]

    def __init__(self, context, machine):
        super().__init__(context, machine)
        self.run_raw_calls = []

    async def run_raw(self, machine_code):
        self.run_raw_calls.append(machine_code)


class _NeverConfirmingDriver(_SelectWcsRecordingDriver):
    """Fake driver whose parser state never confirms a WCS switch."""

    async def read_parser_state(self):
        return "MACHINE"


@pytest.mark.usefixtures("lite_context")
class TestMachineController:
    """Test suite for the MachineController class."""

    def test_controller_initialization(self, lite_context):
        """Test that MachineController can be initialized."""
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        controller = MachineController(
            machine, lite_context, task_mgr.schedule_on_main_thread
        )
        assert controller is not None
        assert controller.machine == machine
        assert controller.context == lite_context
        assert controller.driver is not None

    def test_controller_driver_property(self, lite_context):
        """Test that the controller has a driver property."""
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        controller = machine.controller
        assert controller.driver is not None

    def test_controller_signals_exist(self, lite_context):
        """Test that controller has all required signals."""
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        controller = machine.controller
        assert hasattr(controller, "connection_status_changed")
        assert hasattr(controller, "state_changed")
        assert hasattr(controller, "job_finished")
        assert hasattr(controller, "command_status_changed")
        assert hasattr(controller, "wcs_updated")

    def _connected_machine(self, lite_context, driver_cls):
        """Create a machine whose controller owns the given fake driver."""
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        controller = machine.controller
        driver = driver_cls(lite_context, machine)
        controller.driver = driver
        machine.connection_status = TransportStatus.CONNECTED
        return machine, controller, driver

    @pytest.mark.asyncio
    async def test_switch_active_wcs_routes_to_select_wcs_when_overridden(
        self, lite_context
    ):
        """A driver overriding select_wcs must receive the WCS selection."""
        machine, controller, driver = self._connected_machine(
            lite_context, _SelectWcsRecordingDriver
        )

        await controller.switch_active_wcs("MACHINE")

        assert driver.select_wcs_calls == ["MACHINE"]
        assert driver.run_raw_calls == []
        assert machine.active_wcs == "MACHINE"
        assert controller._confirmed_active_wcs == "MACHINE"
        assert driver.offset_reads == 1

    @pytest.mark.asyncio
    async def test_switch_active_wcs_routes_to_run_raw_when_not_overridden(
        self, lite_context
    ):
        """A base select_wcs must route the raw WCS command instead."""
        machine, controller, driver = self._connected_machine(
            lite_context, _RunRawOnlyDriver
        )

        await controller.switch_active_wcs("MACHINE")

        assert driver.run_raw_calls == ["MACHINE"]
        assert machine.active_wcs == "MACHINE"
        assert controller._confirmed_active_wcs == "MACHINE"

    @pytest.mark.asyncio
    async def test_switch_active_wcs_invalid_raises_without_mutation(
        self, lite_context
    ):
        """A rejected WCS must raise before mutating model state."""
        machine, controller, driver = self._connected_machine(
            lite_context, _SelectWcsRecordingDriver
        )
        machine.active_wcs = "MACHINE"
        controller._confirmed_active_wcs = "MACHINE"

        with pytest.raises(ValueError, match="MACHINE"):
            await controller.switch_active_wcs("G55")

        assert machine.active_wcs == "MACHINE"
        assert controller._confirmed_active_wcs == "MACHINE"
        assert driver.select_wcs_calls == []

    @pytest.mark.asyncio
    async def test_switch_active_wcs_disconnected_records_intent(
        self, lite_context
    ):
        """A disconnected machine records the intent without a device."""
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        controller = machine.controller

        await controller.switch_active_wcs("MACHINE")

        assert machine.active_wcs == "MACHINE"
        assert controller._confirmed_active_wcs == "MACHINE"

    @pytest.mark.asyncio
    async def test_switch_active_wcs_confirm_failure_keeps_intent(
        self, lite_context
    ):
        """A failed device confirmation records intent without syncing."""
        machine, controller, driver = self._connected_machine(
            lite_context, _NeverConfirmingDriver
        )

        await controller.switch_active_wcs("CURRENT")

        assert machine.active_wcs == "CURRENT"
        assert controller._confirmed_active_wcs is None
        assert driver.offset_reads == 0
