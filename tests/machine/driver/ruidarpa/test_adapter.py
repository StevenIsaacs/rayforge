"""
Adapter tests for the RuidaRPAAdapter.

The adapter wraps either RpaDirectDriver (direct mode) or RpaRpcClient
(RPC/TUI mode) as ``_backend``. These tests mock the backend and verify
the adapter's public surface:

- Stop/cleanup regression: backend stop()/disconnect() must actually run
- run routing for jobs and runtime commands
- Jog speed tracking: move_to() reuses the last jog() speed (default 600
  before any jog) and a speed-only jog updates the stored speed
- Live bridge behavior per mode (no double-run in RPC mode)
- Fail-loud: backend jog/home failures propagate through the adapter
- set_wcs_offset failing loud while select_wcs still routes to run()
- mm fix: status positions are not divided by 1000
- Reconnect listener hygiene (unregister-before-register in direct mode)
- Connect-time head/tail clearing: TUI RPC clears the server driver's
  head/tail scripts to [] while direct mode never touches them
- Paren-less backend property reads (``is_connected``)

No real network or Ruida hardware is used; the backends are
``unittest.mock`` mocks spec'd against the real backend classes.
"""

import asyncio
import contextlib
import logging
from dataclasses import replace
from typing import Callable
from unittest.mock import Mock, call

import pytest
import pytest_asyncio
from raygeo.ops import Ops

from rayforge.core.doc import Doc
from rayforge.core.varset import FloatVar
from rayforge.machine.driver.driver import (
    Axis,
    DeviceStatus,
    Driver,
    DriverSetupError,
)
from rayforge.machine.driver.ruidarpa import rpa_adapter
from rayforge.machine.driver.ruidarpa.rpa_adapter import (
    DEFAULT_MAX_CUT_SPEED_MMPM,
    DEFAULT_MAX_TRAVEL_SPEED_MMPM,
    DEFAULT_RPC_TIMEOUT_S,
    RuidaRPAAdapter,
    _stage_plan,
    _unwrap_mm,
)
from rayforge.machine.driver.ruidarpa.rpa_direct_driver import (
    RpaDirectDriver,
)
from rayforge.machine.driver.ruidarpa.rpa_rpc_client import RpaRpcClient
from rayforge.machine.models.laser import Laser
from rayforge.machine.transport import TransportStatus
from rayforge.pipeline.encoder.base import EncodedOutput, MachineCodeOpMap

DIRECT_MODE = False
RPC_MODE = True


class _DeadTransportClient(Mock):
    """RPC client mock whose ``is_connected`` read raises.

    Models an RPyC transport that is already dead: the health probe
    fails immediately instead of returning a value.
    """

    @property
    def is_connected(self):
        raise RuntimeError("transport dead")


async def _wait_until(condition: Callable[[], bool], timeout: float = 2.0):
    """Poll ``condition`` until true, failing the test on timeout."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        if loop.time() > deadline:
            pytest.fail("Timed out waiting for condition")
        await asyncio.sleep(0.01)


async def _run_connect_cycle(adapter: RuidaRPAAdapter, condition):
    """Run one connection-loop cycle through the real connect path.

    Starts ``_connect_implementation`` (the reconnection loop), waits
    until ``condition`` holds, then cancels the loop task so tests finish
    deterministically without depending on the 0.5s poll interval.
    """
    adapter._keep_running = True
    await adapter._connect_implementation()
    try:
        await _wait_until(condition)
    finally:
        adapter._keep_running = False
        task = adapter._connection_task
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


@pytest_asyncio.fixture
async def adapter_pair(isolated_context, isolated_machine, request):
    """Create a RuidaRPAAdapter whose backend is a mock for the mode.

    Parametrize with ``indirect=True`` over ``DIRECT_MODE`` (False) and
    ``RPC_MODE`` (True). Yields ``(adapter, backend_mock)``.

    Uses the isolated (mock-context) fixtures because the adapter only
    holds context/machine references; no TaskManager or context
    singleton is needed, keeping each test fast.
    """
    tui_mode = request.param
    machine = isolated_machine

    adapter = RuidaRPAAdapter(isolated_context, machine)
    adapter.setup(udp_host="127.0.0.1", tui=tui_mode)

    backend_cls = RpaRpcClient if tui_mode else RpaDirectDriver
    backend = Mock(spec=backend_cls)
    backend.start.return_value = True
    backend.is_connected = True
    backend.machine_status = {}
    if tui_mode:
        # The RPC client has no unregister surface; expose tracked mocks
        # so tests can assert the adapter never calls them.
        backend.connect.return_value = True
        backend.is_alive.return_value = True
        backend.unregister_status_listener = Mock()
        backend.unregister_error_listener = Mock()
        backend.unregister_reply_listener = Mock()
    adapter._backend = backend

    yield adapter, backend

    await adapter.cleanup()
    await machine.shutdown()


class TestClassAttributes:
    def test_supports_travel_speed(self):
        assert RuidaRPAAdapter.supports_travel_speed is True

    def test_supports_travel_speed_default_false(self):
        assert Driver.supports_travel_speed is False


class TestStopBackendRegression:
    """Stop/disconnect must actually reach the backend (core bug fix)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [RPC_MODE], ids=["rpc"], indirect=True
    )
    async def test_stop_rpc_calls_stop_and_disconnect(self, adapter_pair):
        """RPC stop must call client.stop() and client.disconnect()."""
        adapter, client = adapter_pair
        await adapter._stop_backend()
        client.stop.assert_called_once()
        client.disconnect.assert_called_once()
        assert adapter._backend is client

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [RPC_MODE], ids=["rpc"], indirect=True
    )
    async def test_stop_rpc_never_calls_unregister(self, adapter_pair):
        """RPC stop must not attempt any unregister_* calls."""
        adapter, client = adapter_pair
        await adapter._stop_backend()
        client.unregister_status_listener.assert_not_called()
        client.unregister_error_listener.assert_not_called()
        client.unregister_reply_listener.assert_not_called()
        assert adapter._backend is client

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [DIRECT_MODE], ids=["direct"], indirect=True
    )
    async def test_stop_direct_unregisters_stored_refs_then_stops(
        self, adapter_pair
    ):
        """Direct stop must unregister the stored listener refs, then stop."""
        adapter, backend = adapter_pair
        await adapter._stop_backend()
        backend.unregister_status_listener.assert_called_once_with(
            adapter._on_rpa_status
        )
        backend.unregister_error_listener.assert_called_once_with(
            adapter._on_rpa_error
        )
        backend.unregister_reply_listener.assert_called_once_with(
            adapter._on_rpa_reply
        )
        backend.stop.assert_called_once()
        assert adapter._backend is backend

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [RPC_MODE], ids=["rpc"], indirect=True
    )
    async def test_stop_rpc_stop_raises_disconnect_still_called(
        self, adapter_pair
    ):
        """A raising stop() must not skip disconnect() (dead transport)."""
        adapter, client = adapter_pair
        client.stop.side_effect = RuntimeError("transport dead")
        await adapter._stop_backend()
        client.stop.assert_called_once()
        client.disconnect.assert_called_once()
        assert adapter._backend is client

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [RPC_MODE], ids=["rpc"], indirect=True
    )
    async def test_stop_rpc_is_connected_raises_disconnect_still_called(
        self, adapter_pair
    ):
        """A raising is_connected read must still reach disconnect()."""
        adapter, _client = adapter_pair
        client = _DeadTransportClient(spec=RpaRpcClient)
        adapter._backend = client
        await adapter._stop_backend()
        client.stop.assert_not_called()
        client.disconnect.assert_called_once()
        assert adapter._backend is client

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [DIRECT_MODE], ids=["direct"], indirect=True
    )
    async def test_stop_direct_stop_raises_cleanup_completes(
        self, adapter_pair
    ):
        """A raising stop() must not erase the preceding unregisters."""
        adapter, backend = adapter_pair
        backend.stop.side_effect = RuntimeError("transport dead")
        await adapter._stop_backend()
        backend.unregister_status_listener.assert_called_once_with(
            adapter._on_rpa_status
        )
        backend.unregister_error_listener.assert_called_once_with(
            adapter._on_rpa_error
        )
        backend.unregister_reply_listener.assert_called_once_with(
            adapter._on_rpa_reply
        )
        backend.stop.assert_called_once()
        assert adapter._backend is backend

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [DIRECT_MODE], ids=["direct"], indirect=True
    )
    async def test_stop_direct_unregister_raises_remaining_cleanup_runs(
        self, adapter_pair
    ):
        """A raising unregister must not skip the remaining cleanup."""
        adapter, backend = adapter_pair
        backend.unregister_status_listener.side_effect = RuntimeError(
            "listener registry dead"
        )
        await adapter._stop_backend()
        backend.unregister_error_listener.assert_called_once_with(
            adapter._on_rpa_error
        )
        backend.unregister_reply_listener.assert_called_once_with(
            adapter._on_rpa_reply
        )
        backend.stop.assert_called_once()
        assert adapter._backend is backend

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [RPC_MODE], ids=["rpc"], indirect=True
    )
    async def test_cleanup_rpc_stops_and_disconnects(self, adapter_pair):
        """cleanup() must reach client.stop()/disconnect()."""
        adapter, client = adapter_pair
        await adapter.cleanup()
        client.stop.assert_called_once()
        client.disconnect.assert_called_once()
        assert adapter._backend is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [DIRECT_MODE], ids=["direct"], indirect=True
    )
    async def test_cleanup_direct_stops_backend(self, adapter_pair):
        """cleanup() must reach driver.stop()."""
        adapter, backend = adapter_pair
        await adapter.cleanup()
        backend.stop.assert_called_once()
        assert adapter._backend is None


class TestIsConnectedGating:
    """Backend ``is_connected`` is read paren-less and gates stop cleanup."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [RPC_MODE], ids=["rpc"], indirect=True
    )
    async def test_stop_rpc_disconnected_disconnects_without_stop(
        self, adapter_pair
    ):
        """A disconnected client skips stop() but still disconnects()."""
        adapter, client = adapter_pair
        client.is_connected = False
        await adapter._stop_backend()
        client.stop.assert_not_called()
        client.disconnect.assert_called_once()
        assert adapter._backend is client

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [DIRECT_MODE], ids=["direct"], indirect=True
    )
    async def test_stop_direct_disconnected_stops_without_unregister(
        self, adapter_pair
    ):
        """A disconnected driver skips unregister but still stops()."""
        adapter, backend = adapter_pair
        backend.is_connected = False
        await adapter._stop_backend()
        backend.unregister_status_listener.assert_not_called()
        backend.unregister_error_listener.assert_not_called()
        backend.unregister_reply_listener.assert_not_called()
        backend.stop.assert_called_once()
        assert adapter._backend is backend


class TestRunRouting:
    """Jobs and runtime commands route through backend run()."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair",
        [DIRECT_MODE, RPC_MODE],
        ids=["direct", "rpc"],
        indirect=True,
    )
    async def test_run_script_routes_to_backend_run(self, adapter_pair):
        """_run_script must call backend.run without job framing."""
        adapter, backend = adapter_pair
        await adapter._run_script(["PAUSE_JOB"])
        backend.run.assert_called_once_with(["PAUSE_JOB"], False)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [DIRECT_MODE], ids=["direct"], indirect=True
    )
    async def test_run_direct_mode_routes_text_through_run(self, adapter_pair):
        """run() in direct mode must route the text through backend.run."""
        adapter, backend = adapter_pair
        encoded = EncodedOutput(
            text="HOME_XY\nMOVE_NEAR_XY X=1.000mm Y=1.000mm",
            op_map=MachineCodeOpMap(),
        )
        await adapter.run(encoded, Doc(), Ops())
        backend.run.assert_called_once_with(
            ["HOME_XY", "MOVE_NEAR_XY X=1.000mm Y=1.000mm"], True
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [RPC_MODE], ids=["rpc"], indirect=True
    )
    async def test_run_rpc_stages_plan_and_runs_staged_job(self, adapter_pair):
        """run() in RPC mode must re-stage the plan server-side."""
        adapter, client = adapter_pair
        plan = [
            ("declare_job", ("Job", "MACHINE", None, 1, 1, 0.0, 0.0)),
            (
                "declare_layer",
                ("Cut", "#ff6600", "VECTOR", "NONE", 100.0, 20.0, 50.0, 50.0),
            ),
            ("add_layer_action", (1, ["MIN_POWER_1 Power=50.0%"])),
            ("move_xy_to", (5.0, 5.0)),
            ("end_job", ()),
        ]
        encoded = EncodedOutput(
            text="\n".join(
                ["REF_POINT_SET", "MOVE_NEAR_XY X=5.000mm Y=5.000mm"]
            ),
            op_map=MachineCodeOpMap(),
            driver_data={"rpa_plan": plan},
        )
        await adapter.run(encoded, Doc(), Ops())
        client.root.exposed_new_gluescript.assert_called_once_with()
        deltas = client.root.exposed_stage_gluescript_delta
        assert deltas.call_count == 2
        first = deltas.call_args_list[0]
        assert first.args[0] == 0
        assert first.args[1] == [
            "declare_job('Job', 'MACHINE', None, 1, 1, 0.0, 0.0)",
            "declare_layer('Cut', '#ff6600', 'VECTOR', 'NONE', "
            "100.0, 20.0, 50.0, 50.0)",
        ]
        assert first.kwargs["require_complete"] is False
        client.root.exposed_add_layer_action.assert_called_once_with(
            1, ["MIN_POWER_1 Power=50.0%"]
        )
        final = deltas.call_args_list[-1]
        assert final.args[0] == 2
        assert final.args[1] == ["move_xy_to(5.0, 5.0)", "end_job()"]
        assert final.kwargs["require_complete"] is True
        client.run_staged_job.assert_called_once_with()
        client.run.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [RPC_MODE], ids=["rpc"], indirect=True
    )
    async def test_run_rpc_without_plan_falls_back_to_text(self, adapter_pair):
        """run() in RPC mode without a plan must ship text via backend.run
        (drop-in compatibility with the direct driver)."""
        adapter, client = adapter_pair
        encoded = EncodedOutput(
            text="HOME_XY\nMOVE_NEAR_XY X=1.000mm Y=1.000mm",
            op_map=MachineCodeOpMap(),
        )
        await adapter.run(encoded, Doc(), Ops())
        client.run.assert_called_once_with(
            ["HOME_XY", "MOVE_NEAR_XY X=1.000mm Y=1.000mm"], True
        )
        client.run_staged_job.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair",
        [DIRECT_MODE, RPC_MODE],
        ids=["direct", "rpc"],
        indirect=True,
    )
    async def test_run_with_empty_text_skips_backend(self, adapter_pair):
        """Empty output must skip backend dispatch but still signal
        job_finished."""
        adapter, backend = adapter_pair
        encoded = EncodedOutput(text="", op_map=MachineCodeOpMap())
        fired = []

        def _record_finished(sender):
            fired.append(sender)

        adapter.job_finished.connect(_record_finished)
        await adapter.run(encoded, Doc(), Ops())
        assert fired == [adapter]
        backend.run.assert_not_called()
        run_staged_job = getattr(backend, "run_staged_job", None)
        if run_staged_job is not None:
            run_staged_job.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [RPC_MODE], ids=["rpc"], indirect=True
    )
    async def test_run_rpc_empty_plan_is_present_routes_to_staged(
        self, adapter_pair
    ):
        """run() in RPC mode must treat an empty list plan as present and
        route to staged, never to raw text."""
        adapter, client = adapter_pair
        encoded = EncodedOutput(
            text="HOME_XY\nMOVE_NEAR_XY X=1.000mm Y=1.000mm",
            op_map=MachineCodeOpMap(),
            driver_data={"rpa_plan": []},
        )
        await adapter.run(encoded, Doc(), Ops())
        client.run_staged_job.assert_called_once_with()
        client.run.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair",
        [DIRECT_MODE, RPC_MODE],
        ids=["direct", "rpc"],
        indirect=True,
    )
    async def test_run_raw_routes_through_run_with_checksum(
        self, adapter_pair
    ):
        """run_raw() must route through _run_script with auto_checksum=True."""
        adapter, backend = adapter_pair
        await adapter.run_raw("HOME_XY\nMOVE_NEAR_XY X=1.000mm Y=1.000mm")
        backend.run.assert_called_once_with(
            ["HOME_XY", "MOVE_NEAR_XY X=1.000mm Y=1.000mm"], True
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair",
        [DIRECT_MODE, RPC_MODE],
        ids=["direct", "rpc"],
        indirect=True,
    )
    async def test_set_hold_pause_calls_backend_pause(self, adapter_pair):
        """set_hold(True) must pause via the live backend."""
        adapter, backend = adapter_pair
        await adapter.set_hold(True)
        backend.pause.assert_called_once_with()
        backend.run.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair",
        [DIRECT_MODE, RPC_MODE],
        ids=["direct", "rpc"],
        indirect=True,
    )
    async def test_set_hold_resume_calls_backend_resume(self, adapter_pair):
        """set_hold(False) must resume via the live backend."""
        adapter, backend = adapter_pair
        await adapter.set_hold(False)
        backend.resume.assert_called_once_with()
        backend.run.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair",
        [DIRECT_MODE, RPC_MODE],
        ids=["direct", "rpc"],
        indirect=True,
    )
    async def test_cancel_routes_to_stop_job(self, adapter_pair):
        """cancel() must stop the job via the live backend."""
        adapter, backend = adapter_pair
        await adapter.cancel()
        backend.stop_job.assert_called_once_with()
        backend.run.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair",
        [DIRECT_MODE, RPC_MODE],
        ids=["direct", "rpc"],
        indirect=True,
    )
    async def test_clear_alarm_routes_to_stop_job(self, adapter_pair):
        """clear_alarm() must stop the job via the live backend."""
        adapter, backend = adapter_pair
        await adapter.clear_alarm()
        backend.stop_job.assert_called_once_with()
        backend.run.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair",
        [DIRECT_MODE, RPC_MODE],
        ids=["direct", "rpc"],
        indirect=True,
    )
    async def test_set_power_is_unsupported(self, adapter_pair, caplog):
        """set_power() is unsupported — warn and send no command."""
        caplog.set_level(logging.WARNING, logger=rpa_adapter.logger.name)
        adapter, backend = adapter_pair
        head = Laser()
        await adapter.set_power(head, 0.5)
        backend.run.assert_not_called()
        assert any("set_power" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair",
        [DIRECT_MODE, RPC_MODE],
        ids=["direct", "rpc"],
        indirect=True,
    )
    async def test_set_focus_power_is_unsupported(self, adapter_pair, caplog):
        """set_focus_power() is unsupported — warn and send no command."""
        caplog.set_level(logging.WARNING, logger=rpa_adapter.logger.name)
        adapter, backend = adapter_pair
        head = Laser()
        await adapter.set_focus_power(head, 0.25)
        backend.run.assert_not_called()
        assert any(
            "set_focus_power" in record.message for record in caplog.records
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair",
        [DIRECT_MODE, RPC_MODE],
        ids=["direct", "rpc"],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "wcs", ["MACHINE", "ANCHOR", "CURRENT", "SET_POINT"]
    )
    async def test_select_wcs_valid_saves_selection(self, adapter_pair, wcs):
        """select_wcs() must record a supported WCS without sending scripts."""
        adapter, backend = adapter_pair
        await adapter.select_wcs(wcs)
        assert adapter._selected_wcs == wcs
        backend.run.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair",
        [DIRECT_MODE, RPC_MODE],
        ids=["direct", "rpc"],
        indirect=True,
    )
    @pytest.mark.parametrize("wcs", ["REF0", "REF1", "G55"])
    async def test_select_wcs_invalid_raises_value_error(
        self, adapter_pair, wcs
    ):
        """select_wcs() must reject unknown names without mutating state."""
        adapter, backend = adapter_pair
        with pytest.raises(ValueError, match="MACHINE"):
            await adapter.select_wcs(wcs)
        assert adapter._selected_wcs == "MACHINE"
        backend.run.assert_not_called()


class TestRunStagedJob:
    """The TUI RPC staged pipeline re-stages the plan server-side."""

    PLAN = [
        ("declare_job", ("Job", "MACHINE", None, 1, 1, 0.0, 0.0)),
        (
            "declare_layer",
            ("Cut", "#ff6600", "VECTOR", "NONE", 100.0, 20.0, 50.0, 50.0),
        ),
        ("add_layer_action", (1, ["MIN_POWER_1 Power=50.0%"])),
        ("move_xy_to", (5.0, 5.0)),
        ("end_job", ()),
    ]

    def _encoded(self, plan=PLAN):
        return EncodedOutput(
            text="dummy",
            op_map=MachineCodeOpMap(),
            driver_data={"rpa_plan": plan},
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [RPC_MODE], ids=["rpc"], indirect=True
    )
    async def test_run_staged_job_stages_then_runs(self, adapter_pair):
        """_run_staged_job must replay the plan and run the staged job."""
        adapter, client = adapter_pair
        await adapter._run_staged_job(self._encoded())
        client.root.exposed_new_gluescript.assert_called_once_with()
        assert client.root.exposed_stage_gluescript_delta.call_count == 2
        client.run_staged_job.assert_called_once_with()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [RPC_MODE], ids=["rpc"], indirect=True
    )
    async def test_run_staged_job_missing_plan_raises(self, adapter_pair):
        """A plan-less encoded output must fail fast in RPC mode."""
        adapter, client = adapter_pair
        with pytest.raises(DriverSetupError, match="rpa_plan"):
            await adapter._run_staged_job(self._encoded(plan=None))
        client.root.exposed_new_gluescript.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [RPC_MODE], ids=["rpc"], indirect=True
    )
    async def test_run_staged_job_raises_when_not_initialized(
        self, adapter_pair
    ):
        """A missing backend must fail fast."""
        adapter, client = adapter_pair
        adapter._backend = None
        with pytest.raises(DriverSetupError, match="Backend"):
            await adapter._run_staged_job(self._encoded())


class TestStagePlan:
    """_stage_plan replays the recorded plan through the RPC sink."""

    def test_resets_before_staging(self):
        """The pipeline starts from a clean server-side gluescript."""
        client = Mock(spec=RpaRpcClient)
        _stage_plan(client, [])
        client.root.exposed_new_gluescript.assert_called_once_with()
        # The server-side gluescript cursor tracks the live machine
        # position, so staging must not clobber it with a zero reset.
        client.root.exposed_update_position.assert_not_called()

    def test_single_delta_when_no_raw_actions(self):
        """A plan without add_layer_action flushes once, complete."""
        client = Mock(spec=RpaRpcClient)
        plan = [
            ("declare_job", ("Job", "MACHINE", None, 1, 1, 0.0, 0.0)),
            ("end_job", ()),
        ]
        _stage_plan(client, plan)
        client.root.exposed_stage_gluescript_delta.assert_called_once_with(
            0,
            [
                "declare_job('Job', 'MACHINE', None, 1, 1, 0.0, 0.0)",
                "end_job()",
            ],
            require_complete=True,
        )

    def test_raw_actions_forwarded_at_interleaved_position(self):
        """add_layer_action lines forward in recorded order."""
        client = Mock(spec=RpaRpcClient)
        _stage_plan(client, TestRunStagedJob.PLAN)
        deltas = client.root.exposed_stage_gluescript_delta
        assert deltas.call_count == 2
        first = deltas.call_args_list[0]
        assert first.args[0] == 0
        assert first.args[1][0].startswith("declare_job(")
        assert first.kwargs["require_complete"] is False
        client.root.exposed_add_layer_action.assert_called_once_with(
            1, ["MIN_POWER_1 Power=50.0%"]
        )
        final = deltas.call_args_list[-1]
        assert final.args[0] == 2
        assert final.args[1] == ["move_xy_to(5.0, 5.0)", "end_job()"]
        assert final.kwargs["require_complete"] is True

    def test_failed_stage_resets_staged_state(self):
        """A rejected replay must tear down so no stale job can run."""
        client = Mock(spec=RpaRpcClient)
        client.root.exposed_stage_gluescript_delta.side_effect = RuntimeError(
            "Re-staged gluescript is missing end_job()"
        )
        plan = [
            ("declare_job", ("Job", "MACHINE", None, 1, 1, 0.0, 0.0)),
            ("add_layer_action", (1, ["MIN_POWER_1 Power=50.0%"])),
        ]
        with pytest.raises(RuntimeError, match="end_job"):
            _stage_plan(client, plan)
        client._reset_staged.assert_called_once_with()

    def test_comment_expands_per_item(self):
        """Multi-item comment calls mirror one transcript line per item."""
        client = Mock(spec=RpaRpcClient)
        plan = [
            ("comment", (["# Ops Actions", "# Ops Section Start"],)),
            ("end_job", ()),
        ]
        _stage_plan(client, plan)
        deltas = client.root.exposed_stage_gluescript_delta
        assert deltas.call_count == 1
        lines = deltas.call_args_list[0].args[1]
        assert lines == [
            "comment(['# Ops Actions'])",
            "comment(['# Ops Section Start'])",
            "end_job()",
        ]

    def test_migrated_settings_flush_via_generic_delta(self):
        """Migrated per-op settings replay as generic transcript lines."""
        client = Mock(spec=RpaRpcClient)
        plan = [
            ("declare_job", ("Job", "MACHINE", None, 1, 1, 0.0, 0.0)),
            ("cut_speed", (250.0,)),
            ("frequency", (25.0,)),
            ("pwm", (50.0,)),
            ("delay", ("250.000ms",)),
            ("end_job", ()),
        ]
        _stage_plan(client, plan)
        deltas = client.root.exposed_stage_gluescript_delta
        assert deltas.call_count == 1
        assert deltas.call_args_list[0].args[1] == [
            "declare_job('Job', 'MACHINE', None, 1, 1, 0.0, 0.0)",
            "cut_speed(250.0)",
            "frequency(25.0)",
            "pwm(50.0)",
            "delay('250.000ms')",
            "end_job()",
        ]
        client.root.exposed_add_layer_action.assert_not_called()

    def test_add_layer_action_still_forwards_between_migrated_calls(self):
        """Raw add_layer_action between migrated calls keeps forwarding."""
        client = Mock(spec=RpaRpcClient)
        plan = [
            ("declare_job", ("Job", "MACHINE", None, 1, 1, 0.0, 0.0)),
            ("cut_speed", (250.0,)),
            ("add_layer_action", (1, ["LASER_DEVICE_1"])),
            ("frequency", (25.0,)),
            ("end_job", ()),
        ]
        _stage_plan(client, plan)
        deltas = client.root.exposed_stage_gluescript_delta
        assert deltas.call_count == 2
        client.root.exposed_add_layer_action.assert_called_once_with(
            1, ["LASER_DEVICE_1"]
        )
        assert deltas.call_args_list[0].args[1] == [
            "declare_job('Job', 'MACHINE', None, 1, 1, 0.0, 0.0)",
            "cut_speed(250.0)",
        ]
        assert deltas.call_args_list[-1].args[1] == [
            "frequency(25.0)",
            "end_job()",
        ]

    def test_select_laser_replays_via_generic_delta(self):
        """select_laser is recorded and replays as a generic transcript line.

        The encoder records ``select_laser`` into the rpa_plan (it is not
        in the ``_UNRECORDED`` set), so ``_stage_plan`` replays it as a
        generic GlueScript transcript line rather than forwarding it via
        the ``add_layer_action`` special case.
        """
        client = Mock(spec=RpaRpcClient)
        plan = [
            ("declare_job", ("Job", "MACHINE", None, 1, 1, 0.0, 0.0)),
            ("select_laser", (2,)),
            ("end_job", ()),
        ]
        _stage_plan(client, plan)
        deltas = client.root.exposed_stage_gluescript_delta
        assert deltas.call_count == 1
        assert deltas.call_args_list[0].args[1] == [
            "declare_job('Job', 'MACHINE', None, 1, 1, 0.0, 0.0)",
            "select_laser(2)",
            "end_job()",
        ]
        client.root.exposed_add_layer_action.assert_not_called()


class TestWcsHandling:
    """WCS selection and offset reads behave per the framework contract."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair",
        [DIRECT_MODE, RPC_MODE],
        ids=["direct", "rpc"],
        indirect=True,
    )
    async def test_set_wcs_offset_raises_not_implemented(self, adapter_pair):
        """set_wcs_offset must raise NotImplementedError (fail loud)."""
        adapter, _backend = adapter_pair
        with pytest.raises(NotImplementedError):
            await adapter.set_wcs_offset("SET_POINT", 1.0, 2.0, 3.0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair",
        [DIRECT_MODE, RPC_MODE],
        ids=["direct", "rpc"],
        indirect=True,
    )
    async def test_read_wcs_offsets_returns_four_slots(self, adapter_pair):
        """read_wcs_offsets() must return four zeroed slots and fire the
        signal."""
        adapter, _backend = adapter_pair
        received = []

        def _record_offsets(sender, offsets):
            received.append(offsets)

        adapter.wcs_updated.connect(_record_offsets)
        offsets = await adapter.read_wcs_offsets()
        assert offsets == {
            "MACHINE": (0.0, 0.0, 0.0),
            "ANCHOR": (0.0, 0.0, 0.0),
            "CURRENT": (0.0, 0.0, 0.0),
            "SET_POINT": (0.0, 0.0, 0.0),
        }
        assert received == [offsets]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair",
        [DIRECT_MODE, RPC_MODE],
        ids=["direct", "rpc"],
        indirect=True,
    )
    async def test_read_parser_state_returns_selected_wcs(self, adapter_pair):
        """read_parser_state() mirrors the selected WCS, default MACHINE."""
        adapter, _backend = adapter_pair
        assert await adapter.read_parser_state() == "MACHINE"
        await adapter.select_wcs("ANCHOR")
        assert await adapter.read_parser_state() == "ANCHOR"
        await adapter.select_wcs("SET_POINT")
        assert await adapter.read_parser_state() == "SET_POINT"


class TestLiveBridgeRpc:
    """RPC live bridge uses client jog/home without double-running."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [RPC_MODE], ids=["rpc"], indirect=True
    )
    async def test_home_uses_client_home_without_run(self, adapter_pair):
        """home() must call client.home and never run() its result."""
        adapter, client = adapter_pair
        await adapter.home()
        client.home.assert_called_once()
        client.run.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [RPC_MODE], ids=["rpc"], indirect=True
    )
    async def test_home_z_uses_client_home_z(self, adapter_pair):
        """home(Axis.Z) must call client.home_z only."""
        adapter, client = adapter_pair
        await adapter.home(Axis.Z)
        client.home_z.assert_called_once()
        client.home.assert_not_called()
        client.run.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [RPC_MODE], ids=["rpc"], indirect=True
    )
    async def test_home_xy_uses_client_home(self, adapter_pair):
        """home(Axis.X | Axis.Y) must call client.home."""
        adapter, client = adapter_pair
        await adapter.home(Axis.X | Axis.Y)
        client.home.assert_called_once()
        client.run.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [RPC_MODE], ids=["rpc"], indirect=True
    )
    async def test_move_to_before_any_jog_uses_default_speed(
        self, adapter_pair
    ):
        """move_to() before any jog uses the default 600 mm/s speed."""
        adapter, client = adapter_pair
        await adapter.move_to(10.0, 20.0)
        client.jog_set_xy_speed.assert_called_once_with(600.0)
        client.jog_xy_to.assert_called_once_with(10.0, 20.0)
        client.run.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [RPC_MODE], ids=["rpc"], indirect=True
    )
    async def test_jog_xy_uses_client_jog_xy_rel_without_run(
        self, adapter_pair
    ):
        """jog(x,y) must set speed then call client.jog_xy_rel."""
        adapter, client = adapter_pair
        await adapter.jog(speed=600, x=5.0, y=5.0)
        client.jog_set_xy_speed.assert_called_once_with(10.0)
        client.jog_xy_rel.assert_called_once_with(5.0, 5.0)
        client.run.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [RPC_MODE], ids=["rpc"], indirect=True
    )
    async def test_jog_z_sets_speed_and_uses_client_jog_z_rel(
        self, adapter_pair
    ):
        """jog(z) must set the xy and z speeds and call client.jog_z_rel."""
        adapter, client = adapter_pair
        await adapter.jog(speed=600, z=3.0)
        client.jog_set_xy_speed.assert_called_once_with(10.0)
        client.jog_set_z_speed.assert_called_once_with(10.0)
        client.jog_z_rel.assert_called_once_with(3.0)
        client.run.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [RPC_MODE], ids=["rpc"], indirect=True
    )
    async def test_jog_u_sets_speed_and_uses_client_jog_u_rel(
        self, adapter_pair
    ):
        """jog(u) must set the xy and u speeds and call client.jog_u_rel."""
        adapter, client = adapter_pair
        await adapter.jog(speed=600, u=4.0)
        client.jog_set_xy_speed.assert_called_once_with(10.0)
        client.jog_set_u_speed.assert_called_once_with(10.0)
        client.jog_u_rel.assert_called_once_with(4.0)
        client.run.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [RPC_MODE], ids=["rpc"], indirect=True
    )
    async def test_jog_reasserts_xy_speed_every_jog(self, adapter_pair):
        """jog_set_xy_speed must re-run on every jog, not just the first."""
        adapter, client = adapter_pair
        await adapter.jog(speed=600, x=1.0)
        await adapter.jog(speed=600, y=2.0)
        client.jog_set_xy_speed.assert_has_calls([call(10.0), call(10.0)])
        client.jog_x_rel.assert_called_once_with(1.0)
        client.jog_y_rel.assert_called_once_with(2.0)
        client.run.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [RPC_MODE], ids=["rpc"], indirect=True
    )
    async def test_move_to_uses_last_jog_speed(self, adapter_pair):
        """move_to() reuses the last jog() speed instead of a hardcoded one."""
        adapter, client = adapter_pair
        await adapter.jog(speed=600, x=1.0)
        await adapter.move_to(0.0, 0.0)
        await adapter.jog(speed=600, x=2.0)
        client.jog_set_xy_speed.assert_has_calls(
            [call(10.0), call(10.0), call(10.0)]
        )
        client.jog_xy_to.assert_called_once_with(0.0, 0.0)
        client.jog_x_rel.assert_has_calls([call(1.0), call(2.0)])
        client.run.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [RPC_MODE], ids=["rpc"], indirect=True
    )
    async def test_speed_only_jog_records_speed_without_backend(
        self, adapter_pair
    ):
        """A delta-less jog must no-op on the backend but record speed."""
        adapter, client = adapter_pair
        await adapter.jog(speed=900)
        client.jog_set_xy_speed.assert_not_called()
        client.jog_xy_rel.assert_not_called()
        client.jog_z_rel.assert_not_called()
        client.jog_u_rel.assert_not_called()
        await adapter.move_to(0.0, 0.0)
        client.jog_set_xy_speed.assert_called_once_with(15.0)
        client.jog_xy_to.assert_called_once_with(0.0, 0.0)


class TestLiveBridgeDirect:
    """Direct live bridge delegates jog/home to the backend wrapper."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [DIRECT_MODE], ids=["direct"], indirect=True
    )
    async def test_home_calls_backend_home_without_run(self, adapter_pair):
        """home() must call backend.home, never backend.run."""
        adapter, backend = adapter_pair
        await adapter.home()
        backend.home.assert_called_once()
        backend.run.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [DIRECT_MODE], ids=["direct"], indirect=True
    )
    async def test_home_z_calls_backend_home_z(self, adapter_pair):
        """home(Axis.Z) must call backend.home_z only."""
        adapter, backend = adapter_pair
        await adapter.home(Axis.Z)
        backend.home_z.assert_called_once()
        backend.home.assert_not_called()
        backend.run.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [DIRECT_MODE], ids=["direct"], indirect=True
    )
    async def test_move_to_before_any_jog_uses_default_speed(
        self, adapter_pair
    ):
        """move_to() before any jog uses the default 600 mm/s speed."""
        adapter, backend = adapter_pair
        await adapter.move_to(10.0, 20.0)
        backend.jog_set_xy_speed.assert_called_once_with(600.0)
        backend.jog_xy_to.assert_called_once_with(10.0, 20.0)
        backend.run.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [DIRECT_MODE], ids=["direct"], indirect=True
    )
    async def test_move_to_uses_last_jog_speed(self, adapter_pair):
        """move_to() reuses the last jog() speed instead of a hardcoded one."""
        adapter, backend = adapter_pair
        await adapter.jog(speed=600, x=1.0)
        await adapter.move_to(0.0, 0.0)
        await adapter.jog(speed=600, x=2.0)
        backend.jog_set_xy_speed.assert_has_calls(
            [call(10.0), call(10.0), call(10.0)]
        )
        backend.jog_xy_to.assert_called_once_with(0.0, 0.0)
        backend.jog_x_rel.assert_has_calls([call(1.0), call(2.0)])
        backend.run.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [DIRECT_MODE], ids=["direct"], indirect=True
    )
    async def test_jog_xy_sets_speed_and_uses_backend_jog_xy_rel(
        self, adapter_pair
    ):
        """jog(x,y) must set speed then call backend.jog_xy_rel."""
        adapter, backend = adapter_pair
        await adapter.jog(speed=600, x=5.0, y=5.0)
        backend.jog_set_xy_speed.assert_called_once_with(10.0)
        backend.jog_xy_rel.assert_called_once_with(5.0, 5.0)
        backend.run.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [DIRECT_MODE], ids=["direct"], indirect=True
    )
    async def test_jog_z_sets_xy_speed_then_z_speed_and_rel(
        self, adapter_pair
    ):
        """jog(z) must re-assert xy speed, then set z speed and jog_z_rel."""
        adapter, backend = adapter_pair
        await adapter.jog(speed=600, z=2.0)
        backend.jog_set_xy_speed.assert_called_once_with(10.0)
        backend.jog_set_z_speed.assert_called_once_with(10.0)
        backend.jog_z_rel.assert_called_once_with(2.0)
        backend.run.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [DIRECT_MODE], ids=["direct"], indirect=True
    )
    async def test_jog_x_sets_xy_speed_and_uses_backend_jog_x_rel(
        self, adapter_pair
    ):
        """jog(x) without y must set speed then call backend.jog_x_rel."""
        adapter, backend = adapter_pair
        await adapter.jog(speed=600, x=5.0)
        backend.jog_set_xy_speed.assert_called_once_with(10.0)
        backend.jog_x_rel.assert_called_once_with(5.0)
        backend.run.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [DIRECT_MODE], ids=["direct"], indirect=True
    )
    async def test_jog_y_sets_xy_speed_and_uses_backend_jog_y_rel(
        self, adapter_pair
    ):
        """jog(y) without x must set speed and call backend.jog_y_rel only."""
        adapter, backend = adapter_pair
        await adapter.jog(speed=600, y=5.0)
        backend.jog_set_xy_speed.assert_called_once_with(10.0)
        backend.jog_y_rel.assert_called_once_with(5.0)
        backend.jog_x_rel.assert_not_called()
        backend.run.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [DIRECT_MODE], ids=["direct"], indirect=True
    )
    async def test_jog_u_sets_speed_and_uses_backend_jog_u_rel(
        self, adapter_pair
    ):
        """jog(u) must set the xy and u speeds and call backend.jog_u_rel."""
        adapter, backend = adapter_pair
        await adapter.jog(speed=600, u=2.0)
        backend.jog_set_xy_speed.assert_called_once_with(10.0)
        backend.jog_set_u_speed.assert_called_once_with(10.0)
        backend.jog_u_rel.assert_called_once_with(2.0)
        backend.run.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [DIRECT_MODE], ids=["direct"], indirect=True
    )
    async def test_speed_only_jog_records_speed_without_backend(
        self, adapter_pair
    ):
        """A delta-less jog must no-op on the backend but record speed."""
        adapter, backend = adapter_pair
        await adapter.jog(speed=900)
        backend.jog_set_xy_speed.assert_not_called()
        backend.jog_xy_rel.assert_not_called()
        backend.jog_z_rel.assert_not_called()
        backend.jog_u_rel.assert_not_called()
        await adapter.move_to(0.0, 0.0)
        backend.jog_set_xy_speed.assert_called_once_with(15.0)
        backend.jog_xy_to.assert_called_once_with(0.0, 0.0)


class TestFailLoud:
    """Backend jog/home failures must propagate through the adapter."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair",
        [DIRECT_MODE, RPC_MODE],
        ids=["direct", "rpc"],
        indirect=True,
    )
    async def test_jog_propagates_backend_failure(self, adapter_pair):
        """A raising jog_set_xy_speed must surface as RuntimeError."""
        adapter, backend = adapter_pair
        backend.jog_set_xy_speed.side_effect = RuntimeError("jog failed")
        with pytest.raises(RuntimeError, match="jog failed"):
            await adapter.jog(speed=600, x=1.0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair",
        [DIRECT_MODE, RPC_MODE],
        ids=["direct", "rpc"],
        indirect=True,
    )
    async def test_home_propagates_backend_failure(self, adapter_pair):
        """A raising home must surface as RuntimeError."""
        adapter, backend = adapter_pair
        backend.home.side_effect = RuntimeError("home failed")
        with pytest.raises(RuntimeError, match="home failed"):
            await adapter.home()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair",
        [DIRECT_MODE, RPC_MODE],
        ids=["direct", "rpc"],
        indirect=True,
    )
    async def test_move_to_propagates_backend_failure(self, adapter_pair):
        """A raising jog_xy_to must surface as RuntimeError."""
        adapter, backend = adapter_pair
        backend.jog_xy_to.side_effect = RuntimeError("move failed")
        with pytest.raises(RuntimeError, match="move failed"):
            await adapter.move_to(0.0, 0.0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair",
        [DIRECT_MODE, RPC_MODE],
        ids=["direct", "rpc"],
        indirect=True,
    )
    async def test_jog_z_propagates_backend_failure(self, adapter_pair):
        """A raising jog_z_rel must surface as RuntimeError."""
        adapter, backend = adapter_pair
        backend.jog_z_rel.side_effect = RuntimeError("z jog failed")
        with pytest.raises(RuntimeError, match="z jog failed"):
            await adapter.jog(speed=600, z=1.0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair",
        [DIRECT_MODE, RPC_MODE],
        ids=["direct", "rpc"],
        indirect=True,
    )
    async def test_jog_u_propagates_backend_failure(self, adapter_pair):
        """A raising jog_u_rel must surface as RuntimeError."""
        adapter, backend = adapter_pair
        backend.jog_u_rel.side_effect = RuntimeError("u jog failed")
        with pytest.raises(RuntimeError, match="u jog failed"):
            await adapter.jog(speed=600, u=1.0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair",
        [DIRECT_MODE, RPC_MODE],
        ids=["direct", "rpc"],
        indirect=True,
    )
    async def test_jog_x_propagates_backend_failure(self, adapter_pair):
        """A raising jog_x_rel must surface as RuntimeError."""
        adapter, backend = adapter_pair
        backend.jog_x_rel.side_effect = RuntimeError("x jog failed")
        with pytest.raises(RuntimeError, match="x jog failed"):
            await adapter.jog(speed=600, x=1.0)


class TestStatusMmFix:
    """Status positions arrive in mm and must not be divided by 1000."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair",
        [DIRECT_MODE, RPC_MODE],
        ids=["direct", "rpc"],
        indirect=True,
    )
    async def test_position_tuple_not_divided_by_1000(self, adapter_pair):
        """A float_mm tuple position must pass through unchanged."""
        adapter, _backend = adapter_pair
        adapter._on_rpa_status(
            {
                "POSITION_X": (123.456, "X"),
                "POSITION_Y": (45.678, "Y"),
                "POSITION_Z": (7.89, "Z"),
            }
        )
        assert adapter.state.machine_pos == (123.456, 45.678, 7.89)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair",
        [DIRECT_MODE, RPC_MODE],
        ids=["direct", "rpc"],
        indirect=True,
    )
    async def test_plain_float_position_passes_through(self, adapter_pair):
        """A bare float position must pass through unchanged."""
        adapter, _backend = adapter_pair
        adapter._on_rpa_status(
            {
                "POSITION_X": 12.5,
                "POSITION_Y": 34.5,
                "POSITION_Z": 56.5,
            }
        )
        assert adapter.state.machine_pos == (12.5, 34.5, 56.5)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair",
        [DIRECT_MODE, RPC_MODE],
        ids=["direct", "rpc"],
        indirect=True,
    )
    async def test_partial_position_keeps_other_axes(self, adapter_pair):
        """A missing axis must retain the previously reported value."""
        adapter, _backend = adapter_pair
        adapter.state = replace(adapter.state, machine_pos=(10.0, 20.0, 30.0))
        adapter._on_rpa_status({"POSITION_X": (1.0, "X")})
        assert adapter.state.machine_pos == (1.0, 20.0, 30.0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [RPC_MODE], ids=["rpc"], indirect=True
    )
    async def test_machine_status_dict_accepted(self, adapter_pair):
        """A machine-status-shaped dict must not raise or move the state."""
        adapter, _backend = adapter_pair
        adapter._on_rpa_status({"MACHINE_STATUS": 1})
        assert adapter.state.machine_pos == (None, None, None)

    def test_unwrap_mm_tuple_returns_first_element(self):
        """_unwrap_mm must return the float first element of a tuple."""
        assert _unwrap_mm((12.5, "description")) == 12.5

    def test_unwrap_mm_plain_value_passes_through(self):
        """_unwrap_mm must pass bare floats through unchanged."""
        assert _unwrap_mm(12.5) == 12.5


class TestReconnectListenerHygiene:
    """Reconnect must unregister-before-register to avoid double-fires."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [DIRECT_MODE], ids=["direct"], indirect=True
    )
    async def test_reconnect_does_not_double_register(self, adapter_pair):
        """Two connect cycles must keep one listener per event type."""
        adapter, backend = adapter_pair
        registries = {"status": [], "error": [], "reply": []}

        def _register(kind):
            def register(callback):
                registries[kind].append(callback)

            return register

        def _unregister(kind):
            def unregister(callback):
                if callback in registries[kind]:
                    registries[kind].remove(callback)

            return unregister

        for kind in registries:
            getattr(
                backend, f"register_{kind}_listener"
            ).side_effect = _register(kind)
            getattr(
                backend, f"unregister_{kind}_listener"
            ).side_effect = _unregister(kind)

        await _run_connect_cycle(
            adapter, lambda: len(registries["status"]) == 1
        )
        assert len(registries["status"]) == 1
        assert len(registries["error"]) == 1
        assert len(registries["reply"]) == 1

        await _run_connect_cycle(
            adapter,
            lambda: backend.register_status_listener.call_count == 2,
        )
        assert len(registries["status"]) == 1
        assert len(registries["error"]) == 1
        assert len(registries["reply"]) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [DIRECT_MODE], ids=["direct"], indirect=True
    )
    async def test_reconnect_unregisters_stored_refs_before_registering(
        self, adapter_pair
    ):
        """Each register must be preceded by an unregister of the same ref."""
        adapter, backend = adapter_pair
        await _run_connect_cycle(
            adapter, lambda: backend.register_status_listener.called
        )
        await _run_connect_cycle(
            adapter,
            lambda: backend.register_status_listener.call_count == 2,
        )

        for entry in backend.unregister_status_listener.call_args_list:
            assert entry.args[0] == adapter._on_rpa_status
        for entry in backend.unregister_error_listener.call_args_list:
            assert entry.args[0] == adapter._on_rpa_error
        for entry in backend.unregister_reply_listener.call_args_list:
            assert entry.args[0] == adapter._on_rpa_reply

        unreg_indices = [
            index
            for index, entry in enumerate(backend.method_calls)
            if entry[0] == "unregister_status_listener"
        ]
        reg_indices = [
            index
            for index, entry in enumerate(backend.method_calls)
            if entry[0] == "register_status_listener"
        ]
        assert len(unreg_indices) == len(reg_indices) == 2
        for unreg_index, reg_index in zip(unreg_indices, reg_indices):
            assert unreg_index < reg_index


class TestStringStatusEvents:
    """String status events drive adapter connection state and signals."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair",
        [DIRECT_MODE, RPC_MODE],
        ids=["direct", "rpc"],
        indirect=True,
    )
    async def test_status_connected_transitions_state(self, adapter_pair):
        """'CONNECTED' marks the adapter connected and IDLE."""
        adapter, _backend = adapter_pair
        connection_mock = Mock()
        state_mock = Mock()
        adapter.connection_status_changed.send = connection_mock
        adapter.state_changed.send = state_mock
        adapter._on_rpa_status("CONNECTED")
        assert adapter._is_connected is True
        assert adapter.state.status == DeviceStatus.IDLE
        connection_mock.assert_called_once()
        assert connection_mock.call_args.kwargs["status"] == (
            TransportStatus.CONNECTED
        )
        state_mock.assert_called_once()
        assert state_mock.call_args.kwargs["state"].status == (
            DeviceStatus.IDLE
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair",
        [DIRECT_MODE, RPC_MODE],
        ids=["direct", "rpc"],
        indirect=True,
    )
    async def test_status_disconnected_transitions_state(self, adapter_pair):
        """'DISCONNECTED' marks the adapter disconnected and UNKNOWN."""
        adapter, _backend = adapter_pair
        connection_mock = Mock()
        state_mock = Mock()
        adapter.connection_status_changed.send = connection_mock
        adapter.state_changed.send = state_mock
        adapter._on_rpa_status("DISCONNECTED")
        assert adapter._is_connected is False
        assert adapter.state.status == DeviceStatus.UNKNOWN
        connection_mock.assert_called_once()
        assert connection_mock.call_args.kwargs["status"] == (
            TransportStatus.DISCONNECTED
        )
        state_mock.assert_called_once()
        assert state_mock.call_args.kwargs["state"].status == (
            DeviceStatus.UNKNOWN
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair",
        [DIRECT_MODE, RPC_MODE],
        ids=["direct", "rpc"],
        indirect=True,
    )
    async def test_status_terminated_transitions_state(self, adapter_pair):
        """'TERMINATED' marks the adapter disconnected and UNKNOWN."""
        adapter, _backend = adapter_pair
        connection_mock = Mock()
        state_mock = Mock()
        adapter.connection_status_changed.send = connection_mock
        adapter.state_changed.send = state_mock
        adapter._on_rpa_status("TERMINATED")
        assert adapter._is_connected is False
        assert adapter.state.status == DeviceStatus.UNKNOWN
        connection_mock.assert_called_once()
        assert connection_mock.call_args.kwargs["status"] == (
            TransportStatus.DISCONNECTED
        )
        state_mock.assert_called_once()
        assert state_mock.call_args.kwargs["state"].status == (
            DeviceStatus.UNKNOWN
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair",
        [DIRECT_MODE, RPC_MODE],
        ids=["direct", "rpc"],
        indirect=True,
    )
    async def test_status_late_event_inert_after_shutdown(self, adapter_pair):
        """After shutdown, late status events must be inert."""
        adapter, _backend = adapter_pair
        connection_mock = Mock()
        state_mock = Mock()
        adapter.connection_status_changed.send = connection_mock
        adapter.state_changed.send = state_mock
        adapter._shutting_down = True
        adapter._on_rpa_status("CONNECTED")
        assert adapter._is_connected is False
        assert adapter.state.status == DeviceStatus.UNKNOWN
        connection_mock.assert_not_called()
        state_mock.assert_not_called()


class TestConnectFailureBackoff:
    """A failed start drives the stop-cleanup path; the adapter survives."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair",
        [DIRECT_MODE, RPC_MODE],
        ids=["direct", "rpc"],
        indirect=True,
    )
    async def test_start_false_runs_stop_cleanup_and_survives(
        self, adapter_pair
    ):
        """start()==False must run _stop_backend cleanup, not raise out."""
        adapter, backend = adapter_pair
        backend.start.return_value = False
        await _run_connect_cycle(adapter, lambda: backend.stop.called)
        assert backend.stop.called
        assert adapter._is_connected is False
        assert adapter._backend is backend


class TestConnectClearsServerHeadTail:
    """Connect-time head/tail neutralization is TUI-mode only."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [RPC_MODE], ids=["rpc"], indirect=True
    )
    async def test_rpc_connect_clears_server_head_tail(self, adapter_pair):
        """TUI connect must clear the server driver's head/tail scripts."""
        adapter, client = adapter_pair
        await _run_connect_cycle(
            adapter, lambda: client.set_head_script.called
        )
        client.set_head_script.assert_any_call([])
        client.set_tail_script.assert_any_call([])

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [DIRECT_MODE], ids=["direct"], indirect=True
    )
    async def test_direct_connect_never_clears_head_tail(self, adapter_pair):
        """Direct mode has no head/tail surface and must not clear it."""
        adapter, backend = adapter_pair
        await _run_connect_cycle(adapter, lambda: adapter._is_connected)
        assert not hasattr(backend, "set_head_script")
        assert not hasattr(backend, "set_tail_script")


class TestHealthPoll:
    """Poll health per mode: RPC probes is_alive, direct reads is_connected."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [RPC_MODE], ids=["rpc"], indirect=True
    )
    async def test_rpc_controller_down_keeps_rpc_alive(
        self, adapter_pair, monkeypatch
    ):
        """A controller going quiet must not tear down the RPC transport."""
        adapter, backend = adapter_pair
        backend.is_connected = False
        backend.is_alive.return_value = True
        monkeypatch.setattr(RuidaRPAAdapter, "CONNECTION_POLL_INTERVAL", 0.01)

        adapter._keep_running = True
        await adapter._connect_implementation()
        try:
            await _wait_until(lambda: adapter._is_connected)
            adapter._on_rpa_status("DISCONNECTED")
            await _wait_until(lambda: not adapter._is_connected)
            await asyncio.sleep(0.05)
            assert backend.connect.call_count == 1
            adapter._on_rpa_status("CONNECTED")
            await _wait_until(lambda: adapter._is_connected)
            assert backend.connect.call_count == 1
        finally:
            adapter._keep_running = False
            task = adapter._connection_task
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [RPC_MODE], ids=["rpc"], indirect=True
    )
    async def test_rpc_transport_dead_reconnects(
        self, adapter_pair, monkeypatch
    ):
        """A failed is_alive probe must tear down and reconnect RPC."""
        adapter, backend = adapter_pair
        backend.is_alive.return_value = False
        monkeypatch.setattr(RuidaRPAAdapter, "CONNECTION_POLL_INTERVAL", 0.01)
        monkeypatch.setattr(RuidaRPAAdapter, "RECONNECT_BASE_DELAY", 0.01)

        adapter._keep_running = True
        await adapter._connect_implementation()
        try:
            await _wait_until(lambda: backend.connect.call_count >= 2)
            await _wait_until(lambda: not adapter._is_connected)
            backend.disconnect.assert_not_called()
            backend.stop.assert_not_called()
        finally:
            adapter._keep_running = False
            task = adapter._connection_task
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [DIRECT_MODE], ids=["direct"], indirect=True
    )
    async def test_direct_controller_down_reconnects(
        self, adapter_pair, monkeypatch
    ):
        """A direct-mode controller going down must keep reconnecting."""
        adapter, backend = adapter_pair
        backend.is_connected = False
        monkeypatch.setattr(RuidaRPAAdapter, "CONNECTION_POLL_INTERVAL", 0.01)
        monkeypatch.setattr(RuidaRPAAdapter, "RECONNECT_BASE_DELAY", 0.01)

        adapter._keep_running = True
        await adapter._connect_implementation()
        try:
            await _wait_until(lambda: backend.start.call_count >= 2)
            assert backend.start.call_count >= 2
            assert adapter._backend is backend
        finally:
            adapter._keep_running = False
            task = adapter._connection_task
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task


class TestRpcTimeoutSetup:
    """The setup 'timeout' var drives the RPyC sync request timeout."""

    def test_timeout_var_present_with_driver_defaults(
        self, isolated_context, isolated_machine
    ):
        """get_setup_vars must expose a timeout var with driver defaults."""
        adapter = RuidaRPAAdapter(isolated_context, isolated_machine)
        varset = adapter.get_setup_vars()
        timeout_var = varset.get("timeout")
        assert timeout_var is not None
        assert isinstance(timeout_var, FloatVar)
        assert timeout_var.default == DEFAULT_RPC_TIMEOUT_S
        assert timeout_var.min_val == 1.0
        assert timeout_var.digits == 1
        assert timeout_var.visible_when is not None
        assert timeout_var.visible_when({"tui": True}) is True
        assert timeout_var.visible_when({"tui": False}) is False

    @pytest.mark.asyncio
    async def test_setup_passes_timeout_to_rpc_client(
        self, isolated_context, isolated_machine, monkeypatch
    ):
        """setup(tui=True, timeout=42.0) must construct the client with it."""
        adapter = RuidaRPAAdapter(isolated_context, isolated_machine)
        mock_client_cls = Mock()
        monkeypatch.setattr(rpa_adapter, "RpaRpcClient", mock_client_cls)

        adapter.setup(tui=True, timeout=42.0)

        mock_client_cls.assert_called_once_with(timeout=42.0)

        await adapter.cleanup()
        await isolated_machine.shutdown()

    @pytest.mark.asyncio
    async def test_setup_omits_timeout_uses_driver_default(
        self, isolated_context, isolated_machine, monkeypatch
    ):
        """setup(tui=True) without timeout must use DEFAULT_RPC_TIMEOUT_S."""
        adapter = RuidaRPAAdapter(isolated_context, isolated_machine)
        mock_client_cls = Mock()
        monkeypatch.setattr(rpa_adapter, "RpaRpcClient", mock_client_cls)

        adapter.setup(tui=True)

        mock_client_cls.assert_called_once_with(timeout=DEFAULT_RPC_TIMEOUT_S)

        await adapter.cleanup()
        await isolated_machine.shutdown()

    @pytest.mark.parametrize(
        "value, match",
        [
            (0, "positive"),
            (float("nan"), "positive"),
            (float("inf"), "positive"),
            ("abc", "number"),
            (None, "number"),
        ],
    )
    def test_invalid_timeout_raises_driver_setup_error(
        self, isolated_context, isolated_machine, value, match
    ):
        """Non-numeric or non-positive timeouts must fail loudly."""
        adapter = RuidaRPAAdapter(isolated_context, isolated_machine)
        with pytest.raises(DriverSetupError, match=match):
            adapter._setup_implementation(tui=True, timeout=value)


class TestSeedMachineSpeedDefaults:
    """The adapter seeds Ruida speed defaults only while unconfigured."""

    @pytest.mark.parametrize(
        "tui", [DIRECT_MODE, RPC_MODE], ids=["direct", "rpc"]
    )
    def test_seeds_defaults_at_framework_defaults(self, isolated_context, tui):
        """A machine at framework defaults gets the Ruida speed limits."""
        from rayforge.machine.models.machine import Machine

        m = Machine(isolated_context)
        a = RuidaRPAAdapter(isolated_context, m)
        a._setup_implementation(tui=tui)

        assert m.max_cut_speed == DEFAULT_MAX_CUT_SPEED_MMPM
        assert m.max_travel_speed == DEFAULT_MAX_TRAVEL_SPEED_MMPM

    def test_does_not_overwrite_user_values(self, isolated_context):
        """User-configured speeds must never be clobbered by seeding."""
        from rayforge.machine.models.machine import Machine

        m = Machine(isolated_context)
        m.set_max_cut_speed(20000)
        m.set_max_travel_speed(50000)
        a = RuidaRPAAdapter(isolated_context, m)
        a._setup_implementation(tui=False)

        assert m.max_cut_speed == 20000
        assert m.max_travel_speed == 50000
