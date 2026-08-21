"""
Adapter tests for the RuidaRPAAdapter.

The adapter wraps either RpaDirectDriver (direct mode) or RpcRdDriver
(RPC/TUI mode) as ``_backend``. These tests mock the backend and verify
the adapter's public surface:

- Stop/cleanup regression: backend stop()/close() must actually run
- run routing: run() re-encodes ops into the backend GlueScript and
  runs via run_job()
- Jog speed tracking: move_to() reuses the last jog() speed (default 600
  before any jog) and a speed-only jog updates the stored speed
- Live bridge behavior per mode (no double-run in RPC mode)
- Fail-loud: backend jog/home failures propagate through the adapter
- set_wcs_offset failing loud while select_wcs still routes to run()
- mm fix: status positions are not divided by 1000
- Reconnect listener hygiene (unregister-before-register in direct mode)
- Connect-time head/tail clearing: both modes clear the wrapped
  GlueScript's head/tail scripts to []
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
from rpalib.rpyc_client import RpcRdDriver
from ruidadriver.rd_gluescript import GlueScript

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
    _unwrap_mm,
)
from rayforge.machine.driver.ruidarpa.rpa_direct_driver import (
    RpaDirectDriver,
)
from rayforge.machine.models.laser import Laser
from rayforge.machine.transport import TransportStatus
from rayforge.pipeline.encoder.base import EncodedOutput, MachineCodeOpMap

DIRECT_MODE = False
RPC_MODE = True


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
async def adapter_pair(
    isolated_context, isolated_machine, request, monkeypatch
):
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

    backend_cls = RpcRdDriver if tui_mode else RpaDirectDriver
    backend = Mock(spec=backend_cls)
    backend.start.return_value = True
    backend.is_connected = True
    backend.machine_status = {}
    if tui_mode:
        # RpcRdDriver self-connects in its constructor; patch the class
        # so the connection loop builds our mock instead of a real one.
        monkeypatch.setattr(rpa_adapter, "RpcRdDriver", lambda **kw: backend)
        # RpcRdDriver has no unregister surface; expose tracked mocks so
        # tests can assert the adapter never calls them.
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
    """Stop/close must actually reach the backend (core bug fix)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [RPC_MODE], ids=["rpc"], indirect=True
    )
    async def test_stop_rpc_calls_stop_and_close(self, adapter_pair):
        """RPC stop must call client.stop() and client.close()."""
        adapter, client = adapter_pair
        await adapter._stop_backend()
        client.stop.assert_called_once()
        client.close.assert_called_once()
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
    async def test_stop_rpc_stop_raises_close_still_called(self, adapter_pair):
        """A raising stop() must not skip close() (dead transport)."""
        adapter, client = adapter_pair
        client.stop.side_effect = RuntimeError("transport dead")
        await adapter._stop_backend()
        client.stop.assert_called_once()
        client.close.assert_called_once()
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
    async def test_cleanup_rpc_stops_and_closes(self, adapter_pair):
        """cleanup() must reach client.stop()/close()."""
        adapter, client = adapter_pair
        await adapter.cleanup()
        client.stop.assert_called_once()
        client.close.assert_called_once()
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
    """run() re-encodes ops into the backend GlueScript, then run_job()."""

    @staticmethod
    def _gluescript_backend():
        """A real GlueScript with a mock run_job, usable as a run() backend."""
        gs = GlueScript()
        gs.run_job = Mock()
        gs.new_gluescript = Mock(wraps=gs.new_gluescript)
        return gs

    @staticmethod
    def _make_adapter(isolated_context, machine, tui_mode, gs):
        """Build an adapter whose backend authors into the real GlueScript."""
        adapter = RuidaRPAAdapter(isolated_context, machine)
        adapter.setup(udp_host="127.0.0.1", tui=tui_mode)
        if tui_mode:
            adapter._backend = gs
        else:
            driver = RpaDirectDriver()
            driver._driver = gs
            adapter._backend = driver
        return adapter

    @staticmethod
    def _job_ops(doc):
        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=doc.layers[0].uid)
        ops.move_to(5.0, 5.0, 0.0)
        ops.line_to(10.0, 8.0, 0.0)
        ops.layer_end(layer_uid=doc.layers[0].uid)
        ops.job_end()
        return ops

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
        "tui_mode", [DIRECT_MODE, RPC_MODE], ids=["direct", "rpc"]
    )
    async def test_run_reencodes_and_runs_job(
        self, isolated_context, isolated_machine, tui_mode
    ):
        """run() must re-encode ops into the backend GlueScript and run it."""
        machine = isolated_machine
        gs = self._gluescript_backend()
        adapter = self._make_adapter(isolated_context, machine, tui_mode, gs)
        doc = Doc()
        ops = self._job_ops(doc)
        encoded = EncodedOutput(text="dummy", op_map=MachineCodeOpMap())

        await adapter.run(encoded, doc, ops)

        gs.run_job.assert_called_once_with()
        # The encoder authored into the backend GlueScript.
        assert any(line.startswith("declare_job(") for line in gs.gluescript)

        await adapter.cleanup()
        await machine.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tui_mode", [DIRECT_MODE, RPC_MODE], ids=["direct", "rpc"]
    )
    async def test_run_empty_ops_skips_run_job(
        self, isolated_context, isolated_machine, tui_mode
    ):
        """run() with empty ops must not run a stale prior job."""
        machine = isolated_machine
        gs = self._gluescript_backend()
        adapter = self._make_adapter(isolated_context, machine, tui_mode, gs)
        doc = Doc()
        encoded = EncodedOutput(text="", op_map=MachineCodeOpMap())

        await adapter.run(encoded, doc, Ops())

        gs.run_job.assert_not_called()

        await adapter.cleanup()
        await machine.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tui_mode", [DIRECT_MODE, RPC_MODE], ids=["direct", "rpc"]
    )
    async def test_run_failed_encode_calls_new_gluescript_then_raises(
        self, isolated_context, isolated_machine, tui_mode
    ):
        """A failed encode must tear down the backend then re-raise."""
        machine = isolated_machine
        gs = self._gluescript_backend()
        gs.stage_gluescript = Mock(side_effect=RuntimeError("stage failed"))
        adapter = self._make_adapter(isolated_context, machine, tui_mode, gs)
        doc = Doc()
        ops = self._job_ops(doc)
        encoded = EncodedOutput(text="dummy", op_map=MachineCodeOpMap())

        with pytest.raises(RuntimeError, match="stage"):
            await adapter.run(encoded, doc, ops)

        # new_gluescript is called by the encode itself (at start and via
        # declare_job) plus the teardown; the teardown must add at least one
        # call beyond the encode's own resets.
        assert gs.new_gluescript.call_count >= 2

        await adapter.cleanup()
        await machine.shutdown()

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
    async def test_direct_connect_clears_wrapped_head_tail(self, adapter_pair):
        """Direct connect must clear the wrapped driver's head/tail scripts."""
        adapter, backend = adapter_pair
        await _run_connect_cycle(adapter, lambda: adapter._is_connected)
        backend.gluescript.set_head_script.assert_any_call([])
        backend.gluescript.set_tail_script.assert_any_call([])


class TestHealthPoll:
    """Poll health per mode: both modes read backend ``is_connected``."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "adapter_pair", [RPC_MODE], ids=["rpc"], indirect=True
    )
    async def test_rpc_controller_down_reconnects(
        self, adapter_pair, monkeypatch
    ):
        """A controller going quiet must tear down and reconnect RPC."""
        adapter, backend = adapter_pair
        backend.is_connected = False
        monkeypatch.setattr(RuidaRPAAdapter, "CONNECTION_POLL_INTERVAL", 0.01)
        monkeypatch.setattr(RuidaRPAAdapter, "RECONNECT_BASE_DELAY", 0.01)

        adapter._keep_running = True
        await adapter._connect_implementation()
        try:
            await _wait_until(lambda: backend.start.call_count >= 2)
            await _wait_until(lambda: not adapter._is_connected)
            backend.close.assert_not_called()
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
    async def test_setup_tui_stores_timeout_without_constructing_backend(
        self, isolated_context, isolated_machine
    ):
        """setup(tui=True, timeout=42.0) must store the timeout and leave the
        backend unconstructed (RpcRdDriver self-connects per attempt)."""
        adapter = RuidaRPAAdapter(isolated_context, isolated_machine)

        adapter.setup(tui=True, timeout=42.0)

        assert adapter._rpc_timeout == 42.0
        assert adapter._backend is None

        await adapter.cleanup()
        await isolated_machine.shutdown()

    @pytest.mark.asyncio
    async def test_setup_tui_omits_timeout_uses_driver_default(
        self, isolated_context, isolated_machine
    ):
        """setup(tui=True) without timeout must use DEFAULT_RPC_TIMEOUT_S."""
        adapter = RuidaRPAAdapter(isolated_context, isolated_machine)

        adapter.setup(tui=True)

        assert adapter._rpc_timeout == DEFAULT_RPC_TIMEOUT_S
        assert adapter._backend is None

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
