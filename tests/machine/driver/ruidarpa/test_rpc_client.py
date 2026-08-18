"""
Unit tests for the real RpaRpcClient health probe and timeout config.

The is_alive() probe reports RPC transport health ONLY: a controller
that goes quiet (is_connected() returns False remotely) must not count
as transport death, while a failed round trip must.
"""

from unittest.mock import Mock

import pytest

from rayforge.machine.driver.ruidarpa import rpa_rpc_client
from rayforge.machine.driver.ruidarpa.rpa_rpc_client import (
    SYNC_REQUEST_TIMEOUT,
    RpaRpcClient,
)


class TestIsAlive:
    """The is_alive() probe reports RPC transport health only."""

    def test_is_alive_without_connection_returns_false(self):
        """A client that never connected is not alive."""
        client = RpaRpcClient()
        assert client.is_alive() is False

    def test_is_alive_rpc_alive_when_controller_down(self):
        """Controller-down must not kill the RPC health probe."""
        client = RpaRpcClient()
        client._conn = Mock()
        client._conn.root.is_connected.return_value = False
        assert client.is_alive() is True

    def test_is_alive_transport_error_returns_false(self):
        """A failed round trip means the RPC transport is dead."""
        client = RpaRpcClient()
        client._conn = Mock()
        client._conn.root.is_connected.side_effect = RuntimeError(
            "transport dead"
        )
        assert client.is_alive() is False


class TestConnectConfig:
    """connect() must configure the RPyC sync request timeout."""

    def test_connect_passes_sync_request_timeout(self, monkeypatch):
        """connect() must pass the 5s timeout and custom-exception flags."""
        mock_rpyc = Mock()
        mock_bg_thread = Mock()
        monkeypatch.setattr(rpa_rpc_client, "rpyc", mock_rpyc)
        monkeypatch.setattr(rpa_rpc_client, "BgServingThread", mock_bg_thread)

        client = RpaRpcClient(host="127.0.0.1", port=18812)
        result = client.connect()

        assert result is True
        mock_rpyc.connect.assert_called_once_with(
            "127.0.0.1",
            18812,
            config={
                "sync_request_timeout": SYNC_REQUEST_TIMEOUT,
                "import_custom_exceptions": True,
                "instantiate_custom_exceptions": True,
            },
        )
        mock_bg_thread.assert_called_once()
        assert SYNC_REQUEST_TIMEOUT == 5.0


class TestRootProperty:
    """root exposes typed netref access with a connected guard."""

    def test_root_raises_when_not_connected(self):
        """root must fail fast before a connection exists."""
        client = RpaRpcClient()
        try:
            client.root
        except RuntimeError as exc:
            assert "not connected" in str(exc)
        else:
            raise AssertionError("root should raise when not connected")

    def test_root_returns_conn_root_when_connected(self):
        """root must hand back the raw service root once connected."""
        client = RpaRpcClient()
        client._conn = Mock()
        assert client.root is client._conn.root


class TestRunStagedJob:
    """run_staged_job routes to run_job with no job body (staged run)."""

    def test_run_staged_job_calls_exposed_run_job_with_none(self):
        """run_staged_job must send job=None so the staged script runs."""
        client = RpaRpcClient()
        client._conn = Mock()
        exposed = client._conn.root.exposed_run_job

        client.run_staged_job()

        exposed.assert_called_once_with(None, auto_checksum=False)

    def test_run_staged_job_passes_auto_checksum(self):
        """auto_checksum must reach the exposed run_job call."""
        client = RpaRpcClient()
        client._conn = Mock()

        client.run_staged_job(auto_checksum=True)

        client._conn.root.exposed_run_job.assert_called_once_with(
            None, auto_checksum=True
        )


class TestResetStaged:
    """_reset_staged must reset the server-side staged state."""

    def test_reset_staged_calls_exposed_new_gluescript(self):
        """The RPC new_gluescript is the server-side staged-state reset."""
        client = RpaRpcClient()
        client._conn = Mock()

        client._reset_staged()

        client._conn.root.exposed_new_gluescript.assert_called_once_with()


class TestSetHeadTailScript:
    """set_head_script/set_tail_script must reach the server-side methods."""

    def test_set_head_script_calls_exposed_set_head_script(self):
        """set_head_script must dispatch to the exposed server method."""
        client = RpaRpcClient()
        client._conn = Mock()
        exposed = client._conn.root.exposed_set_head_script

        client.set_head_script([])

        exposed.assert_called_once_with([])

    def test_set_tail_script_calls_exposed_set_tail_script(self):
        """set_tail_script must dispatch to the exposed server method."""
        client = RpaRpcClient()
        client._conn = Mock()
        exposed = client._conn.root.exposed_set_tail_script

        client.set_tail_script([])

        exposed.assert_called_once_with([])


class TestJobControl:
    """Pause/resume/stop_job/reset dispatch and discard returned lines."""

    def test_pause_calls_exposed_pause(self):
        """pause must dispatch once and return None (lines discarded)."""
        client = RpaRpcClient()
        client._conn = Mock()
        exposed = client._conn.root.exposed_pause

        result = client.pause()

        exposed.assert_called_once_with()
        assert result is None

    def test_resume_calls_exposed_resume(self):
        """resume must dispatch once and return None (lines discarded)."""
        client = RpaRpcClient()
        client._conn = Mock()
        exposed = client._conn.root.exposed_resume

        result = client.resume()

        exposed.assert_called_once_with()
        assert result is None

    def test_stop_job_calls_exposed_stop_job(self):
        """stop_job must dispatch once and return None (lines discarded)."""
        client = RpaRpcClient()
        client._conn = Mock()
        exposed = client._conn.root.exposed_stop_job

        result = client.stop_job()

        exposed.assert_called_once_with()
        assert result is None

    def test_reset_calls_exposed_reset(self):
        """reset must dispatch once and return None (lines discarded)."""
        client = RpaRpcClient()
        client._conn = Mock()
        exposed = client._conn.root.exposed_reset

        result = client.reset()

        exposed.assert_called_once_with()
        assert result is None

    @pytest.mark.parametrize("name", ["pause", "resume", "stop_job", "reset"])
    def test_raises_when_not_connected(self, name):
        """Job-control calls must fail loudly before a connection exists."""
        client = RpaRpcClient()

        with pytest.raises(RuntimeError, match="not connected"):
            getattr(client, name)()
