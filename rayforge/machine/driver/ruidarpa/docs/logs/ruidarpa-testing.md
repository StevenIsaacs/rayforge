# Ruida RPA Testing Log

## 2026-06-19 — RuidaRPAEncoder + RpaDirectDriver integration

### Changes made

#### RuidaRPAEncoder (`rpa_encoder.py`)
- Full implementation (391 lines) covering all 23 CommandTypes
- State tracking with guard clauses to skip redundant config commands
- Linearization for arcs, beziers, scanlines via `ops.linearize()`
- Proper unit conversions: Hz→KHz (÷1000), µs→mS (÷1000), normalized power→% (×100)
- Job framing: `SET_ABSOLUTE` + `START_PROCESS` / `BLOCK_END` + `SET_FILE_SUM`
- Unknown command types raise ValueError (fail-fast)
- Exported through `ruidarpa/__init__.py`

#### Encoder integration into `RpaDirectDriver` (`rpa_direct_driver.py`)
- Added `create_encoder()` — static factory with lazy import (avoids circular deps)
- Added `run_encoded(encoded: EncodedOutput)` — extracts rpascript text and executes
- Added `encode_and_run(ops, machine, doc)` — encode then run in one call
- Fixed import pattern: `if TYPE_CHECKING`/`else` for optional `ruidadriver` dependency
- Fixed `_require_connected() -> RdDriver` returning narrowed non-Optional type (fixes pyright)

#### Lint fix in `RpaRpcAdapter` (`rpa_adapter.py`)
- Fixed pyright `reportOptionalMemberAccess` in connection loop closure (lines 269-270)

#### Driver registration (`driver/__init__.py`)
- Registered `RuidaRPAAdapter` in driver registry so `get_driver_cls("RuidaRPAAdapter")` resolves correctly
- Fixes: `generic-ruida-rpa` device profile can now be selected in Machine Settings

### Verification

| Check | Result |
|---|---|
| flake8 (all changed files) | ✅ PASS |
| pyflakes (all changed files) | ✅ PASS |
| pyright (all changed files) | ✅ PASS (0 new errors) |
| generic-ruida-rpa profile selectable | ✅ Fixed |
| Existing ruida encoder tests | ✅ All 62 pass |

### Files changed

- `rayforge/machine/driver/ruidarpa/rpa_encoder.py` — Full implementation (new)
- `rayforge/machine/driver/ruidarpa/rpa_direct_driver.py` — Encoder integration + lint fixes
- `rayforge/machine/driver/ruidarpa/rpa_adapter.py` — Pyright OptionalMemberAccess fix
- `rayforge/machine/driver/__init__.py` — Register RuidaRPAAdapter driver class

## 2026-06-21 — RPA RPC client bug fixes (BgServingThread + disconnect + reconnect)

### Changes made

#### RPA RPC client (`rpa_rpc_client.py`)
- **BgServingThread integration**: Added import (lazy via `TYPE_CHECKING` guard), background serving thread creation in `connect()`, proper teardown in `disconnect()`, liveness check in `is_connected`, import guard in `_ensure_imported()` — ensures server-initiated callbacks are processed reliably on a daemon thread.
- **disconnect() deadlock fix**: Close connection FIRST to unblock `serve()` in the bg thread via socket error, then stop `BgServingThread`. Catches `AssertionError` when bg thread already exited due to socket close. Reverse cleanup order eliminates indefinite `thread.join()` block.
- **is_connected AttributeError fix**: `BgServingThread` has no `is_alive()` method — fixed to use `_bg_thread._thread.is_alive()`.
- **Unregister listener methods**: Added `unregister_status_listener`, `unregister_error_listener`, `unregister_reply_listener` with backward compatibility (`except AttributeError: pass`) for servers that don't support unregister (pre-v0.8.0).
- **Import pattern fix**: Switched from eager `import rpyc` to `if TYPE_CHECKING`/`else` guarded import for cleaner type checking and runtime optional dependency handling.

#### RPA adapter (`rpa_adapter.py`)
- **Reconnect bug fix**: `_stop_backend()` destroyed `self._backend` by setting it to None, causing every retry to fail with "Backend not initialized". Fixed by moving `self._backend = None` to `cleanup()` only.
- **UI freeze fix**: `register_status_listener`/`register_error_listener` were synchronous RPyC RPC calls on the event loop thread, freezing GTK UI. Fixed by wrapping in `run_in_executor`.
- **Callback registration**: Added status/error/reply listener registration in `_connection_loop` after successful connect.
- **Callback unregistration**: Added `unregister_status_listener`/`unregister_error_listener` calls in `_stop_backend()` before stop/disconnect, preventing stale-callback warnings on the server.
- **Callback handlers**: Added `_on_rpc_status` (handles CONNECTED/DISCONNECTED/TERMINATED string events + StatusDict dict updates), `_on_rpc_error` (logs), and `_on_rpc_reply` (logs).

### Verification

| Check | Result |
|---|---|
| flake8 (all changed files) | ✅ PASS |
| pyflakes (all changed files) | ✅ PASS |
| pyright (all changed files) | ✅ PASS (0 new errors; 16 pre-existing in unrelated files) |
| Translations compile | ✅ PASS |
| BgServingThread import check | ✅ `rpyc.utils.helpers.BgServingThread` imports correctly |

### Files changed

- `rayforge/machine/driver/ruidarpa/rpa_rpc_client.py` — BgServingThread, disconnect deadlock fix, is_connected fix, unregister methods, import pattern
- `rayforge/machine/driver/ruidarpa/rpa_adapter.py` — Reconnect fix, UI freeze fix, callback registration/unregistration/handlers

## 2026-06-21 (later) — RPyC callable identity fix + shutdown guard + event routing

### Changes made

#### RPA RPC client (`rpa_rpc_client.py`)
- **RPyC callable identity fix**: Bound methods create a new Python object on each access (e.g., `self._on_rpc_status`), giving a different `id()` each time. RPyC uses `id()` for export identity, so `unregister_*_listener` silently failed — the server never found the callback to remove. Fixed by storing the callback reference at register time (`self._status_listener = callback`) and using the stored reference (not the passed parameter) for unregistration.
- **Register methods updated**: `register_status_listener`, `register_error_listener`, `register_reply_listener` now store the callback in `self._*_listener` before exporting via RPyC.
- **Unregister methods updated**: `unregister_*_listener` now use `self._*_listener` (the stored reference) instead of the passed `callback` parameter, ensuring `id()` matches between register and unregister. Clears the stored reference to `None` after unregister.

#### RPA adapter (`rpa_adapter.py`)
- **Shutdown guard (`_shutting_down`)**: Added `self._shutting_down: bool = False` in `__init__`. Set `True` at the top of `cleanup()` before cancelling the connection task. All three `_on_rpc_*` callbacks early-return if `_shutting_down` is True, preventing late-arriving callbacks (e.g., a 'CONNECTED' callback re-setting `_is_connected = True` while cleanup is in progress) from making invalid state changes.
- **Event routing**: `_connection_loop` no longer sends `connection_status_changed` events. These are now sent from `_on_rpc_status` when it handles 'CONNECTED' (`TransportStatus.CONNECTED`), 'DISCONNECTED' (`TransportStatus.DISCONNECTED`), and 'TERMINATED' (`TransportStatus.DISCONNECTED`). The CONNECTING and ERROR events remain in `_connection_loop` since they fire before any callback is registered or when no connection exists.

### Verification

| Check | Result |
|---|---|
| flake8 (all changed files) | ✅ PASS |
| pyflakes (all changed files) | ✅ PASS |
| pyright (all changed files) | ✅ PASS (0 new errors; 16 pre-existing in unrelated files) |
| Translations compile | ✅ PASS |

### Files changed

- `rayforge/machine/driver/ruidarpa/rpa_rpc_client.py` — Stored callback references for reliable unregister
- `rayforge/machine/driver/ruidarpa/rpa_adapter.py` — Shutdown guard, event routing moved to _on_rpc_status

## 2026-06-23 — Cursor position updates, can_jog, netref fix, origin-aware move_to

### Changes made

#### RPA adapter (`rpa_adapter.py`)
- **Cursor position updates**: Added `_unwrap_um` helper to extract integer µm
  from `MEM_CURRENT_POSITION_*` fields, which may be bare ints or
  `(int_um, str_description)` tuples. Position extraction in `_on_rpc_status`
  dict branch converts µm → mm and updates `self.state.machine_pos` via
  `replace()`, emitting `state_changed` on position change.
- **`(current[N] or 0.0)` fix**: When position dict lacks some axis values,
  the old `None` from initial `DeviceState()` was preserved. Downstream
  `all(p is not None)` guard in `surface.py` blocked cursor movement. Now
  missing axes default to `0.0`.
- **Netref dict conversion**: RPyC netref proxy dicts don't unwrap with
  `dict(event)` — tuple values remain as proxy objects, causing
  `isinstance(value, (list, tuple))` checks in `_unwrap_um` to fail.
  Fixed by using `{k: event[k] for k in event}` dict comprehension which
  creates a clean local dict.
- **Jog speed**: `jog()` now uses the `speed` parameter from the GUI
  (mm/min), converting to mm/s for the RPA TUI service. Previously
  hardcoded to `Speed:600`.
- **`can_jog()`**: Added override returning `True` (was inheriting `False`
  from base `Driver` class).

#### Bottom panel (`bottom_panel.py`)
- **Origin-aware `_on_move_to_position`**: The `"ll"` and `"ur"` coordinate
  mappings now respect `machine.origin` for all four corners:
  - `BOTTOM_LEFT` (south-west): ll=min, ur=max (standard, unchanged)
  - `BOTTOM_RIGHT` (south-east): ll=(max_x, min_y), ur=(min_x, max_y)
  - `TOP_LEFT` (north-west): ll=(min_x, max_y), ur=(max_x, min_y)
  - `TOP_RIGHT` (north-east): ll=max, ur=min (Ruida)
- Fixes "Move to Lower-Left" / "Move to Upper-Right" buttons for Ruida
  where home is NORTH-EAST (TOP_RIGHT origin).

### Verification

| Check | Result |
|---|---|
| flake8 (all changed files) | ✅ PASS |
| pyflakes (all changed files) | ✅ PASS |
| pyright (all changed files) | ✅ PASS (0 new errors; 16 pre-existing in unrelated files) |

### Files changed

- `rayforge/machine/driver/ruidarpa/rpa_adapter.py` — Cursor position updates, netref dict conversion, jog speed, can_jog
- `rayforge/ui_gtk/doceditor/bottom_panel.py` — Origin-aware ll/ur coordinate mapping

## 2026-06-24 — Encoder UUID crash fix, home() refactor, ; prefix removal

### Changes made

#### RPA encoder (`rpa_encoder.py`)
- **UUID-safe laser_uid parsing**: `laser_uid` is a string (e.g. `"laser_42"` or
  a UUID like `"1d3c9891-8f3a-4061-b94b-fe7f536647ec"`). The old
  `laser_uid % 2` triggered string formatting TypeError, and the first fix
  using `int(laser_uid.split("_")[-1]) % 2` crashed on UUIDs with
  `ValueError`. Now uses try/except: attempts numeric suffix extraction,
  falls back to `sum(ord(c) for c in laser_uid) % 2` for non-numeric UIDs.
- **`; ` comment prefix removed** from `_handle_layer_start`,
  `_handle_layer_end`, `_handle_workpiece_start`, `_handle_workpiece_end`.
  Markers like `LAYER_START` and `WORKPIECE_START` are now emitted as
  Rpascript commands instead of comments, so the RPA TUI service sees them.

#### RPA adapter (`rpa_adapter.py`)
- **Home() refactored**: `SPEED_LASER_1 Speed:600` is now always prepended
  before execution (inserted at index 0), not only in the axes-specified
  branch. The `axes=None` case now emits `REL_MOVE_XY Option=0 X=-5 Y=-5`
  instead of `HOME_XY`/`HOME_Z` — moves off limit switches rather than
  homing.

### Verification

| Check | Result |
|---|---|
| flake8 (all changed files) | ✅ PASS |
| pyflakes (all changed files) | ✅ PASS |
| pyright (all changed files) | ✅ PASS (0 new errors; 16 pre-existing in unrelated files) |

### Files changed

- `rayforge/machine/driver/ruidarpa/rpa_encoder.py` — UUID-safe laser_uid parsing, removed `;` prefix from layer/workpiece markers
- `rayforge/machine/driver/ruidarpa/rpa_adapter.py` — Home() refactored

## 2026-06-24 (later) — _emit refactored to accept List[str] for batch encoding

### Changes made

#### RPA encoder (rpa_encoder.py)
- **_emit signature changed**: `_emit(self, line: str)` → `_emit(self, lines: List[str])`.
  Implementation uses `self.lines.extend(lines)` for O(1) batch append.
- **All 21 single-line call sites**: Wrapped argument in `[...]` list literal.
- **`_handle_job_start`**: Batched `["SET_ABSOLUTE", "START_PROCESS"]` into a single call.
- **`_handle_job_end`**: Batched `["BLOCK_END", "SET_FILE_SUM"]` into a single call.
- **`_handle_set_frequency`**: Multi-line emit argument properly bracketed.

### Verification

| Check | Result |
|---|---|
| flake8 | ✅ PASS |
| pyflakes | ✅ PASS |
| pyright | ✅ PASS (0 new errors; 16 pre-existing in unrelated files) |
| Code review (3 reviews) | ✅ All APPROVE, zero issues |

### Files changed

- `rayforge/machine/driver/ruidarpa/rpa_encoder.py` — _emit refactored, all call sites updated

## 2026-07-08 — supported_wcs connection guard removed, read_wcs_offsets emits all three WCS slots

### Changes made

#### RPA adapter (`rpa_adapter.py`)
- **supported_wcs connection guard removed**: `supported_wcs` previously returned `["MACHINE"]` when disconnected and `["MACHINE", "REF0", "REF1"]` when connected. Now it always returns all three, since users may define layers with the machine off.
- **read_wcs_offsets fills all three coordinate systems**: Now emits `MACHINE`, `REF0`, and `REF1` (all with `(0, 0, 0)` offset) instead of only `MACHINE`. This ensures `Machine.update_wcs_offsets_batch()` populates all three into `Machine.coordinate_systems`, so the WCS dropdown in Layer Settings shows all options.

### Verification

| Check | Result |
|---|---|
| flake8 (rpa_adapter.py) | ✅ PASS |
| pyflakes (rpa_adapter.py) | ✅ PASS |
| pyright (rpa_adapter.py) | ✅ PASS (0 new errors) |

### Files changed

- `rayforge/machine/driver/ruidarpa/rpa_adapter.py` — `supported_wcs` guard removed, `read_wcs_offsets` emits all three WCS slots

## 2026-07-18 — Missing CommandType handlers in RPA encoder

### Changes made

#### RPA encoder (`rpa_encoder.py`)
- **SET_AIR_ASSIST handler**: New `_handle_set_air_assist` method reads `AirAssistMode`
  directly from `ops.air_assist(idx)` (returns `AirAssistMode.ON` or `AirAssistMode.OFF`).
  Emits `AIR_ASSIST_ON`/`AIR_ASSIST_OFF` with redundant-state deduplication.
  Import added: `from raygeo.ops.state import AirAssistMode`.
- **SET_COOLANT legacy path preserved**: Renamed `_handle_air_assist` →
  `_handle_coolant_as_air_assist`. Still checks `ops.coolant(idx) == "Air"` for
  backward compatibility. Updated dispatch comment replacing stale TODO.
- **No-op handlers added**: `SET_SPINDLE_RPM`, `SET_HEAD_COOLANT`,
  `STATE_BLOCK_START`, `STATE_BLOCK_END` — all emit nothing but prevent
  `ValueError("Unknown command type")` crashes.
- **_handle_command dispatch**: 5 new elif branches added before the `else: raise`.

### Verification

| Check | Result |
|---|---|
| flake8 | ✅ PASS |
| pyflakes | ✅ PASS |
| pyright (rpa_encoder.py) | ✅ PASS (0 new errors; 19 pre-existing in unrelated files) |
| Translations compile | ✅ PASS |
| ruidarpa tests | ⚪ 0 tests collected (no test files exist) |

### Files changed

- `rayforge/machine/driver/ruidarpa/rpa_encoder.py` — SET_AIR_ASSIST, no-op handlers, renamed legacy method

## 2026-07-25 — Skip `# Ops command:` comment for MOVE_TO and LINE_TO

### Changes made

#### RPA encoder (`rpa_encoder.py`)
- **Guard on comment emit in `_handle_command`**: The `# Ops command: {ct.name}`
  comment was emitted for every command type, including `MOVE_TO` and `LINE_TO`.
  These two ops are high-frequency movement commands whose handlers already
  produce meaningful rpascript (`MOVE_ABS_XY`, `CUT_ABS_XY`), making the
  comment pure output noise. Added a guard that skips the comment emit when
  the command type is `MOVE_TO` or `LINE_TO`.

### Verification

| Check | Result |
|---|---|
| flake8 (rpa_encoder.py) | ✅ PASS |
| pyflakes (rpa_encoder.py) | ✅ PASS |
| pyright (rpa_encoder.py) | ✅ PASS (0 new errors) |
| ruidarpa tests | ⚪ 0 tests exist for this file |

### Files changed

- `rayforge/machine/driver/ruidarpa/rpa_encoder.py` — Guard on `# Ops command:` comment for MOVE_TO and LINE_TO

## 2026-08-07 — Driver test suite added (encoder, adapter, RPC client, golden)

### Changes made

Since the last entry (2026-07-25), no tests existed for the driver — `pixi run -e ruidarpa test` collected zero tests. This entry closes that coverage gap with a four-file test suite plus a golden output fixture that locks the encoder's rpascript output.

#### Encoder tests (`test_encoder.py`, 47 tests)
- Unit tests for `RuidaRPAEncoder` covering all 23 CommandTypes: rpascript line output, mm coordinate formatting, normalized power → percent conversion, state reset between encodes, arc/scanline linearization, and job/layer/workpiece framing.
- Op-map layout pinning: tests assert exact op-to-machine-code positions for single-, two-, and three-layer layouts, plus reverse-mapping consistency.
- GlueScript plan recording: tests verify `encode()` populates `rpa_plan`, records structural and raw calls, positional arg tuples, replay-without-re-recording, and one-based layer keys.

#### Adapter tests (`test_adapter.py`, 64 tests)
- Run routing: `run`/`run_raw`/`set_hold`/`cancel`/`clear_alarm`/`set_power`/`set_focus_power`/`select_wcs` dispatch to the correct backend calls in both direct and RPC modes.
- Reconnect and listener hygiene: `_stop_backend` unregisters stored listener refs, reconnect does not double-register, and cleanup always runs even when stop/disconnect raise.
- Health polling: status CONNECTED/DISCONNECTED/TERMINATED state transitions, late-event inertness after shutdown, and controller-down / transport-dead reconnect behavior for both modes.
- Live bridge RPC vs direct comparison via the `adapter_pair` fixture.

#### RPC client tests (`test_rpc_client.py`, 11 tests)
- Config and behavior: `is_alive` when disconnected / controller down / transport error, `connect` `sync_request_timeout` passing, `root` guard, staged-job `run_job`, `reset_staged` → `new_gluescript`, head/tail script wrappers → exposed server methods.

#### Golden output (`test_golden.py`, 8 tests + `golden/staged_job.rpas`)
- Byte-exact golden test comparing encoder output for a staged job against the checked-in `staged_job.rpas` fixture; verifies determinism, job framing, layer attribute/action blocks, and the presence of moves/cuts/arc/scan.
- `golden/regen_golden.py` regenerates `staged_job.rpas` when encoder output legitimately changes.

### Verification

| Check | Result |
|---|---|
| pytest 9.1.1 (`pixi run -e ruidarpa test tests/machine/driver/ruidarpa/`) | ✅ 152 passed |

### Files changed

- `tests/machine/driver/ruidarpa/test_encoder.py` — Encoder unit tests (47 tests)
- `tests/machine/driver/ruidarpa/test_adapter.py` — Adapter tests (64 tests)
- `tests/machine/driver/ruidarpa/test_rpc_client.py` — RPC client tests (11 tests)
- `tests/machine/driver/ruidarpa/test_golden.py` — Golden output tests (8 tests)
- `tests/machine/driver/ruidarpa/golden/staged_job.rpas` — Golden output fixture (new)
- `tests/machine/driver/ruidarpa/golden/regen_golden.py` — Golden regeneration script (new)

## 2026-08-07 — Staged job execution over TUI RPC via recorded GlueScript plan

### Problem

TUI RPC mode could not run encoded jobs the way direct mode did. The encoder produced rpascript text, but the RPA TUI service owns its own GlueScript document; handing it raw script text would bypass the server's staged-job machinery and risk divergence between what the client encoded and what the controller executed.

### Solution

The encoder now records every GlueScript call it makes into a replayable plan (rpa_plan on the EncodedOutput) using _RecordedCall and _RecordingGlueScriptProxy, normalizing kwargs to positional argument tuples. In RPC mode the adapter's run() re-stages that plan server-side via exposed_new_gluescript and exposed_stage_gluescript_delta (with require_complete gating and interleaved exposed_add_layer_action forwarding), then executes through run_staged_job. A failed stage calls client._reset_staged() before re-raising. Live runtime commands (jog, home, move_to) remain client-side auto-run RPCs and are never staged.

### Verification

Run routing and staging behavior are covered by test_adapter.py (TestRunRouting, TestStagePlan, TestRunStagedJob), which assert the exact staged-delta sequences — the declare_job render, require_complete False then True, add_layer_action interleave, run_staged_job called once and run_job not called — plus the _reset_staged failure path. test_golden.py locks encoder output byte-for-byte against golden/staged_job.rpas. Full package: pixi run -e ruidarpa test tests/machine/driver/ruidarpa/ → 154 passed; flake8, pyflakes and pyright clean.

## 2026-08-07 — Encoder laser device fallback maps to 1-based devices

### Problem

_handle_set_laser derived the Ruida laser device from the head uid using a 0-based modulo (int(suffix) % 2, and sum(ord(c)) % 2 as fallback). Whenever the machine.heads lookup failed, even tool numbers produced device 0 — an invalid LASER_DEVICE_0 mnemonic.

### Solution

Both fallbacks now map into {1, 2}: ((int(suffix) - 1) % 2) + 1 and (sum(ord(c)) % 2) + 1. The machine.heads tool_number override path is unchanged. The stale debug message was updated to describe the 1-based derivation, and two regression tests exercise the numeric-suffix and char-sum fallback branches.

### Verification

Mapping checked by hand: 1→1, 2→2, 3→1, 4→2; char-sum always {1, 2}. Two new tests in test_encoder.py TestSettingsCommands (test_set_head_numeric_suffix_fallback_selects_device, test_set_head_char_sum_fallback_selects_device) bring encoder tests to 49. Full package: pixi run -e ruidarpa test tests/machine/driver/ruidarpa/ → 154 passed; flake8, pyflakes and pyright clean on the changed files.

## 2026-08-08 — ruida-pa 0.15.0 StatusDict key decoupling

### Problem

The external/ruida-pa checkout was updated from 0.14.0 to 0.15.0. In 0.15.0 the StatusDict keys emitted by RdDriver (and passed through the TUI RPC service to clients) were renamed from Ruida mnemonics to generic keys: MEM_CURRENT_POSITION_X/Y/Z/U → POSITION_X/Y/Z/U, MEM_MACHINE_STATUS → MACHINE_STATUS, plus MEM_CARD_ID → CARD_ID and MEM_BED_SIZE_X/Y → BED_SIZE_X/Y (the latter two are not consumed by rayforge). The rpa_adapter.py status-dict consumption would silently stop seeing positions and machine status under the old mnemonic keys.

### Solution

rpa_adapter.py now consumes the new generic keys in _on_rpa_status: event.get("MACHINE_STATUS") for the status value and event.get("POSITION_X/Y/Z") for the current position. This covers both direct and TUI RPC modes, since rpyc_service passes StatusDicts through unchanged. _unwrap_mm is unchanged — it already tolerates both bare values and (float, str) tuples, so no logic change was needed. test_adapter.py event dicts were updated to the new keys (tuple and bare-float position cases, partial-position case, and the machine-status dict case).

### Verification

Full package: pixi run -e ruidarpa test tests/machine/driver/ruidarpa/ → 154 passed; flake8, pyflakes and pyright clean.

## 2026-08-13 — ruida-pa 0.15.2 migration: unified jog/home transport, dwell delay fix

### Problem

The external/ruida-pa checkout was updated from 0.15.0 to 0.15.2. In 0.15.2, RdDriver (direct) and the new RpcRdDriver (RPC) expose an identical jog/home/live-command surface (integration-guide §4 "single-reference pattern"), but the ruidarpa driver still treated the transports separately: the adapter's home()/move_to()/jog() carried direct-mode branches that generated raw rpascript lines and sent them via backend.run(), while RpaDirectDriver had no jog/home methods at all. The encoder also emitted an uppercase `DELAY ...` mnemonic that the rpascript runner silently drops (it matches flow-control mnemonics lowercase-only), so dwells never actually paused the controller.

### Solution

Transport-unified jog/home. RpaDirectDriver gained delegation wrappers for home(), home_z(), jog_xy_to(), the jog_*_rel joggers, and the jog_set_*_speed setters, matching the RpaRpcClient surface exactly. The jog/home wrappers require a live connection (_require_connected) and discard the RdDriver's returned lines — the driver auto-sends them, and calling run() on the result would double-send every jog. The jog_set_*_speed setters delegate without a connection (session-less attribute sets, documented in the docstrings). The adapter's home()/move_to()/jog() no longer branch on _tui_mode; they call the unified backend methods through run_in_executor. jog() re-asserts jog_set_xy_speed on every call (the once-only _ensure_jog_config/_jog_config_initialized logic was removed) so move_to()'s fixed 600 mm/s speed cannot clobber the jog speed; move_to() gained the MOVE_TO_JOG_SPEED_MM_S constant. The isinstance(RpaDirectDriver) guards on get/set head/tail were dropped (RpaRpcClient has head/tail methods); the get_protect/set_protect guard remains because RpaRpcClient lacks protect. The encoder now emits lowercase `delay ...` for dwells and reports the 0.15.2 version floor. New test_direct_driver.py covers the RpaDirectDriver wrappers (delegation, never-run-returned-lines, fail-loud when disconnected); test_adapter.py asserts the unified backend calls in both modes plus a jog→move_to→jog speed-reassertion regression.

### Verification

Full package: pixi run -e ruidarpa test tests/machine/driver/ruidarpa/ → 178 passed (was 154; test_direct_driver.py adds 24). Golden fixture regenerated with golden/regen_golden.py; the diff against the 0.15.0 fixture is exactly two lines — the `# Generated by: GlueScript 0.15.0` → `0.15.2` header and the `DELAY 250.000ms` → `delay 250.000ms` dwell line. flake8 and pyflakes clean. pyright clean on the changed files when resolved against the ruidarpa environment; the default-env lint retains only the pre-existing reportMissingImports for ruidadriver/rpyc (those packages are not installed in the default environment) plus pre-existing unrelated errors.

## 2026-08-13 (follow-up) — move_to uses the GUI jog speed; empty-deltas jog no-op; adapter head/tail removed

### Problem

move_to() always sent absolute jogs at the hardcoded DEFAULT_MOVE_TO_JOG_SPEED_MM_S (600 mm/s), ignoring the jog speed the user chose in the GUI jog widget. The GUI speed reaches the driver only through jog() calls, so move_to() could visibly fight the selected speed. A speed-only jog() call (no axis deltas) also sent a pointless jog_set_xy_speed to the backend. And with GlueScript staging providing ref-point framing, the adapter's get/set head/tail script methods and its on-connect clearing of inherited head/tail scripts were dead weight.

### Solution

jog() now records each call's converted mm/s speed in _jog_speed_mm_s before any axis dispatch, and move_to() uses that stored speed (falling back to DEFAULT_MOVE_TO_JOG_SPEED_MM_S before the first jog) instead of a hardcoded constant. An empty-deltas jog() is now a no-op after tracking the speed (with an inline comment noting the speed is recorded before the early return), so it updates the move_to speed without touching the backend. Head/tail script support has been removed from the entire ruidarpa stack: the adapter's get_head_script/set_head_script/get_tail_script/set_tail_script methods and the connection-loop head/tail clearing, the RpaDirectDriver/RpaRpcClient head/tail wrappers, and run_job's head+job+tail composition are all gone. Every script — jobs and runtime commands — is sent raw via backend.run(); the encoder's self-framed staged output (REF_POINT_ABSOLUTE/SET_ABSOLUTE/REF_POINT_SET/START_JOB…END_JOB) needs no head/tail, and RdDriver's intentional non-empty default head script is therefore never composed. TUI RPC mode is the one exception to the removal: the server's RdDriver still composes its non-empty default head script around every staged job (run_job(None) concatenates head + staged rpascript + tail), so the RPC client re-clears the server driver's head/tail to [] at connect — the encoder output is self-framed, and without that clearing every staged job would get duplicate ref-point framing plus ENABLE_BLOCK_CUTTING State:OFF prepended at the controller. Direct mode needs no clearing: RpaDirectDriver has no run_job, and its run()/run_raw() path composes nothing. test_adapter.py's connect test asserting head/tail are never touched was deleted and replaced with a mode-dependent test asserting TUI RPC clears the server driver's head/tail to [] while direct mode never touches it (structurally impossible to call there).

### Verification

Full package: pixi run -e ruidarpa test tests/machine/driver/ruidarpa/ → 191 passed (was 189; the TUI RPC connect-time head/tail clearing adds two mode-dependent tests). flake8 and pyflakes clean on the changed files. pyright clean on the changed files when resolved against the ruidarpa environment; the default-env lint retains only the pre-existing reportMissingImports for ruidadriver/rpyc plus pre-existing unrelated errors.

## 2026-08-13 — move_to no longer inverts machine-frame coordinates; stale UI-inversion TODO removed

### Problem

`move_to()` negated `pos_x`/`pos_y` behind a stale TODO ("The coordinates
coming from the UI are inverted. Why?"). The GUI already computes machine-frame
coordinates (`bottom_panel._on_move_to_position` converts world→machine via
`world_point_to_machine` with TOP_RIGHT origin handling, then subtracts WCS
offsets before calling `driver.move_to`), `machine_pos` is reported from
`POSITION_*` status keys with no inversion, and `jog()` passes deltas through
unchanged. `move_to()` therefore double-inverted: it sent the head to the
mirror-image of the requested position (e.g. a requested +X move left of home
went right of home). The negation predated the coordinate-space origin handling
(2026-06-23) and was the last remaining sign-flip in the driver stack — the
jog-delta inversion had already been removed in 748e46f3 and the status-path
negation in the GlueScript rewrite (51ee674b).

### Solution

Deleted the 4-line negation/TODO block from `rpa_adapter.py` `move_to()`;
`pos_x`/`pos_y` now flow through unchanged to `backend.jog_xy_to()`. Added a
`move_to()` docstring documenting the machine-frame contract (same frame as
`POSITION_*` reporting; +X left of home, +Y down from home).
`test_adapter.py`'s two default-speed tests (RPC and direct modes) updated to
assert pass-through `jog_xy_to(10.0, 20.0)` — non-zero coordinates so a
re-introduced negation would be caught. The still-valid mm/min→mm/s jog-speed
TODO was left in place.

### Verification

Full package: `pixi run -e ruidarpa test tests/machine/driver/ruidarpa/` → 193
passed (both updated `test_move_to_before_any_jog_uses_default_speed` variants
pass). flake8 and pyflakes clean. pyright clean on the changed files when
resolved against the ruidarpa environment; the default-env lint retains only
the pre-existing reportMissingImports for ruidadriver/rpyc plus pre-existing
unrelated errors.

## 2026-08-14 — RuidaRPAAdapter WCS uses the four ruida-pa reference points

### Problem

The RuidaRPAAdapter (`rayforge/machine/driver/ruidarpa/rpa_adapter.py`) exposed
WCS slots `["MACHINE", "REF0", "REF1"]` and `select_wcs` translated REF0/REF1
into rpascript `REF_POINT_1`/`REF_POINT_2`. Those are not valid ruida-pa
mnemonics: the rpascript ScriptParser mnemonic map is built from the CT command
table and has no REF_POINT_1/2; the old REF0↔REF_POINT_CURRENT / REF1↔
REF_POINT_ANCHOR mapping existed only in the reference-only `ruida/` prototype.
The encoder (`rpa_encoder.py` `_handle_job_start`) also hardcoded
`declare_job(label, "MACHINE", ...)` regardless of the machine's active WCS, so
a user-selected reference point never reached the job framing. Finally, the UI
WCS-switch path (`ui_gtk/doceditor/bottom_panel.py` → `machine.switch_active_wcs`
→ `controller.switch_active_wcs`) sent the bare WCS string through
`driver.run_raw(wcs)`, bypassing the driver's `select_wcs` method entirely — for
ruidarpa that meant a bare "G55"-style string reached the backend with no
validation.

### Solution

`rpa_adapter.py` `supported_wcs` now returns the four ruida-pa reference points
`["MACHINE", "ANCHOR", "CURRENT", "SET_POINT"]` (the RELT names from ruida-pa's
ruida_protocol.py). `select_wcs(wcs)` guard-validates the name (raising
`ValueError` listing the valid names otherwise) and saves it to
`self._selected_wcs` (initialized `"MACHINE"`); it no longer sends any script.
`read_wcs_offsets()` returns all four slots at `(0.0, 0.0, 0.0)` and fires
`wcs_updated` so the UI dropdown, driven by `machine.supported_wcs`, picks up
the four names automatically. `read_parser_state()` returns `self._selected_wcs`
(the device-committed selection) instead of None. `machine_space_wcs` stays
`"MACHINE"` and `set_wcs_offset` still raises `NotImplementedError`.

`rpa_encoder.py` `_handle_job_start` now derives the GlueScript `declare_job`
ref_point from `machine.active_wcs` via a module-level `_WCS_TO_REF_POINT` dict:
MACHINE→"MACHINE", ANCHOR→"ABSOLUTE" (with `abs_xy=None`, which ruida-pa's
rd_gluescript converts to `[0.0,0.0]` — safe), CURRENT→"CURRENT", SET_POINT→
"SET_POINT". The framework default `"G54"` (machine.py:162) falls back to the
MACHINE reference point with a `logger.warning` that fires only when the value
changes (module-level `_last_fallback_wcs` dedup; a fresh encoder per encode
keeps the state module-level, and a real WCS resets it). Any other unknown WCS
raises `ValueError` listing the valid names. This keeps the golden fixture
byte-identical, since the golden machine carries the default `"G54"`.

`rayforge/machine/models/controller.py` `switch_active_wcs` now routes to
`driver.select_wcs(wcs)` when the driver overrides the base no-op (detected via
`type(self.driver).select_wcs is not Driver.select_wcs`), else falls back to
`driver.run_raw(wcs)` — so ruida/ruidarpa get validated selection while
grbl/smoothie/marlin/octoprint/dummy keep their previous behavior. It validates
`wcs` against `driver.supported_wcs` in the connected branch only, BEFORE
mutating `machine.active_wcs` (comment: "Mirrors select_wcs so active_wcs is
not mutated for a rejected WCS"), so a rejected selection raises `ValueError`
and leaves model state untouched; the disconnected path still records
model-only intent without raising, so a never-connected machine's G54–G59
dropdown picks must not fail. The plan doc
(`rayforge/machine/driver/ruidarpa/docs/plans/ruidarpa-implementation-plan.md`)
was updated (decision-table WCS model row, `supported_wcs` description, and
Phase 4.1 WCS ops spec).

### Verification

New/updated tests in test_adapter.py: `test_select_wcs_valid_saves_selection`
(four names × direct/rpc), `test_select_wcs_invalid_raises_value_error`
(REF0/REF1/G55 × direct/rpc, asserts `_selected_wcs` unchanged and
`backend.run` not called), `test_read_wcs_offsets_returns_four_slots` (four
zero slots plus `wcs_updated` fired), `test_read_parser_state_returns_selected_wcs`,
and `test_set_wcs_offset_raises_not_implemented` slot updated to "SET_POINT".
test_encoder.py gains `TestWcsToRefPoint` (parametrized mapping incl.
ANCHOR→ABSOLUTE with `abs_xy=None` pinned, G54→MACHINE, G55→ValueError,
`machine=None`→MACHINE, warn-only-when-changed with module-state reset fixture +
caplog). `tests/machine/models/test_machine_controller.py` adds fake-driver
routing tests (overridden `select_wcs` vs base `run_raw`), a pre-mutation guard
(invalid raises, `machine.active_wcs` and `_confirmed_active_wcs` untouched),
disconnected intent-recording, and a confirm-failure path.

Full package: `pixi run -e ruidarpa test tests/machine/driver/ruidarpa/` → 208
passed; the 7 remaining failures are pre-existing ruida-pa version drift
(installed GlueScript 0.15.4 vs fixture 0.15.2: the `MOVE_NEAR_XY nearX=`
format and the `# Generated by: GlueScript` header line) — identical before and
after, not regressions. `pixi run test tests/machine/models/test_machine_controller.py`
→ 8 passed. flake8 and pyflakes clean; pyright at the pre-existing 20-error
baseline (0 new). Golden fixture byte-identical (verified via `cmp` against the
git-stashed baseline, 12072 chars). Code review approved with no critical/major
issues; the review follow-ups (3 SHOULD test-coverage items + 2 NITs: aligned
`ValueError` message listing valid names, benign-race comment on
`_last_fallback_wcs`) were all addressed and re-verified. Constraints honored:
no changes under `rayforge/machine/driver/ruida/` (reference-only prototype) or
`external/ruida-pa/`.
