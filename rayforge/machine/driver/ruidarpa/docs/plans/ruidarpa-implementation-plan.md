---
status: not-started
phase: 1
updated: 2026-06-19
---
# Implementation Plan: RuidaRPA Driver Integration

## Goal
Implement a `RuidaRPAAdapter(Driver)` with two modes (direct `RdDriver` in-process and TUI RPyC client) and a `RuidaRPAEncoder(OpsEncoder)` producing rpascript output, enabling Ruida controller support via the RPA library.

## Context & Decisions
| Decision | Rationale | Source |
|----------|-----------|--------|
| Two-mode architecture (tui bool flag) | Dev mode uses RPyC for isolation, production uses direct RdDriver | `RayforgeIntegration-design.md:14-16` |
| Adapter wraps RdDriver (synchronous/threaded) | RdDriver uses daemon threads internally; Rayforge driver API is async | `ruidadriver/ruida_driver.py:95-109` |
| Adapter bridges asyncio ↔ threaded via `asyncio.to_thread()` / `loop.run_in_executor()` | Keeps event loop responsive; RdDriver's threaded API requires it | `integration-guide.md:134-157` |
| WCS model matches RuidaDriver ("MACHINE", "REF0", "REF1") | Consistent with existing Ruida UDP driver behavior | `ruida_driver.py:85-97` |
| Encoder produces rpascript text in `EncodedOutput.text` | Rpascript is the native command format for RdDriver | `rpascript-guide.md`, design doc |
| Driver registration via `register_driver()` + `drivers.append()` in `driver/__init__.py` | Auto-discovery via `isdriver()` checks `Driver` subclass; device YAML references class name | `driver/__init__.py:30-32` |
| Maturity = EXPERIMENTAL | Design doc explicitly states "Driver maturity: EXPERIMENTAL" | `RayforgeIntegration-design.md:26` |
| `ruidarpa` pixi environment already configured | Includes `ruida-protocol-analyzer` and `rpyc` deps | `pixi.toml:10,15-16` |
| Device profiles already exist for `generic-ruida-rpa` and `monport-mp570-co2` | Both reference `driver: RuidaRPAAdapter` | `resources/devices/generic-ruida-rpa/device.yaml`, `resources/devices/monport-mp570-co2/device.yaml` |
| Auto-reconnect: exponential backoff 1s→2s→4s→max 30s with jitter | Prevents reconnect storms; emits `connection_status_changed` on each attempt | Review feedback (Fail Fast, Law 4) |
| Rpascript uses mm natively — no unit conversion in encoder | Rpascript parameters accept mm; RdDriver handles internal µm conversion | `rpascript-guide.md:228-234` |
| **precheck allows both udp_host and usb_device** | RPA auto-selects available interface; requiring exactly one would be incorrect | User feedback, RPA documentation |

## Phase 1: Foundation & Package Structure [PENDING]
- [x] **1.1 Create `rayforge/machine/driver/ruidarpa/__init__.py`** — package init, exports `RuidaRPAAdapter`, `RuidaRPAEncoder` ✅ 2026-06-19
- [ ] **1.2 Create `rayforge/machine/driver/ruidarpa/rpa_adapter.py`** — implement `RuidaRPAAdapter(Driver)` class skeleton with:
  - Class attributes: `label=_("Ruida RPA")`, `subtitle=_("Connect via Ruida Protocol Analyzer")`, `supports_settings=False`, `reports_granular_progress=False`, `uses_gcode=False`, `maturity=DriverMaturity.EXPERIMENTAL`
  - `__init__(self, context, machine)` — store mode flag (`_tui_mode: bool = False`), driver params
  - `machine_space_wcs` → `"MACHINE"`, `machine_space_wcs_display_name` → `_("Machine Coordinates")`
  - `supported_wcs` property: returns `["MACHINE"]` before connection, then `list(self._client.ref_points)` in direct mode, or `["MACHINE", "REF0", "REF1"]` in RPC mode
  - `resource_uri` property: returns `f"ruidarpa://{host_or_usb}"` for resource conflict detection
  - `get_setup_vars(cls)` → `VarSet` with:
    - `udp_host` (HostnameVar, key="udp_host", label=_("Hostname"))
    - `usb_device` (StringVar, key="usb_device", label=_("USB"), description including three forms)
    - `tui` (BoolVar, key="tui", label=_("TUI RPC"), description="Enable TUI RPC connection")
  - `precheck(cls, **kwargs)` — validate **at least one** of `udp_host` or `usb_device` provided (raise `DriverPrecheckError` if neither). **Both is allowed** — RPA auto-selects the available interface. No validation on individual values beyond emptiness.
  - `_setup_implementation(**kwargs)` — parse config into `self._config` dict, set `self._tui_mode`, instantiate appropriate backend wrapper (deferred to Phase 2/3)
  - `get_laser_capabilities(laser)` — return `PWMCapability` for non-diode lasers (matching `RuidaDriver`)
  - `create_encoder(cls, machine)` → `RuidaRPAEncoder()`
- [ ] **1.3 Stub all remaining abstract methods** (16 total from `Driver` ABC) with appropriate defaults:
  - `_connect_implementation()` — deferred to Phase 2/3
  - `run()`, `run_raw()` — deferred to Phase 2/3
  - `set_hold()`, `cancel()` — deferred to Phase 4
  - `home()`, `move_to()` — deferred to Phase 4
  - `jog()` — deferred to Phase 4
  - `select_tool()` — no-op matching RuidaDriver
  - `read_settings()` — emit empty `settings_read`
  - `write_setting()` — no-op
  - `clear_alarm()` — deferred to Phase 4
  - `set_power()`, `set_focus_power()` — deferred to Phase 4
  - `set_wcs_offset()`, `read_wcs_offsets()` — deferred to Phase 4
  - `read_parser_state()` — return `None`
  - `run_probe_cycle()` — return `None` (probe not supported)
  - `select_wcs()` — no-op
  - `get_setting_vars()` → `[VarSet(title=_("No settings"))]`
  - `cleanup()` — deferred to Phase 2/3
- [ ] **1.4 Register driver** — import `RuidaRPAAdapter` in `rayforge/machine/driver/__init__.py`, add to `__all__`, call `register_driver(RuidaRPAAdapter)`, and append to `drivers` list
- [ ] **1.5 Set up logger** — `logger = logging.getLogger(__name__)` in adapter module; use `_log_extra()` with category `"TUI_RPC"` or `"RPA"` for all log calls

## Phase 2: Direct Mode — RdDriver In-Process [PENDING]
- [ ] **2.1 Create `rayforge/machine/driver/ruidarpa/rpa_direct_driver.py`** — wrapper around `RdDriver`:
  - `__init__` — instantiate `RdDriver()`
  - `start(udp_host, usb_device)` → `RdDriver.start()`
  - `stop()` → `RdDriver.stop()`
  - `run_script(script_lines, auto_checksum=False)` → `RdDriver.run()`
  - `cancel_script()` → `RdDriver.cancel_script()`
  - Register listeners and forward to Rayforge signals:
    - Status listener → `call_soon_threadsafe` → emit `connection_status_changed`, `state_changed`, `wcs_updated`
    - Error listener → `call_soon_threadsafe` → log at ERROR
    - Reply listener → `call_soon_threadsafe` → log at DEBUG
  - Properties: `is_connected` → `rd_driver.is_connected`, `machine_status` → `rd_driver.machine_status`
  - `resource_uri` → `f"ruidarpa://{host}"` for direct UDP or `f"ruidarpa://{usb_device}"` for USB
- [ ] **2.2 Implement `_connect_implementation` for direct mode**:
  - Start background asyncio task
  - Via executor: call `direct_driver.start(udp_host=..., usb_device=...)`
  - Poll `direct_driver.is_connected` every 500ms
  - Auto-reconnect with **exponential backoff**: 1s → 2s → 4s → ... → max 30s (with ±20% jitter), no max retry count
  - Each reconnect attempt: fire `connection_status_changed(CONNECTING)`, then log WARNING
  - On sustained failure: fire `connection_status_changed(ERROR, message)`
  - On success: fire `connection_status_changed(CONNECTED)`, set `self.state.status = DeviceStatus.IDLE`
  - Reset backoff to 1s on successful connection
- [ ] **2.3 Implement `run()` for direct mode**:
  - Extract rpascript lines from `encoded.text` (splitlines, strip blank lines)
  - Pass to `direct_driver.run_script(script_lines)` via executor
  - Handle `on_command_done`: iterate through `encoded.op_map.op_to_machine_code` keys
  - On completion: fire `self.job_finished`
- [ ] **2.4 Implement `run_raw()` for direct mode**:
  - Split `machine_code` into lines, pass as single script to `direct_driver.run_script()`
- [ ] **2.5 Implement `cleanup()` for direct mode**:
  - Mark `_keep_running = False`
  - Cancel background asyncio task
  - Via executor: call `direct_driver.stop()`
  - Fire `connection_status_changed(DISCONNECTED)`
  - Call `super().cleanup()`

## Phase 3: TUI RPC Mode — RPyC Client [PENDING]
- [ ] **3.1 Create `rayforge/machine/driver/ruidarpa/rpa_rpc_client.py`** — RPyC client wrapper:
  - `connect(host="127.0.0.1", port=18812)` → `rpyc.connect(host, port)`, return `conn.root`
  - `disconnect()` → close connection
  - Delegated API items (all via netref):
    - `start(udp_host, usb_device)` → `svc.start()`
    - `stop()` → `svc.stop()`
    - `run(script_lines)` → `svc.run()`
    - `cancel_script()` → `svc.cancel_script()`
    - `register_status_listener(cb)` → `svc.register_status_listener(cb)`
    - `register_error_listener(cb)` → `svc.register_error_listener(cb)`
    - `register_reply_listener(cb)` → `svc.register_reply_listener(cb)`
    - `is_connected` property → `svc.is_connected()`
    - `machine_status` property → `svc.machine_status()`
  - Netref callback wrappers for thread-safe signal forwarding
- [ ] **3.2 Implement `_connect_implementation` for TUI mode**:
  - Connect to RPyC server via executor
  - Call `svc.start(udp_host=..., usb_device=...)`
  - Register netref callbacks, forwarding to asyncio event loop via `call_soon_threadsafe`
  - Poll `svc.is_connected()` for state tracking
  - Same exponential backoff strategy as Phase 2.2
- [ ] **3.3 Implement `run()` for TUI mode** — extract rpascript lines, call `svc.run()` via executor
- [ ] **3.4 Implement `run_raw()` for TUI mode** — same as direct mode, split to lines, call via RPC
- [ ] **3.5 Implement `cleanup()` for TUI mode** — call `svc.stop()` via executor, disconnect RPyC, fire DISCONNECTED

## Phase 4: Operations Layer (shared across modes) [PENDING]
- [ ] **4.1 WCS operations**:
  - `select_wcs(wcs_slot)` → run rpascript `"REF_POINT_2"` or `"REF_POINT_1"` depending on slot
  - `set_wcs_offset(wcs_slot, x, y, z)` → run `"SET_SETTING MEM_USER_ORIGIN_X=%d MEM_USER_ORIGIN_Y=%d"` or equivalent
  - `read_wcs_offsets()` → run `GET_SETTING` queries for each ref point; fire `wcs_updated`
  - `read_parser_state()` → return `self._machine.active_wcs`
- [ ] **4.2 Movement**:
  - `home(axes)` → `"HOME_XY"` / `"HOME_Z"`
  - `move_to(pos_x, pos_y)` → `"MOVE_ABS_XY X={pos_x}mm Y={pos_y}mm"`
  - `jog(speed, **deltas)` → `"AXIS_X_MOVE X={delta}mm"` / `"AXIS_Y_MOVE Y={delta}mm"`
- [ ] **4.3 Job control**:
  - `set_hold(hold=True)` → `"PAUSE_PROCESS"`, `set_hold(False)` → `"RESTORE_PROCESS"`
  - `cancel()` → `"STOP_PROCESS"`
  - `clear_alarm()` → `"STOP_PROCESS"`
- [ ] **4.4 Power/Laser**:
  - `set_power(head, percent)` → `"IMD_POWER_1 Power={pct}%"` (tool_number determines 1 or 2)
  - `set_focus_power(head, percent)` → same as `set_power`
- [ ] **4.5 Remaining stubs**:
  - `select_tool(tool_number)` → no-op
  - `read_settings()` → emit `settings_read` with empty list
  - `write_setting(key, value)` → no-op
  - `get_setting_vars()` → `[VarSet(title=_("No settings"))]`
  - `run_probe_cycle()` → return `None` (not supported)

## Phase 5: RuidaRPAEncoder [PENDING]
- [ ] **5.1 Create `rayforge/machine/driver/ruidarpa/rpa_encoder.py`** — implement `RuidaRPAEncoder(OpsEncoder)`:
  - State tracking: `power`, `cut_speed`, `travel_speed`, `current_pos`, `active_laser`, `air_assist`
  - `encode(ops, machine, doc)` → `EncodedOutput` with rpascript lines in `text`
  - Command dispatch via `_handle_command()` based on `CommandType`
  - Coordinates stay in mm (rpascript uses mm natively)
  - Power converted from normalized (0.0-1.0) to percentage (0-100%)
- [ ] **5.2 Implement command handlers** mapping Ops to rpascript:
  - `SET_POWER` → `IMD_POWER_{n} Power={percent:.1f}%`
  - `SET_CUT_SPEED` → `SPEED_LASER_{n} Speed={speed:.3f}mm/S`
  - `SET_TRAVEL_SPEED` → `SPEED_AXIS Speed={speed:.3f}mm/S`
  - `SET_FREQUENCY` → `FREQUENCY_PART Laser={n} Part=1 Freq={freq:.3f}KHz`
  - `SET_PULSE_WIDTH` → `LASER_INTERVAL {pw:.3f}mS`
  - `ENABLE_AIR_ASSIST` → `AIR_ASSIST_ON`
  - `DISABLE_AIR_ASSIST` → `AIR_ASSIST_OFF`
  - `SET_LASER` → `LASER_DEVICE_{n}` (derive n from tool_number)
  - `MOVE_TO` → `MOVE_ABS_XY X={x:.3f}mm Y={y:.3f}mm`
  - `LINE_TO` → `CUT_ABS_XY X={x:.3f}mm Y={y:.3f}mm`
  - `ARC_TO` → linearize to `CUT_ABS_XY` segments (same pattern as `RuidaEncoder._handle_arc_to`)
  - `SCAN_LINE` → linearize to power/`CUT_ABS_XY` segments (same pattern as `RuidaEncoder._handle_scan_line`)
  - `JOB_START` → `SET_ABSOLUTE` + `START_PROCESS`
  - `JOB_END` → `BLOCK_END` + `SET_FILE_SUM`
  - `LAYER_START` → `LAYER_END` + `; --- Layer {uid[:8]} ---`
  - `LAYER_END` → `LAYER_END` + `; --- End Layer ---`
  - `WORKPIECE_START` → `; --- Workpiece {uid[:8]} ---`
  - `WORKPIECE_END` → `; --- End Workpiece ---`
- [ ] **5.3 `_reset_state()`** — reset all state tracking between encode passes
- [ ] **5.4 Helper methods** — `_power_to_percent(normalized)`, `_speed_format(speed)`, `_coord_format(mm)`

## Phase 6: Testing [PENDING]
- [ ] **6.1 Write encoder unit test** — `tests/machine/driver/ruidarpa/test_encoder.py`:
  - Verify rpascript output for each command type
  - Test coordinate formatting, power conversion
  - Test `_reset_state()` between encode calls
  - Run with: `pixi run -e ruidarpa test tests/machine/driver/ruidarpa/test_encoder.py`
- [ ] **6.2 Write adapter unit tests** — `tests/machine/driver/ruidarpa/test_adapter.py`:
  - Mock `RdDriver`, verify lifecycle
  - Test mode switching via `tui` config flag
  - Test config parsing: host only, USB only, **both provided** (allowed), neither (expect `DriverPrecheckError`)
  - Test `get_setup_vars` VarSet structure
  - Run with: `pixi run -e ruidarpa test tests/machine/driver/ruidarpa/test_adapter.py`
- [ ] **6.3 Verify with `pixi run lint`** — fix all flake8/pyflakes/pyright errors
- [ ] **6.4 Verify with `pixi run -e ruidarpa test`** — ensure no regressions in existing tests

## Phase 7: Cleanup & Documentation [PENDING]
- [ ] **7.1 Delete deprecated `rayforge/machine/driver/ruidatui/`** — confirmed deprecated by user
- [ ] **7.2 Add implementation reference in `ruidarpa/docs/`** — point to RPA integration guide and rpascript guide
- [ ] **7.3 Final verification** — `pixi run lint`, `pixi run -e ruidarpa test` pass cleanly

## Notes
- The 16 abstract methods on `Driver` ABC that must be implemented: `machine_space_wcs` (property), `machine_space_wcs_display_name` (property), `precheck()`, `_setup_implementation()`, `get_setup_vars()`, `create_encoder()`, `get_setting_vars()`, `_connect_implementation()`, `run()`, `run_raw()`, `set_hold()`, `cancel()`, `home()`, `move_to()`, `select_tool()`, `read_settings()`, `write_setting()`, `clear_alarm()`, `set_power()`, `set_focus_power()`, `jog()`, `set_wcs_offset()`, `read_wcs_offsets()`, `run_probe_cycle()`.
- RPA integration guide: `external/ruida-protocol-analyzer/docs/guides/integration-guide.md`
- Rpascript command reference: `external/ruida-protocol-analyzer/docs/guides/rpascript-guide.md`
- RdDriver source: `external/ruida-protocol-analyzer/ruidadriver/ruida_driver.py` (class at line 54, 804 lines)
- TuiAdapter source: `external/ruida-protocol-analyzer/rpascript/tui_adapter.py` (2859 lines)
- Thread safety: RdDriver listeners fire from background threads — use `asyncio.get_event_loop().call_soon_threadsafe()` to forward to async context
- RPyC auth: RPC server on localhost (127.0.0.1) uses no auth — `\x00` length byte sent before RPyC handshake
- Rpascript uses mm natively — no unit conversion needed in encoder; RdDriver handles internal µm conversion
- Encoder should be pure: same `Ops` input produces identical `EncodedOutput` — no side effects between encode calls except `_reset_state()`
- **precheck revised 2026-06-19**: RPA auto-selects between UDP and USB, so both `udp_host` and `usb_device` may be provided simultaneously. Only the case where neither is provided should raise `DriverPrecheckError`.
