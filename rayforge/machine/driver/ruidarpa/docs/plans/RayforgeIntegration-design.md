# AGENTS
**IMPORTANT:** This is a design document which defines the integration of the Ruida Protocoal Analyzer (RPA) driver into the Rayforge application and is intended to be the basis for writing a corresponding implementation plan. Do not modify this document. Ask questions for clarification.

A git clone of the RPA driver source code is stored in `external/ruida-protocol-analyzer`.

The driver integration guide is in `external/ruida-protocol-analyzer/docs/guides`.

In the following the pattern **AGENT:** indicates a specific instruction for an agent.
# Overview
This document describes features and behaviors needed to integrate the RPA driver into Rayforge. 

Integration supports two modes, development or production. 

The development mode is a TUI RPC client to connect to a TUI server via RPyC to expose driver behavior for integration testing. Assume the TUI RPC server binds to `localhost` or 127.0.0.1 on port 18812. Therefore, authentication is not used.

The production mode uses the Ruida driver (`RdDriver`) directly. A configuration option is used to control which mode is active.
# Goal
Create a `RuidaRPAAdapter(Driver)` with two modes (direct `RdDriver` in-process or TUI RPC client). Also, create a corresponding `RuidaRPAEncoder(OpsEncoder)` which supports `rpascript`.
# Context and Decisions

- **Name**: RuidaRPA
- Development git branch: `ruidarpa`
- Pixi environment (`pixi.toml` already updated): `ruidarpa`
- Use existing `ruidarpa/` directory.
- **Driver maturity**: EXPERIMENTAL
- Test scripts and programs to be stored in `ruidaarp/test`.
- Device profiles in `rayforge/resources/devices` are `generic-ruida-rpa` and `monport-mp570-c02`.
# Driver Configuration Variables
The following `VarSet` variables are defined in `get_setup_vars`.

**Ruida Controller IP Address**
```
key="udp_host",
label="Hostname",
description="The IP address or hostname of the Ruida controller.",
```

**Ruida Controller USB Device**
```
key="usb_device",
label="USB",
description="The USB serial device",
```
NOTE: The RPA supports three forms:
 - `/dev/<device>` The typical full path form for Linux USB serial devices (e.g. `/dev/ttyUSB0`).
- `<device>` Assumes a `/dev/` prefix for a Linux USB serial device (e.g. `ttyUSB0`). For Windows this is a normal Windows serial device (e.g. `COM1`).
- `<vid>:<pid>` The USB vendor and device IDs form.

**Remote Control via RPyC**
```
key="tui",
label="TUI RPC",
description="Enable TUI RPC connection.",
```
This is a boolean flag that, when True, enables TUI RPC mode to connect to a RPA RPyC server. When False `RdDriver` is used directly in-process.
