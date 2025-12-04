# Introduction
This document provides an overview of the Ruida driver for Rayforge. Described are overall structure, Ruida idiosyncrasies and how they are handled, and deviations from the Rayforge behavior (if any).

NOTE: Currently only the X and Y axes are supported. The code is tested using a Ruida RDC6442S controller. Behavior with other controllers is unknown at this time even though many should work as well -- providing the correct swizzle magic number is used.

**IMPORTANT**: Raster engraves involve a huge number of direction changes. To avoid distortion of the resulting images and possible damage to hardware (e.g. breaking teeth off of the belt), an overscan is used to allow for acceleration and deceleration of the laser head. For GRBL machines the amount of overscan needs to be configured in the raster layer. Ruida controller based machines, on the other hand, automatically handle overscan with the amount of overscan determined by the speed of the scan and the vendor's settings for acceleration and deceleration. Because of this the overscan for the raster layer needs to be **set to 0**.

# Structure
This driver conforms, in large part, to the OSI communications model and is structured as follows:
```mermaid
block
columns 3
    Application
    block:a
        app["Rayforge (Ops/Jogs)"]
    end
    Operations/Jogging

    Presentation
    block:p
        pres["Driver"]
    end
    api["API/Translate\nEncode/Decode"]
    app <--> pres

    Session
    block:sess
        ses["Connect"]
        sta["Status"]
        thr["(through)"]
    end
    coding["Auto-connect/Transaction\nStatus Events"]

    rdp_t["(Ruida Transport Protocol)"]
    block:rdp
        ruida["Ruida Protocol"]
        block:cod
            enc["Swizzle"]
            dec["Unswizzle"]
        end
    end
    cmd["Command/Response/Execute\nChunking"]
    pres <--> thr
    ses <--> ruida
    sta <--> ruida
    thr <--> ruida

    Transport
    block:trans
        udp["UDP"]
        ser["Serial"]
    end
    Comms
    ruida <--> udp
    ruida <--> ser
    Network/Datalink/Physical
    block:phy
        Ethernet
        USB
    end
    udp <--> Ethernet
    ser <--> USB
```
NOTE: Data paths are shown. For simplicity, control and status paths are not shown.

# Presentation/Driver
This layer is the interface between Rayforge and Ruida controllers and uses the _Session_ and _Ruida Transport Protocol_ (RTP) layers to communicate with a Ruida controller. It receives move (jog) and operation requests from Rayforge, translates them into Ruida commands and forwards them to the Ruida controller using the _RTP_ layer. There are two forms:
- Atomic commands -- typically used for jogging (moves)
- Operation commands and blobs -- for running jobs

This layer is responsible for:
- **Translating** Rayforge API calls into corresponding Ruida commands
- **Relaying** commands and data to and from a Ruida controller using the _Session_ and _RTP_ layers.

This layer conforms to the structure described in: https://rayforge.org/docs/0.22/developer/driver.html

# Session
This layer is responsible for:
- **Connection** using the configured transport protocol (UDP or Serial)
- **Connection monitoring** and signalling
- **Auto-connect and reconnect** to restore a connection or when switching transports.
- **Connection status** change notifications including:
    - Connecting -- initial connect or when reconnecting
    - Connected -- ready to communicate with the machine
    - Connection failure -- a problem with the transport (e.g. could not open port or invalid IP)
- **Machine status** monitoring to detect when the machine is busy and inform upper layers of status changes which include:
    - Hardware version info (e.g. Card ID)
    - Machine bed size
    - Machine busy -- moving or running a job
    - Current head position
- **Decoding** status related data received from the controller.

# Ruida Transport Protocol (RTP)
This layer is responsible for:
- **Swizzle/unswizzle** Ruida controllers require a light obfuscation of data which is handled in this layer so that other layers are not concerned with obfuscated data
- **Checksums** -- for UDP transport, each message sent to the Ruida controller must be preceded by a simple checksum
- **Command/response** handling of sequence and timing of sending commands and receiving replies which can be ACKs or, in the case of memory reads, data
- **Failure notification** to inform the session layer of timeouts or unexpected replies
- **Chunking** -- breaking large jobs into transport compatible chunks for transmission to the machine

# Encoding and Decoding (ruida_translator:RDCTranslator)
Jogging commands, machine status monitoring and job execution require translation to and from Ruida controller commands and data formats. The _Driver_ and _Session_ layers in particular need to encode and decode data. Because multiple layers need this capability, this is a separate module.

The most notable characteristic of Ruida data is that it is transmitted and received in a 7 bit format. Only command bytes, either commands or in replies to commands, have the top bit set. Because of this, data such as integers are transferred as 7 bit values. Large integers are expressed using five bytes making the full range what can be expressed in 35 bits. Two byte values are therefore 14 bits. This module hides the complexity of performing these conversions.

# Development Plan
Development of this driver involves a number of development iterations as shown below. As development progresses the plan will be updated as needed and completed features will be checked off.

## Iterations

1. UDP Connection (first because can use `tshark` to verify)
	- [x] Create `driver/ruida` directory and add to `driver/__init__.py`
	- [ ]  Create `transport/udp.py` and add `udp` to `transport/__init__.py`
	- [ ] Implement `connect` logic and tasks/coroutines (using asyncio)
		- [ ] Implement status monitoring logic and status change events
	- [ ] Test connect/reconnect use cases.
		- [ ] Status monitoring
		- [ ] Connect status updates (send events)
		- [ ] Pull and reconnect cable
	- [ ] Demo video for iteration on streamable.
2. USB Connection
	- [ ] Add `purge` method to `serial.py` (and other transports?) -- needed for resync of comms
	- [ ] Add USB to code from *UDP Connection* iteration.
	- [ ] Test connect/reconnect use cases.
		- [ ] Status monitoring
		- [ ] Connect status updates (send events)
		- [ ] Pull and reconnect cable (may need USB list change to allow UDEV symlinks)
		- [ ] Test switching between UDP and USB.
	- [ ] Demo video for iteration on streamable.
3. Jogging
	- [ ] Rel vrs abs moves.
	- [ ] Rapid moves
	- [ ] Demo video for iteration on streamable.
4. Running Jobs -- Cut and Engrave
	- [ ] Job file structure
		- [ ] Head and tail (with checksum)
		- [ ] Layers
	- [ ] Cut/engrave moves
	- [ ] Speed settings
	- [ ] Power settings
		- [ ] Min/max for corner decel/accel (may require upper layer support)
	- [ ] Assume 20KHz frequency for time being
	- [ ] Job running status updates
	- [ ] Demo video for iteration on streamable.
5. Running Jobs -- Raster and Fill
	- [ ] Horizontal
	- [ ] Vertical
	- [ ] Add dynamic frequency settings
	- [ ] Demo video for iteration on streamable.
6. Running Multi-layer and Multi-step Jobs -- Ready for Beta
7. Human Readable Ruida Commands (for use in side panel)
8. Save to`.rd` File (round trip support)
9. Sync Simulator to Click in Side Panel (like G-Code currently does)
10. Final Signoff and Release

# Credit Where Credit is Due
This work is possible because of the hard work of others.

 - MeerK40t: https://github.com/meerk40t/meerk40t/tree/main/meerk40t/ruida
 - Ruida protocol: https://edutechwiki.unige.ch/en/Ruida

 And, of course, Rayforge itself for providing the framework which greatly simplified the structure of this driver.
