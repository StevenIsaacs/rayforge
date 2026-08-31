# Upstream prompt: distinguish full-swing from dithered in Ops section params

**Target**: raygeo maintainers
(source repo https://github.com/barebaric/raygeo)

## Context

The rayforge ruidarpa driver treats the GlueScript transcript as the
source of truth and encodes Ops raster sections into cut and power
actions. For a CONSTANT_POWER raster section the encoder must never
turn a scan line into a move: off-pixels legitimately carry 0.0 power,
yet the scan line must still cut. For a dithered image, by contrast,
0-power pixels may become moves. The encoder currently cannot tell the
two apart.

## Problem

Full-swing raster (DepthMode.CONSTANT_POWER) and dithered raster
(DepthMode.DITHER) are both assembled by the same `Ops::from_mask_scan`
function and both surface as `RasterMode::ConstantPower` in
`ops.section_params`. The depth-mode distinction is lost by the time
the encoder sees the Ops, so the RuidaRPAEncoder cannot tell a
full-swing raster (where 0-power off-pixels should still be cuts) from
a dithered image (where 0-power pixels may become moves).

## Requested change

Add a full-swing flag (or the originating depth mode) to the Ops
section params, for example by extending `ops.section_params` or the
section metadata, so downstream consumers can distinguish full swing
(CONSTANT_POWER depth mode) from dithered (DITHER depth mode).

## Note

The raygeo checkout is not currently on disk in this repo, so this is
documented for when the checkout is available.
