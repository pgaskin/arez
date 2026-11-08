// Copyright 2026 Patrick Gaskin, 2013 Benoît Amiaux. All rights reserved. Use
// of this source code is governed by a MIT-style license that can be found in
// the LICENSE file.

/*
Package arez resizes images, using SIMD where available.

It is based on github.com/bamiaux/rez, but rewritten to use simd/archsimd,
with amd64/arm64/wasm support, more image formats, and a modernized API.

As of Go 1.27, SIMD requires GOEXPERIMENT=simd. If SIMD is not available or
disabled, regular Go scalers are used, but the output is still bit-identical.

Features:

  - Supports YCbCr, Gray, RGBA, and NRGBA images.
  - Supports YCbCr subsample ratio conversions.
  - Supports interlaced resizes.
  - Supports parallel resizing.
  - Supports SIMD on amd64.

New features:

  - Modernized code.
  - Proper error types.
  - Requires Go 1.27 or later.
  - Slightly faster on amd64.
  - Supports SIMD on arm64 and wasm.
  - Supports nearest neighbour, box, and fractional lanczos filters.
  - Supports faster approximate large downscales (like Pillow's [reducing_gap]).
  - Supports Alpha, CMYK, NYCbCrA, Gray16, Alpha16, RGBA64, and NRGBA64 images.
  - Supports premultiplying alpha for NRGBA, NRGBA64, and NYCbCrA.
  - Supports rows larger than 32767 bytes (i.e., 8192 pixels RGBA).
  - Supports concurrent use of a Converter.
  - All outputs are bit-identical to a fallback Go implementation, which itself
    is almost identical to the original one.
  - Fixed many buffer overflows and out-of-bound reads.
  - Fixed many crashes with weirder image layouts.
  - Added fuzzers and tests against simpler (but slower) versions.

Paletted images and other image.Image implementations are not supported and need
to be converted first (e.g., with image/draw).

Channels are filtered independently, which is correct for premultiplied
alpha (RGBA, RGBA64), but may bleed transparent colors into visible pixels for
images with straight alpha (NRGBA, NRGBA64, NYCbCrA).

By default, images are resized in parallel with GOMAXPROCS slices.

Input planes need at least 2 pixels in each direction (2 rows per field when
interlaced), and rows are limited to 2^31-1 bytes (NewConverter returns an
error otherwise).

I used Claude Fable 5.1 to help port and verify parts of the SIMD code (it's a
lot of tedious and mostly mechanical work), write the tests, and help find/test
micro-optimizations, but the rest is mostly done by hand.

[reducing_gap]: https://pillow.readthedocs.io/en/latest/reference/Image.html#PIL.Image.Image.resize
*/
package arez
