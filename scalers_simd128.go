// Copyright 2026 Patrick Gaskin, 2014 Benoît Amiaux. All rights reserved. Use
// of this source code is governed by a MIT-style license that can be found in
// the LICENSE file.

//go:build goexperiment.simd && (arm64 || wasm)

package arez

import (
	"encoding/binary"
	"math"
	"simd/archsimd"
	"slices"
	"unsafe"
)

// The 128-bit scalers shared by arm64 (NEON) and wasm (SIMD128).
//
// This file contains the kernels and their coefficient layouts, using shared
// arch-specific helpers in the corresponding file for cases where archsimd is
// different.
//
// The kernels are stenciled per sample width (sampleWidth).
//
// 16-bit kernels run as 8-bit ones on the sample bytes (widenKernel), but pack
// the sums differently (pack16Wide), with the rounding added by the pack
// instead of the accumulators.

// simd128Pixels is the number of output pixels processed per iteration by the
// 128-bit scalers. Horizontal offsets and coefficients are padded to a multiple
// of it, see prepareKernelSIMD128.
const simd128Pixels = 16

var _ = setScalers(func() bool { return true }, prepareKernelSIMD128, getHorizontalScalerSIMD128, getVerticalScalerSIMD128)

var (
	simd128Round     = [4]int32{1 << (bits - 1), 1 << (bits - 1), 1 << (bits - 1), 1 << (bits - 1)}
	simd128RoundLane = [4]int32{1 << (bits - 1), 0, 0, 0}
	simd128TableHi   = [16]uint8{16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16}
)

// Coefficient layouts, see prepareKernelSIMD128.
const (
	layoutSIMD128N = iota + 1
	layoutSIMD128Tbl16
	layoutSIMD128Tbl32
	layoutSIMD128P4
	layoutSIMD128P8
)

func getHorizontalScalerSIMD128(k *kernel, simd bool) scaler {
	if !simd {
		return getHorizontalScalerGo(k.size, k.wide)
	}
	switch {
	case k.layout == layoutSIMD128Tbl16 && k.wide:
		return h16scaleTbl16SIMD128
	case k.layout == layoutSIMD128Tbl16:
		return h8scaleTbl16SIMD128
	case k.layout == layoutSIMD128Tbl32 && k.wide:
		return h16scaleTbl32SIMD128
	case k.layout == layoutSIMD128Tbl32:
		return h8scaleTbl32SIMD128
	case k.layout == layoutSIMD128P4:
		return h8scaleP4SIMD128
	case k.layout == layoutSIMD128P8 && k.wide:
		return h16scaleP8SIMD128
	case k.layout == layoutSIMD128P8:
		return h8scaleP8SIMD128
	case k.wide:
		return h16scaleNSIMD128
	}
	return h8scaleNSIMD128
}

func getVerticalScalerSIMD128(k *kernel, simd bool) scaler {
	if !simd {
		return getVerticalScalerGo(k.size, k.wide)
	}
	if k.wide {
		return v16scaleNSIMD128
	}
	return v8scaleNSIMD128
}

// prepareKernelSIMD128 converts a kernel from the natural layout used by the Go
// scalers to the layout expected by the 128-bit scalers.
//
// Vertical kernels hold, for every output row and every tap, the coefficient
// repeated 8 times.
//
// Horizontal kernels use the table lookup layout of
// prepareHorizontalCoeffsSIMD128Tbl when every group of 8 pixels reads at most
// 16 source bytes, then for 4 and 8 bytes per pixel the layout of
// prepareHorizontalCoeffsSIMD128Packed, then the table lookup layout again when
// groups read at most 32 bytes, and otherwise the natural layout padded to 8
// taps by padKernel. 16-bit kernels are widened to the bytes of their samples
// by widenKernel and laid out like 8-bit kernels of twice the pack: RGBA64 is 8
// bytes per pixel.
func prepareKernelSIMD128(cfg *ResizerConfig, k kernel) kernel {
	if cfg.Vertical {
		k.coeffs, k.cofscale = prepareVerticalCoeffsSIMD128(k.coeffs, len(k.offsets), k.size)
		return k
	}
	pack := cfg.Pack
	if k.wide {
		k = widenKernel(k)
		pack *= 2
	}
	input := cfg.Input * pack
	if cof, off := prepareHorizontalCoeffsSIMD128Tbl(k.coeffs, k.offsets, k.size, input, 16); cof != nil {
		k.coeffs, k.offsets, k.layout = cof, off, layoutSIMD128Tbl16
		return k
	}
	if pack == 4 && !k.wide {
		if cof, off := prepareHorizontalCoeffsSIMD128Packed(k.coeffs, k.offsets, k.size, 4); cof != nil {
			k.coeffs, k.offsets, k.layout = cof, off, layoutSIMD128P4
			return k
		}
	}
	if pack == 8 {
		if cof, off := prepareHorizontalCoeffsSIMD128Packed(k.coeffs, k.offsets, k.size, 8); cof != nil {
			k.coeffs, k.offsets, k.layout = cof, off, layoutSIMD128P8
			return k
		}
	}
	if cof, off := prepareHorizontalCoeffsSIMD128Tbl(k.coeffs, k.offsets, k.size, input, 32); cof != nil {
		k.coeffs, k.offsets, k.layout = cof, off, layoutSIMD128Tbl32
		return k
	}
	k = padKernel(k, 8, simd128Pixels)
	k.layout = layoutSIMD128N
	return k
}

// prepareVerticalCoeffsSIMD128 repeats every coefficient 8 times so one Int16x8
// holds the coefficient of one tap for 8 pixels.
func prepareVerticalCoeffsSIMD128(cof []int16, size, taps int) ([]int16, int) {
	const xwidth = 8
	dst := make([]int16, size*taps*xwidth)
	for i, c := range cof[:size*taps] {
		for k := range xwidth {
			dst[i*xwidth+k] = c
		}
	}
	return dst, xwidth
}

// prepareHorizontalCoeffsSIMD128Packed prepares the layout of h8scaleP4SIMD128
// (pack 4) and h8scaleP8SIMD128 (pack 8) from a kernel unpacked with that pack,
// where the pack bytes of a pixel share their offset and their coefficients,
// byte c using tap t of the packed kernel as tap pack*t+c. It returns one
// offset per pixel, padded to a multiple of 16/pack pixels, and per pixel and
// chunk of 16/pack packed taps the coefficient of each tap repeated pack times,
// so that a chunk of 16 coefficients multiplies the 16 source bytes of 16/pack
// taps. It returns nil when the kernel is not shaped like that.
func prepareHorizontalCoeffsSIMD128Packed(cof []int16, off []int32, taps, pack int) ([]int16, []int32) {
	if taps%pack != 0 || len(off)%pack != 0 {
		return nil, nil
	}
	ptaps := taps / pack
	chunks := padSize(ptaps, 16/pack) / (16 / pack)
	size := len(off) / pack
	n := padSize(size, simd128Pixels/pack)
	dst := make([]int16, n*chunks*16)
	poff := make([]int32, n)
	for i := range size {
		row := cof[i*pack*taps : (i+1)*pack*taps]
		for c := range pack {
			if off[i*pack+c] != off[i*pack] {
				return nil, nil
			}
			for t, v := range row[c*taps : (c+1)*taps] {
				if (t%pack == c && v != row[t-c]) || (t%pack != c && v != 0) {
					return nil, nil
				}
			}
		}
	}
	for i := range n {
		q := min(i, size-1)
		poff[i] = off[q*pack]
		d := dst[i*chunks*16:]
		for t := range ptaps {
			for c := range pack {
				d[t*pack+c] = cof[q*pack*taps+t*pack]
			}
		}
	}
	return dst, poff
}

// prepareHorizontalCoeffsSIMD128Tbl prepares the layout of h8scaleTbl16SIMD128
// and h8scaleTbl32SIMD128, which load the tbl source bytes read by a group of 8
// pixels once and gather the bytes of each tap with table lookups.
//
// Taps with a zero coefficient are dropped, the remaining ones are packed in
// slots, as many as the largest number of non zero taps of a pixel rounded up
// to even. This removes the zeros added by unpack for Pack > 1 kernels.
//
// It returns one base offset per group and, for every group and pair of slots,
// 16 table indices (slot 2j of pixels 0-7 then slot 2j+1) stored as 8 int16,
// followed by the 16 matching coefficients. It returns nil when a group reads
// more than tbl bytes.
func prepareHorizontalCoeffsSIMD128Tbl(cof []int16, off []int32, taps, input, tbl int) ([]int16, []int32) {
	if input < tbl {
		return nil, nil
	}
	size := len(off)
	slots := 2
	for p := range size {
		n := 0
		for _, c := range cof[p*taps : p*taps+taps] {
			if c != 0 {
				n++
			}
		}
		slots = max(slots, n)
	}
	pairs := padSize(slots, 2) / 2
	groups := padSize(size, simd128Pixels) / 8
	dst := make([]int16, groups*pairs*24)
	idx := unsafe.Slice((*byte)(unsafe.Pointer(&dst[0])), len(dst)*2)
	base := make([]int32, groups)
	for g := range groups {
		lo := math.MaxInt
		for k := range 8 {
			p := min(g*8+k, size-1)
			for t, c := range cof[p*taps : p*taps+taps] {
				if c != 0 {
					lo = min(lo, int(off[p])+t)
				}
			}
		}
		if lo == math.MaxInt {
			lo = int(off[min(g*8, size-1)])
		}
		b := min(lo, input-tbl)
		base[g] = int32(b)
		for k := range 8 {
			p := min(g*8+k, size-1)
			s := 0
			for t, c := range cof[p*taps : p*taps+taps] {
				if c == 0 {
					continue
				}
				i := int(off[p]) + t - b
				if i >= tbl {
					return nil, nil
				}
				j := (g*pairs + s/2) * 24
				idx[j*2+(s&1)*8+k] = byte(i)
				dst[j+8+(s&1)*8+k] = c
				s++
			}
		}
	}
	return dst, base
}

// h8scaleTbl16SIMD128 scales horizontally when every group of 8 pixels reads at
// most 16 source bytes: it loads them once per group, and each pair of slots is
// one table lookup and 4 widening multiplies for 8 pixels.
func h8scaleTbl16SIMD128(dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	hscaleTbl16SIMD128[sample8](dst, src, cof, off, taps, width, height, dp, sp)
}

// h16scaleTbl16SIMD128 is h8scaleTbl16SIMD128 for 16-bit samples, see
// widenKernel.
func h16scaleTbl16SIMD128(dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	hscaleTbl16SIMD128[sample16](dst, src, cof, off, taps, width, height, dp, sp)
}

func hscaleTbl16SIMD128[W sampleWidth](dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	var sw W
	pairs := len(cof) / (len(off) * 24)
	groups := padSize(width, simd128Pixels) / 8
	off = off[:groups]
	cof = cof[:groups*pairs*24]
	reach := int(slices.Max(off)) + 16
	round, pc, pw := loadConsts128(len(sw) == 2)
	di, si := 0, 0
	for range height {
		row := src[si : si+reach]
		base := unsafe.Pointer(&row[0])
		d := dst[di : di+width]
		op := unsafe.Pointer(&off[0])
		cp := unsafe.Pointer(&cof[0])
		for x := 0; ; {
			t0 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(base, int(at32(op, 0)))))
			t1 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(base, int(at32(op, 1)))))
			a0, a1, b0, b1 := round, round, round, round
			c0 := cp
			c1 := unsafe.Add(cp, pairs*48)
			for r := pairs; ; r-- {
				a0, a1 = maddLookup(lookup16(t0, loadIndices(c0)), c0, a0, a1)
				b0, b1 = maddLookup(lookup16(t1, loadIndices(c1)), c1, b0, b1)
				if r == 1 {
					break
				}
				c0 = unsafe.Add(c0, 48)
				c1 = unsafe.Add(c1, 48)
			}
			if len(sw) == 2 {
				store16(d, x, pack16Wide(a0, a1, b0, b1, pc, pw))
			} else {
				store16(d, x, pack16(a0, a1, b0, b1, pc))
			}
			if x += simd128Pixels; x >= width {
				break
			}
			op = unsafe.Add(op, 8)
			cp = unsafe.Add(c1, 48)
		}
		di += dp
		si += sp
	}
}

// h8scaleTbl32SIMD128 is h8scaleTbl16SIMD128 for groups reading at most 32
// source bytes, each lookup is done in both 16 byte halves.
func h8scaleTbl32SIMD128(dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	hscaleTbl32SIMD128[sample8](dst, src, cof, off, taps, width, height, dp, sp)
}

// h16scaleTbl32SIMD128 is h8scaleTbl32SIMD128 for 16-bit samples, see
// widenKernel.
func h16scaleTbl32SIMD128(dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	hscaleTbl32SIMD128[sample16](dst, src, cof, off, taps, width, height, dp, sp)
}

func hscaleTbl32SIMD128[W sampleWidth](dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	var sw W
	pairs := len(cof) / (len(off) * 24)
	groups := padSize(width, simd128Pixels) / 8
	off = off[:groups]
	cof = cof[:groups*pairs*24]
	reach := int(slices.Max(off)) + 32
	round, pc, pw := loadConsts128(len(sw) == 2)
	hi := archsimd.LoadUint8x16Array(&simd128TableHi)
	di, si := 0, 0
	for range height {
		row := src[si : si+reach]
		base := unsafe.Pointer(&row[0])
		d := dst[di : di+width]
		op := unsafe.Pointer(&off[0])
		cp := unsafe.Pointer(&cof[0])
		for x := 0; ; {
			p0 := unsafe.Add(base, int(at32(op, 0)))
			p1 := unsafe.Add(base, int(at32(op, 1)))
			t0 := archsimd.LoadUint8x16Array((*[16]uint8)(p0))
			u0 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p0, 16)))
			t1 := archsimd.LoadUint8x16Array((*[16]uint8)(p1))
			u1 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p1, 16)))
			a0, a1, b0, b1 := round, round, round, round
			c0 := cp
			c1 := unsafe.Add(cp, pairs*48)
			for r := pairs; ; r-- {
				i0 := loadIndices(c0)
				i1 := loadIndices(c1)
				v0 := lookup16(t0, i0).Or(lookup16(u0, i0.Xor(hi)))
				v1 := lookup16(t1, i1).Or(lookup16(u1, i1.Xor(hi)))
				a0, a1 = maddLookup(v0, c0, a0, a1)
				b0, b1 = maddLookup(v1, c1, b0, b1)
				if r == 1 {
					break
				}
				c0 = unsafe.Add(c0, 48)
				c1 = unsafe.Add(c1, 48)
			}
			if len(sw) == 2 {
				store16(d, x, pack16Wide(a0, a1, b0, b1, pc, pw))
			} else {
				store16(d, x, pack16(a0, a1, b0, b1, pc))
			}
			if x += simd128Pixels; x >= width {
				break
			}
			op = unsafe.Add(op, 8)
			cp = unsafe.Add(c1, 48)
		}
		di += dp
		si += sp
	}
}

// h8scaleP4SIMD128 scales 4 channel pixels horizontally: one 16 byte load holds
// 4 taps of the 4 channels, which multiplied by coefficients repeated for each
// channel sum to the 4 channels of one output pixel.
func h8scaleP4SIMD128(dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	chunks := len(cof) / (len(off) * 16)
	n := padSize(width, simd128Pixels) / 4
	off = off[:n]
	cof = cof[:n*chunks*16]
	reach := int(slices.Max(off)) + chunks*16
	stride := chunks * 32
	round := archsimd.LoadInt32x4Array(&simd128Round)
	pc := loadPackConsts(false)
	var scratch []byte
	di, si := 0, 0
	for range height {
		row := src[si:]
		if len(row) < reach {
			if scratch == nil {
				scratch = make([]byte, reach)
			}
			clear(scratch[copy(scratch, row):])
			row = scratch
		}
		base := unsafe.Pointer(&row[0])
		d := dst[di : di+width]
		op := unsafe.Pointer(&off[0])
		cp := unsafe.Pointer(&cof[0])
		for x := 0; ; {
			p0 := unsafe.Add(base, int(at32(op, 0)))
			p1 := unsafe.Add(base, int(at32(op, 1)))
			p2 := unsafe.Add(base, int(at32(op, 2)))
			p3 := unsafe.Add(base, int(at32(op, 3)))
			a0, a1, a2, a3 := round, round, round, round
			for r := chunks; ; r-- {
				a0 = madd16(p0, cp, a0)
				a1 = madd16(p1, unsafe.Add(cp, stride), a1)
				a2 = madd16(p2, unsafe.Add(cp, 2*stride), a2)
				a3 = madd16(p3, unsafe.Add(cp, 3*stride), a3)
				if r == 1 {
					break
				}
				p0 = unsafe.Add(p0, 16)
				p1 = unsafe.Add(p1, 16)
				p2 = unsafe.Add(p2, 16)
				p3 = unsafe.Add(p3, 16)
				cp = unsafe.Add(cp, 32)
			}
			store16(d, x, pack16(a0, a1, a2, a3, pc))
			if x += simd128Pixels; x >= width {
				break
			}
			op = unsafe.Add(op, 4*4)
			cp = unsafe.Add(cp, 3*stride+32)
		}
		di += dp
		si += sp
	}
}

// h8scaleP8SIMD128 scales 8 byte pixels (RGBA64, see widenKernel) horizontally:
// one 16 byte load holds 2 taps of the 8 bytes, which multiplied by the
// coefficients of the two taps repeated 8 times sum to the 8 bytes of one
// output pixel in two accumulators.
func h8scaleP8SIMD128(dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	hscaleP8SIMD128[sample8](dst, src, cof, off, taps, width, height, dp, sp)
}

// h16scaleP8SIMD128 is h8scaleP8SIMD128 for 16-bit samples.
func h16scaleP8SIMD128(dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	hscaleP8SIMD128[sample16](dst, src, cof, off, taps, width, height, dp, sp)
}

func hscaleP8SIMD128[W sampleWidth](dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	var sw W
	chunks := len(cof) / (len(off) * 16)
	n := padSize(width, simd128Pixels) / 8
	off = off[:n]
	cof = cof[:n*chunks*16]
	reach := int(slices.Max(off)) + chunks*16
	stride := chunks * 32
	round, pc, pw := loadConsts128(len(sw) == 2)
	var scratch []byte
	di, si := 0, 0
	for range height {
		row := src[si:]
		if len(row) < reach {
			if scratch == nil {
				scratch = make([]byte, reach)
			}
			clear(scratch[copy(scratch, row):])
			row = scratch
		}
		base := unsafe.Pointer(&row[0])
		d := dst[di : di+width]
		op := unsafe.Pointer(&off[0])
		cp := unsafe.Pointer(&cof[0])
		for x := 0; ; {
			p0 := unsafe.Add(base, int(at32(op, 0)))
			p1 := unsafe.Add(base, int(at32(op, 1)))
			a0, a1, b0, b1 := round, round, round, round
			c1 := unsafe.Add(cp, stride)
			for r := chunks; ; r-- {
				a0, a1 = madd2(archsimd.LoadUint8x16Array((*[16]uint8)(p0)), loadCoeffs8(cp, 0), loadCoeffs8(cp, 1), a0, a1)
				b0, b1 = madd2(archsimd.LoadUint8x16Array((*[16]uint8)(p1)), loadCoeffs8(c1, 0), loadCoeffs8(c1, 1), b0, b1)
				if r == 1 {
					break
				}
				p0 = unsafe.Add(p0, 16)
				p1 = unsafe.Add(p1, 16)
				cp = unsafe.Add(cp, 32)
				c1 = unsafe.Add(c1, 32)
			}
			if len(sw) == 2 {
				store16(d, x, pack16Wide(a0, a1, b0, b1, pc, pw))
			} else {
				store16(d, x, pack16(a0, a1, b0, b1, pc))
			}
			if x += simd128Pixels; x >= width {
				break
			}
			op = unsafe.Add(op, 2*4)
			cp = unsafe.Add(c1, 32)
		}
		di += dp
		si += sp
	}
}

// h8scaleNSIMD128 scales horizontally with any even number of taps, 4 pixels at
// a time, reading the source bytes of each pixel in chunks of 16 and 8.
func h8scaleNSIMD128(dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	hscaleNSIMD128[sample8](dst, src, cof, off, taps, width, height, dp, sp)
}

// h16scaleNSIMD128 is h8scaleNSIMD128 for 16-bit samples, see widenKernel.
func h16scaleNSIMD128(dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	hscaleNSIMD128[sample16](dst, src, cof, off, taps, width, height, dp, sp)
}

func hscaleNSIMD128[W sampleWidth](dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	var sw W
	chunk := padSize(taps, 8)
	n := padSize(width, simd128Pixels)
	off = off[:n]
	cof = cof[:n*chunk]
	reach := int(slices.Max(off)) + padSize(chunk, 16)
	round, pc, pw := loadConsts128(len(sw) == 2)
	if len(sw) != 2 {
		// the reduction sums all lanes, round once
		round = archsimd.LoadInt32x4Array(&simd128RoundLane)
	}
	rc := loadReduceConsts()
	var scratch []byte
	di, si := 0, 0
	for range height {
		row := src[si:]
		if len(row) < reach {
			// the last row may not have enough bytes after the last
			// pixel to read whole chunks, so copy it in a zero padded buffer
			if scratch == nil {
				scratch = make([]byte, reach)
			}
			clear(scratch[copy(scratch, row):])
			row = scratch
		}
		base := unsafe.Pointer(&row[0])
		d := dst[di : di+width]
		op := unsafe.Pointer(&off[0])
		cp := unsafe.Pointer(&cof[0])
		for x := 0; ; {
			var t [4]archsimd.Int32x4
			for q := 0; ; q++ {
				p0 := unsafe.Add(base, int(at32(op, 0)))
				p1 := unsafe.Add(base, int(at32(op, 1)))
				p2 := unsafe.Add(base, int(at32(op, 2)))
				p3 := unsafe.Add(base, int(at32(op, 3)))
				c0 := cp
				c1 := unsafe.Add(cp, chunk*2)
				c2 := unsafe.Add(cp, chunk*4)
				c3 := unsafe.Add(cp, chunk*6)
				s0, s1, s2, s3 := round, round, round, round
				i := 0
				for ; i+16 <= chunk; i += 16 {
					s0 = madd16(unsafe.Add(p0, i), unsafe.Add(c0, i*2), s0)
					s1 = madd16(unsafe.Add(p1, i), unsafe.Add(c1, i*2), s1)
					s2 = madd16(unsafe.Add(p2, i), unsafe.Add(c2, i*2), s2)
					s3 = madd16(unsafe.Add(p3, i), unsafe.Add(c3, i*2), s3)
				}
				if i < chunk {
					s0 = madd8(unsafe.Add(p0, i), unsafe.Add(c0, i*2), s0)
					s1 = madd8(unsafe.Add(p1, i), unsafe.Add(c1, i*2), s1)
					s2 = madd8(unsafe.Add(p2, i), unsafe.Add(c2, i*2), s2)
					s3 = madd8(unsafe.Add(p3, i), unsafe.Add(c3, i*2), s3)
				}
				t[q] = reduce4(s0, s1, s2, s3, rc)
				if q == 3 {
					break
				}
				op = unsafe.Add(op, 4*4)
				cp = unsafe.Add(cp, 4*chunk*2)
			}
			if len(sw) == 2 {
				store16(d, x, pack16Wide(t[0], t[1], t[2], t[3], pc, pw))
			} else {
				store16(d, x, pack16(t[0], t[1], t[2], t[3], pc))
			}
			if x += simd128Pixels; x >= n {
				break
			}
			op = unsafe.Add(op, 4*4)
			cp = unsafe.Add(cp, 4*chunk*2)
		}
		di += dp
		si += sp
	}
}

// v8scaleNSIMD128 scales vertically with any number of taps, 32 then 16 pixels
// at a time. A last partial block is handled by stepping back to the last 16
// pixels of the row.
func v8scaleNSIMD128(dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	vscaleNSIMD128[sample8](dst, src, cof, off, taps, width, height, dp, sp)
}

// v16scaleNSIMD128 is v8scaleNSIMD128 for 16-bit samples.
func v16scaleNSIMD128(dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	vscaleNSIMD128[sample16](dst, src, cof, off, taps, width, height, dp, sp)
}

func vscaleNSIMD128[W sampleWidth](dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	const xwidth = 8
	var sw W
	if width < simd128Pixels {
		vscaleNarrowSIMD128[W](dst, src, cof, off, taps, width, height, dp, sp)
		return
	}
	round, pc, pw := loadConsts128(len(sw) == 2)
	di := 0
	for _, yoff := range off[:height] {
		src = src[sp*int(yoff):]
		rows := src[:sp*(taps-1)+width]
		d := dst[di : di+width]
		c := cof[:taps*xwidth]
		sb := unsafe.Pointer(&rows[0])
		db := unsafe.Pointer(&d[0])
		cb := unsafe.Pointer(&c[0])
		for x := 0; x < width; {
			if x+2*simd128Pixels <= width {
				a0, a1, a2, a3 := round, round, round, round
				b0, b1, b2, b3 := round, round, round, round
				p := unsafe.Add(sb, x)
				cp := cb
				for r := taps; ; r-- {
					k := archsimd.LoadInt16x8Array((*[8]int16)(cp))
					m0, m1, m2, m3 := mul16(archsimd.LoadUint8x16Array((*[16]uint8)(p)), k)
					n0, n1, n2, n3 := mul16(archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, simd128Pixels))), k)
					a0, a1, a2, a3 = a0.Add(m0), a1.Add(m1), a2.Add(m2), a3.Add(m3)
					b0, b1, b2, b3 = b0.Add(n0), b1.Add(n1), b2.Add(n2), b3.Add(n3)
					if r == 1 {
						break
					}
					p = unsafe.Add(p, sp)
					cp = unsafe.Add(cp, xwidth*2)
				}
				if len(sw) == 2 {
					pack16Wide(a0, a1, a2, a3, pc, pw).StoreArray((*[16]uint8)(unsafe.Add(db, x)))
					pack16Wide(b0, b1, b2, b3, pc, pw).StoreArray((*[16]uint8)(unsafe.Add(db, x+simd128Pixels)))
				} else {
					pack16(a0, a1, a2, a3, pc).StoreArray((*[16]uint8)(unsafe.Add(db, x)))
					pack16(b0, b1, b2, b3, pc).StoreArray((*[16]uint8)(unsafe.Add(db, x+simd128Pixels)))
				}
				x += 2 * simd128Pixels
				continue
			}
			x = min(width-simd128Pixels, x)
			a0, a1, a2, a3 := round, round, round, round
			p := unsafe.Add(sb, x)
			cp := cb
			for r := taps; ; r-- {
				k := archsimd.LoadInt16x8Array((*[8]int16)(cp))
				m0, m1, m2, m3 := mul16(archsimd.LoadUint8x16Array((*[16]uint8)(p)), k)
				a0, a1, a2, a3 = a0.Add(m0), a1.Add(m1), a2.Add(m2), a3.Add(m3)
				if r == 1 {
					break
				}
				p = unsafe.Add(p, sp)
				cp = unsafe.Add(cp, xwidth*2)
			}
			if len(sw) == 2 {
				pack16Wide(a0, a1, a2, a3, pc, pw).StoreArray((*[16]uint8)(unsafe.Add(db, x)))
			} else {
				pack16(a0, a1, a2, a3, pc).StoreArray((*[16]uint8)(unsafe.Add(db, x)))
			}
			x += simd128Pixels
		}
		cof = cof[taps*xwidth:]
		di += dp
	}
}

// vscaleNarrowSIMD128 is vscaleNSIMD128 in Go for rows narrower than 16 bytes.
func vscaleNarrowSIMD128[W sampleWidth](dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	const xwidth = 8
	var sw W
	di := 0
	for _, yoff := range off[:height] {
		src = src[sp*int(yoff):]
		for x := 0; x < width; x += len(sw) {
			pix := 0
			for j := range taps {
				v := int(src[sp*j+x])
				if len(sw) == 2 {
					v = int(binary.BigEndian.Uint16(src[sp*j+x:]))
				}
				pix += int(cof[j*xwidth]) * v
			}
			if len(sw) == 2 {
				binary.BigEndian.PutUint16(dst[di+x:], u16((pix+1<<(bits-1))>>bits))
			} else {
				dst[di+x] = u8((pix + 1<<(bits-1)) >> bits)
			}
		}
		cof = cof[taps*xwidth:]
		di += dp
	}
}

// loadConsts128 loads the accumulator initial value and the pack constants of a
// kernel: the rounding constant and the pack16 constants, or zero and the
// pack16Wide constants (which add the rounding) when wide.
func loadConsts128(wide bool) (archsimd.Int32x4, packConsts, packWideConsts) {
	var round archsimd.Int32x4
	var pw packWideConsts
	if wide {
		pw = loadPackWideConsts()
	} else {
		round = archsimd.LoadInt32x4Array(&simd128Round)
	}
	return round, loadPackConsts(wide), pw
}

// loadIndices loads the 16 table indices at p.
func loadIndices(p unsafe.Pointer) archsimd.Uint8x16 {
	return archsimd.LoadUint8x16Array((*[16]uint8)(p))
}

// loadCoeffs8 loads the i'th vector of 8 coefficients at p.
func loadCoeffs8(p unsafe.Pointer, i int) archsimd.Int16x8 {
	return archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(p, 16*i)))
}

// store16 stores the 16 pixels in v at d[x:], or as many as fit.
func store16(d []byte, x int, v archsimd.Uint8x16) {
	if x+simd128Pixels <= len(d) {
		v.StoreArray((*[16]uint8)(d[x : x+simd128Pixels]))
		return
	}
	var tmp [simd128Pixels]uint8
	v.StoreArray(&tmp)
	copy(d[x:], tmp[:])
}
