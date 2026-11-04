// Copyright 2026 Patrick Gaskin, 2014 Benoît Amiaux. All rights reserved. Use
// of this source code is governed by a MIT-style license that can be found in
// the LICENSE file.

//go:build amd64 && goexperiment.simd

package arez

import (
	"simd/archsimd"
	"slices"
	"unsafe"
)

var (
	useSIMD   = archsimd.X86.AVX2()
	useAVX512 = archsimd.X86.AVX512()
	useVBMI   = useAVX512 && archsimd.X86.AVX512VBMI()

	_ = setScalers(func() bool { return useSIMD }, prepareKernelSIMD, getHorizontalScalerSIMD, getVerticalScalerSIMD)
)

var (
	simdMax255       = [16]uint16{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
	simdPackBytes    = [32]int8{0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1, 0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1}
	simdPackLanes    = [8]uint32{0, 1, 4, 5, 0, 1, 4, 5}
	simdPackQwords   = [8]uint64{0, 2, 4, 6, 1, 3, 5, 7}
	simdPermB8       = [16]uint32{0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15}
	simdPermB4       = [16]uint32{0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15}
	simdWidenLo      = [32]int8{0, -1, 1, -1, 2, -1, 3, -1, 4, -1, 5, -1, 6, -1, 7, -1, 0, -1, 1, -1, 2, -1, 3, -1, 4, -1, 5, -1, 6, -1, 7, -1}
	simdWidenHi      = [32]int8{8, -1, 9, -1, 10, -1, 11, -1, 12, -1, 13, -1, 14, -1, 15, -1, 8, -1, 9, -1, 10, -1, 11, -1, 12, -1, 13, -1, 14, -1, 15, -1}
	simdUninterleave = [32]int8{0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15}
	simdSwapBytes    = [32]int8{1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14}
	simdShiftEven8   = [8]uint32{8, 0, 8, 0, 8, 0, 8, 0}
	simdShiftEven16  = [16]uint32{8, 0, 8, 0, 8, 0, 8, 0, 8, 0, 8, 0, 8, 0, 8, 0}
	simdRound8       = [8]int32{1 << (bits - 1), 1 << (bits - 1), 1 << (bits - 1), 1 << (bits - 1), 1 << (bits - 1), 1 << (bits - 1), 1 << (bits - 1), 1 << (bits - 1)}
	simdRound16      = [16]int32{1 << (bits - 1), 1 << (bits - 1), 1 << (bits - 1), 1 << (bits - 1), 1 << (bits - 1), 1 << (bits - 1), 1 << (bits - 1), 1 << (bits - 1), 1 << (bits - 1), 1 << (bits - 1), 1 << (bits - 1), 1 << (bits - 1), 1 << (bits - 1), 1 << (bits - 1), 1 << (bits - 1), 1 << (bits - 1)}
	simdWideSeq      = [16]uint32{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30}
	simdWideLanes    = [16]uint32{0, 2, 16, 18, 4, 6, 20, 22, 8, 10, 24, 26, 12, 14, 28, 30}
)

// Coefficient layouts, see prepareKernelSIMD.
const (
	layoutSIMDN = iota + 1
	layoutSIMDShuf
	layoutSIMDPermB
	layoutSIMDV512
)

func getHorizontalScalerSIMD(k *kernel, simd bool) scaler {
	if !simd || !useSIMD {
		return getHorizontalScalerGo(k.size, k.wide)
	}
	switch {
	case k.layout == layoutSIMDShuf && k.wide:
		return h16scaleShufSIMD
	case k.layout == layoutSIMDShuf:
		return h8scaleShufSIMD
	case k.layout == layoutSIMDPermB && k.wide:
		return h16scalePermBSIMD
	case k.layout == layoutSIMDPermB:
		return h8scalePermBSIMD
	case k.wide:
		return h16scaleNSIMD
	}
	return h8scaleNSIMD
}

func getVerticalScalerSIMD(k *kernel, simd bool) scaler {
	if !simd || !useSIMD {
		return getVerticalScalerGo(k.size, k.wide)
	}
	switch {
	case k.layout == layoutSIMDV512 && k.wide:
		return v16scaleNSIMD512
	case k.layout == layoutSIMDV512:
		return v8scaleNSIMD512
	case k.wide:
		return v16scaleNSIMD
	}
	return v8scaleNSIMD
}

// prepareKernelSIMD converts a kernel from the natural layout used by the Go
// scalers to the layout expected by the archsimd scalers.
//
// Vertical kernels hold, for every output row and every pair of taps, one
// 16 x int16 vector holding (c0, c1) 8 times, or a 32 x int16 vector holding
// it 16 times with AVX-512 (layoutSIMDV512), see prepareVerticalCoeffsSIMD.
//
// Horizontal kernels use, in order of preference, the AVX-512 VBMI layout of
// prepareHorizontalCoeffsSIMDPermB with 16, 8 then 4 output bytes per
// vector, the AVX2 layout of prepareHorizontalCoeffsSIMDShuf with 4 then 2
// output bytes per lane, all of which handle packed kernels with pack times
// less work than their unpacked form but need the source bytes of
// neighbouring output bytes to be close enough, and the natural layout
// padded to 16 taps by padKernel, which handles anything.
//
// 16-bit kernels are widened to the bytes of their samples by widenKernel
// and laid out like 8-bit kernels of twice the pack, the scalers differ only
// in the final pack of the sums, see pack32x8Wide.
func prepareKernelSIMD(cfg *ResizerConfig, k kernel) kernel {
	if cfg.Vertical {
		if useAVX512 {
			k.coeffs, k.cofscale = prepareVerticalCoeffsSIMD(k.coeffs, len(k.offsets), k.size, 32)
			k.layout = layoutSIMDV512
		} else {
			k.coeffs, k.cofscale = prepareVerticalCoeffsSIMD(k.coeffs, len(k.offsets), k.size, 16)
		}
		return k
	}
	pack := cfg.Pack
	if k.wide {
		k = widenKernel(k)
		pack *= 2
	}
	if useVBMI {
		for _, outs := range []int{16, 8, 4} {
			if cof, off := prepareHorizontalCoeffsSIMDPermB(k.coeffs, k.offsets, k.size, pack, outs); cof != nil {
				k.coeffs, k.offsets, k.layout = cof, off, layoutSIMDPermB
				return k
			}
		}
	}
	for _, outs := range []int{4, 2} {
		if cof, off := prepareHorizontalCoeffsSIMDShuf(k.coeffs, k.offsets, k.size, pack, 0, outs); cof != nil {
			k.coeffs, k.offsets, k.layout = cof, off, layoutSIMDShuf
			return k
		}
	}
	k = padKernel(k, 16, simdPixels)
	k.layout = layoutSIMDN
	return k
}

// prepareVerticalCoeffsSIMD repeats every pair of taps xwidth/2 times so one
// xwidth x int16 vector holds the coefficients of xwidth/2 pixels.
func prepareVerticalCoeffsSIMD(cof []int16, size, taps, xwidth int) ([]int16, int) {
	dst := make([]int16, size*taps*xwidth>>1)
	si := 0
	di := 0
	for range size {
		for j := 0; j < taps; j += 2 {
			for k := 0; k < xwidth; k += 2 {
				dst[di+k+0] = cof[si+0]
				dst[di+k+1] = cof[si+1]
			}
			si += 2
			di += xwidth
		}
	}
	return dst, xwidth >> 1
}

// simdPixels is the number of output pixels per group of h8scaleNSIMD, its
// offsets and coefficients are padded to a multiple of it.
const simdPixels = 16

// simdStripBytes is the size of the coefficients of a strip of output bytes
// processed for all rows before moving to the next one, see hscale.
var simdStripBytes = 16 << 10

// prepareHorizontalCoeffsSIMDPermB prepares the layout of h8scalePermBSIMD for
// a kernel in the natural layout, unpacked if pack > 1. Output bytes are
// processed in groups of 32, the last group overlapping the previous one, as
// 32/outs vectors of outs output bytes. The source bytes of a vector are read
// from a 64 byte window, which moves by 32/outs*pack bytes per step of 32/outs
// taps. With outs = 16 every output byte owns one dword of the accumulator,
// with 8 and 4 it owns 2 or 4 dwords in the order permBReduce8 and permBReduce4
// expect.
//
// Offsets hold pack, outs then 32/outs window bases per group. Coefficients
// hold per group 32/outs vectors of 32 permutation indices then per step
// 32/outs vectors of 32 coefficients. It returns nil if a vector spans more
// than a window.
func prepareHorizontalCoeffsSIMDPermB(cof []int16, off []int32, taps, pack, outs int) ([]int16, []int32) {
	const group = 32
	size := len(off)
	vectors := group / outs
	m := 32 / outs
	ptaps := taps / pack
	steps := (ptaps + m - 1) / m
	groups := (size + group - 1) / group
	pixel := func(g, v, k int) int {
		x := g * group
		if size >= group {
			x = min(x, size-group)
		}
		return min(x+outs*v+k, size-1)
	}
	pos := func(p int) int { return int(off[p]) + p%pack }
	word := func(k, t int) int {
		switch outs {
		case 16:
			return 2*k + t
		case 8:
			return 8*(k/2) + 2*(k%2) + 4*(t/2) + t%2
		}
		return 8*k + t
	}
	doff := make([]int32, 2+groups*vectors)
	doff[0], doff[1] = int32(pack), int32(outs)
	stride := (steps + 1) * vectors * 32
	dcof := make([]int16, groups*stride)
	for g := range groups {
		d := dcof[g*stride:]
		for v := range vectors {
			b := pos(pixel(g, v, 0))
			for k := range outs {
				b = min(b, pos(pixel(g, v, k)))
			}
			doff[2+g*vectors+v] = int32(b)
			for k := range outs {
				p := pixel(g, v, k)
				rel := pos(p) - b
				if rel+m*pack > 64 {
					return nil, nil
				}
				for t := range m {
					w := v*32 + word(k, t)
					d[w] = int16(rel + t*pack)
					for j := range steps {
						if j*m+t < ptaps {
							d[(1+j)*vectors*32+w] = cof[p*taps+p%pack+(j*m+t)*pack]
						}
					}
				}
			}
		}
	}
	return dcof, doff
}

// prepareHorizontalCoeffsSIMDShuf prepares the layout of h8scaleShufSIMD for a
// kernel in the natural layout, unpacked if pack > 1. Output bytes are
// processed in groups of 8*outs, the last group overlapping the previous one.
// Each group is split in 8 runs of outs bytes, one per 128-bit lane of 4
// vectors: lane l of vector v holds output bytes 4*outs*l+outs*v to
// 4*outs*l+outs*v+outs-1, which is the order pack32x8 expects with outs = 4 and
// pack16 after a VPHADDD of vectors 0, 1 and 2, 3 with outs = 2. The runs read
// their source bytes from 16 byte windows, one per lane of 1, 2 or 4 vectors
// depending on how many vectors can share a window. A window moves by
// 8/outs*pack bytes per step of 8/outs taps and serves up to seg consecutive
// steps, with one shuffle index vector per vector and step within the segment.
//
// Windows are shared by as many vectors as possible, but only when a single
// window serves all steps. share forces the number of vectors per window if not
// 0. outs = 2 always uses one vector per window and is only used when a window
// serves at least 2 steps and there are at most 10 steps, which is where it
// beats h8scaleNSIMD.
//
// Offsets hold the number of windows per lane, seg, the window step, outs and
// then the window bases of each group, two per window. Coefficients hold per
// group seg sets of 4 index vectors then per step 4 coefficient vectors, 16
// int16 each. An index slot is 0x8000 | i so that VPSHUFB loads source byte i
// zero extended to int16. It returns nil if a run spans more than a window.
func prepareHorizontalCoeffsSIMDShuf(cof []int16, off []int32, taps, pack, share, outs int) ([]int16, []int32) {
	group := 8 * outs
	size := len(off)
	m := 8 / outs
	ptaps := taps / pack
	steps := (ptaps + m - 1) / m
	step := m * pack
	groups := (size + group - 1) / group
	pixel := func(g, v, l, k int) int {
		x := g * group
		if size >= group {
			x = min(x, size-group)
		}
		return min(x+l*group/2+v*outs+k, size-1)
	}
	pos := func(p int) int { return int(off[p]) + p%pack }
	// window base and largest relative index for every window of n vectors
	windows := func(n int) ([]int, int) {
		bases := make([]int, 0, groups*8/n)
		span := 0
		for g := range groups {
			for w := 0; w < 4; w += n {
				for l := range 2 {
					b := pos(pixel(g, w, l, 0))
					for v := w; v < w+n; v++ {
						for k := range outs {
							b = min(b, pos(pixel(g, v, l, k)))
						}
					}
					for v := w; v < w+n; v++ {
						for k := range outs {
							span = max(span, pos(pixel(g, v, l, k))-b+(m-1)*pack)
						}
					}
					bases = append(bases, b)
				}
			}
		}
		return bases, span
	}
	var bases []int
	seg := 0
	shares := []int{4, 2, 1}
	if outs == 2 {
		shares = []int{1}
	}
	for _, n := range shares {
		if share != 0 && n != share {
			continue
		}
		b, span := windows(n)
		if span > 15 {
			continue
		}
		sg := min(steps, (15-span)/step+1)
		if n == 1 || sg == steps {
			bases, share, seg = b, n, sg
			break
		}
	}
	if bases == nil || (outs == 2 && (seg < 2 || steps > 10)) {
		return nil, nil
	}
	doff := make([]int32, 4+len(bases))
	doff[0], doff[1], doff[2], doff[3] = int32(4/share), int32(seg), int32(step), int32(outs)
	for i, b := range bases {
		doff[4+i] = int32(b)
	}
	stride := (seg + steps) * 64
	dcof := make([]int16, groups*stride)
	for g := range groups {
		d := dcof[g*stride:]
		for v := range 4 {
			for l := range 2 {
				b := bases[(g*4/share+v/share)*2+l]
				for k := range outs {
					p := pixel(g, v, l, k)
					rel := pos(p) - b
					for t := range m {
						i := v*16 + l*8 + k*m + t
						for r := range seg {
							d[r*64+i] = int16(uint16(0x8000 | (rel + t*pack + r*step)))
						}
						for j := range steps {
							if j*m+t < ptaps {
								d[(seg+j)*64+i] = cof[p*taps+p%pack+(j*m+t)*pack]
							}
						}
					}
				}
			}
		}
	}
	return dcof, doff
}

// h8scalePermBSIMD scales horizontally with any even number of taps and any
// pack, gathering the source bytes of 16, 8 or 4 output bytes per step with one
// zero masking VPERMB from a 64 byte window, see
// prepareHorizontalCoeffsSIMDPermB.
func h8scalePermBSIMD(dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	hscalePermBSIMD[sample8](dst, src, cof, off, taps, width, height, dp, sp)
}

// h16scalePermBSIMD is h8scalePermBSIMD for 16-bit samples, see widenKernel.
func h16scalePermBSIMD(dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	hscalePermBSIMD[sample16](dst, src, cof, off, taps, width, height, dp, sp)
}

func hscalePermBSIMD[W sampleWidth](dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	pack, outs := int(off[0]), int(off[1])
	vectors := 32 / outs
	m := 32 / outs
	steps := (taps/pack + m - 1) / m
	groups := (width + 31) / 32
	off = off[2 : 2+groups*vectors]
	cof = cof[:groups*(steps+1)*vectors*32]
	reach := int(slices.Max(off)) + 64 + (steps-1)*m*pack
	row := permBRow16[W]
	switch outs {
	case 8:
		row = permBRow8[W]
	case 4:
		row = permBRow4[W]
	}
	hscale(dst, src, width, height, dp, sp, reach, 32, off, vectors, cof, (steps+1)*vectors*32, func(d []byte, s, op, cp unsafe.Pointer, x, end int) {
		row(d, s, op, cp, x, end, steps, m*pack)
	})
}

// permBRow16 computes one row of h8scalePermBSIMD with 16 output bytes per
// vector.
func permBRow16[W sampleWidth](d []byte, s, op, cp unsafe.Pointer, x, end, steps, step int) {
	var sw W
	db := unsafe.Pointer(unsafe.SliceData(d))
	words := archsimd.Mask8x64FromBits(0x5555555555555555)
	for {
		pa := unsafe.Add(s, int(at32(op, 0)))
		pb := unsafe.Add(s, int(at32(op, 1)))
		ia := archsimd.LoadUint8x64Array((*[64]uint8)(cp))
		ib := archsimd.LoadUint8x64Array((*[64]uint8)(unsafe.Add(cp, 64)))
		cp = unsafe.Add(cp, 128)
		a := permBMadd(pa, ia, words, cp)
		b := permBMadd(pb, ib, words, unsafe.Add(cp, 64))
		for j := 1; j < steps; j++ {
			pa = unsafe.Add(pa, step)
			pb = unsafe.Add(pb, step)
			cp = unsafe.Add(cp, 128)
			a = a.Add(permBMadd(pa, ia, words, cp))
			b = b.Add(permBMadd(pb, ib, words, unsafe.Add(cp, 64)))
		}
		if len(sw) == 2 {
			storeGroup(d, db, x, pack32x16Wide(a, b, &simdWideSeq))
		} else {
			storeGroup(d, db, x, pack32x16Seq(a, b))
		}
		if x += 32; x >= end {
			return
		}
		cp = unsafe.Add(cp, 128)
		op = unsafe.Add(op, 8)
	}
}

// permBRow8 computes one row of h8scalePermBSIMD with 8 output bytes per
// vector.
func permBRow8[W sampleWidth](d []byte, s, op, cp unsafe.Pointer, x, end, steps, step int) {
	var sw W
	db := unsafe.Pointer(unsafe.SliceData(d))
	words := archsimd.Mask8x64FromBits(0x5555555555555555)
	for {
		pa := unsafe.Add(s, int(at32(op, 0)))
		pb := unsafe.Add(s, int(at32(op, 1)))
		pc := unsafe.Add(s, int(at32(op, 2)))
		pe := unsafe.Add(s, int(at32(op, 3)))
		ia := archsimd.LoadUint8x64Array((*[64]uint8)(cp))
		ib := archsimd.LoadUint8x64Array((*[64]uint8)(unsafe.Add(cp, 64)))
		ic := archsimd.LoadUint8x64Array((*[64]uint8)(unsafe.Add(cp, 128)))
		ie := archsimd.LoadUint8x64Array((*[64]uint8)(unsafe.Add(cp, 192)))
		cp = unsafe.Add(cp, 256)
		a := permBMadd(pa, ia, words, cp)
		b := permBMadd(pb, ib, words, unsafe.Add(cp, 64))
		c := permBMadd(pc, ic, words, unsafe.Add(cp, 128))
		e := permBMadd(pe, ie, words, unsafe.Add(cp, 192))
		for j := 1; j < steps; j++ {
			pa = unsafe.Add(pa, step)
			pb = unsafe.Add(pb, step)
			pc = unsafe.Add(pc, step)
			pe = unsafe.Add(pe, step)
			cp = unsafe.Add(cp, 256)
			a = a.Add(permBMadd(pa, ia, words, cp))
			b = b.Add(permBMadd(pb, ib, words, unsafe.Add(cp, 64)))
			c = c.Add(permBMadd(pc, ic, words, unsafe.Add(cp, 128)))
			e = e.Add(permBMadd(pe, ie, words, unsafe.Add(cp, 192)))
		}
		if len(sw) == 2 {
			storeGroup(d, db, x, pack32x16Wide(permBReduce8(a, b), permBReduce8(c, e), &simdWideSeq))
		} else {
			storeGroup(d, db, x, pack32x16Seq(permBReduce8(a, b), permBReduce8(c, e)))
		}
		if x += 32; x >= end {
			return
		}
		cp = unsafe.Add(cp, 256)
		op = unsafe.Add(op, 16)
	}
}

// permBRow4 computes one row of h8scalePermBSIMD with 4 output bytes per
// vector.
func permBRow4[W sampleWidth](d []byte, s, op, cp unsafe.Pointer, x, end, steps, step int) {
	var sw W
	db := unsafe.Pointer(unsafe.SliceData(d))
	words := archsimd.Mask8x64FromBits(0x5555555555555555)
	for {
		p0 := unsafe.Add(s, int(at32(op, 0)))
		p1 := unsafe.Add(s, int(at32(op, 1)))
		p2 := unsafe.Add(s, int(at32(op, 2)))
		p3 := unsafe.Add(s, int(at32(op, 3)))
		p4 := unsafe.Add(s, int(at32(op, 4)))
		p5 := unsafe.Add(s, int(at32(op, 5)))
		p6 := unsafe.Add(s, int(at32(op, 6)))
		p7 := unsafe.Add(s, int(at32(op, 7)))
		i0 := archsimd.LoadUint8x64Array((*[64]uint8)(cp))
		i1 := archsimd.LoadUint8x64Array((*[64]uint8)(unsafe.Add(cp, 64)))
		i2 := archsimd.LoadUint8x64Array((*[64]uint8)(unsafe.Add(cp, 128)))
		i3 := archsimd.LoadUint8x64Array((*[64]uint8)(unsafe.Add(cp, 192)))
		i4 := archsimd.LoadUint8x64Array((*[64]uint8)(unsafe.Add(cp, 256)))
		i5 := archsimd.LoadUint8x64Array((*[64]uint8)(unsafe.Add(cp, 320)))
		i6 := archsimd.LoadUint8x64Array((*[64]uint8)(unsafe.Add(cp, 384)))
		i7 := archsimd.LoadUint8x64Array((*[64]uint8)(unsafe.Add(cp, 448)))
		cp = unsafe.Add(cp, 512)
		a0 := permBMadd(p0, i0, words, cp)
		a1 := permBMadd(p1, i1, words, unsafe.Add(cp, 64))
		a2 := permBMadd(p2, i2, words, unsafe.Add(cp, 128))
		a3 := permBMadd(p3, i3, words, unsafe.Add(cp, 192))
		a4 := permBMadd(p4, i4, words, unsafe.Add(cp, 256))
		a5 := permBMadd(p5, i5, words, unsafe.Add(cp, 320))
		a6 := permBMadd(p6, i6, words, unsafe.Add(cp, 384))
		a7 := permBMadd(p7, i7, words, unsafe.Add(cp, 448))
		for j := 1; j < steps; j++ {
			cp = unsafe.Add(cp, 512)
			p0 = unsafe.Add(p0, step)
			p1 = unsafe.Add(p1, step)
			p2 = unsafe.Add(p2, step)
			p3 = unsafe.Add(p3, step)
			p4 = unsafe.Add(p4, step)
			p5 = unsafe.Add(p5, step)
			p6 = unsafe.Add(p6, step)
			p7 = unsafe.Add(p7, step)
			a0 = a0.Add(permBMadd(p0, i0, words, cp))
			a1 = a1.Add(permBMadd(p1, i1, words, unsafe.Add(cp, 64)))
			a2 = a2.Add(permBMadd(p2, i2, words, unsafe.Add(cp, 128)))
			a3 = a3.Add(permBMadd(p3, i3, words, unsafe.Add(cp, 192)))
			a4 = a4.Add(permBMadd(p4, i4, words, unsafe.Add(cp, 256)))
			a5 = a5.Add(permBMadd(p5, i5, words, unsafe.Add(cp, 320)))
			a6 = a6.Add(permBMadd(p6, i6, words, unsafe.Add(cp, 384)))
			a7 = a7.Add(permBMadd(p7, i7, words, unsafe.Add(cp, 448)))
		}
		a := permBReduce4(a0, a1, a2, a3)
		b := permBReduce4(a4, a5, a6, a7)
		if len(sw) == 2 {
			storeGroup(d, db, x, pack32x16Wide(a, b, &simdWideSeq))
		} else {
			storeGroup(d, db, x, pack32x16Seq(a, b))
		}
		if x += 32; x >= end {
			return
		}
		cp = unsafe.Add(cp, 512)
		op = unsafe.Add(op, 32)
	}
}

// h8scaleShufSIMD scales horizontally with any even number of taps and any
// pack, gathering the source bytes of 4 or 2 output bytes per 128-bit lane with
// VPSHUFB from 16 byte windows, see prepareHorizontalCoeffsSIMDShuf.
func h8scaleShufSIMD(dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	hscaleShufSIMD[sample8](dst, src, cof, off, taps, width, height, dp, sp)
}

// h16scaleShufSIMD is h8scaleShufSIMD for 16-bit samples, see widenKernel.
func h16scaleShufSIMD(dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	hscaleShufSIMD[sample16](dst, src, cof, off, taps, width, height, dp, sp)
}

func hscaleShufSIMD[W sampleWidth](dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	windows, seg, step, outs := int(off[0]), int(off[1]), int(off[2]), int(off[3])
	group := 8 * outs
	m := 8 / outs
	pack := step / m
	steps := (taps/pack + m - 1) / m
	groups := (width + group - 1) / group
	off = off[4 : 4+groups*windows*2]
	cof = cof[:groups*(seg+steps)*64]
	reach := 0
	for i := 0; i < len(off); i += 2 {
		reach = max(reach, int(off[i])+32, int(off[i+1])+16)
	}
	reach += (steps - 1) / seg * seg * step
	row := shufRow4[W]
	switch {
	case outs == 2:
		row = shufRow16[W]
	case windows == 1:
		row = shufRow1[W]
	case windows == 2:
		row = shufRow2[W]
	}
	hscale(dst, src, width, height, dp, sp, reach, group, off, windows*2, cof, (seg+steps)*64, func(d []byte, s, op, cp unsafe.Pointer, x, end int) {
		row(d, s, op, cp, x, end, steps, seg, step)
	})
}

// shufRow1 computes one row of h8scaleShufSIMD with one window for all vectors.
func shufRow1[W sampleWidth](d []byte, s, op, cp unsafe.Pointer, x, end, pairs, seg, step int) {
	var sw W
	db := unsafe.Pointer(unsafe.SliceData(d))
	for {
		w := loadWindows(s, op, 0)
		kp := unsafe.Add(cp, pairs*128)
		a := shufMadd(w, cp, kp, 0)
		b := shufMadd(w, cp, kp, 1)
		c := shufMadd(w, cp, kp, 2)
		e := shufMadd(w, cp, kp, 3)
		for j := 1; j < pairs; j++ {
			cp = unsafe.Add(cp, 128)
			kp = unsafe.Add(kp, 128)
			a = a.Add(shufMadd(w, cp, kp, 0))
			b = b.Add(shufMadd(w, cp, kp, 1))
			c = c.Add(shufMadd(w, cp, kp, 2))
			e = e.Add(shufMadd(w, cp, kp, 3))
		}
		if len(sw) == 2 {
			storeGroup(d, db, x, pack32x8Wide(a, b, c, e))
		} else {
			storeGroup(d, db, x, pack32x8(a, b, c, e))
		}
		if x += 32; x >= end {
			return
		}
		op = unsafe.Add(op, 8)
		cp = unsafe.Add(kp, 128)
	}
}

// shufRow2 computes one row of h8scaleShufSIMD with one window per two vectors.
func shufRow2[W sampleWidth](d []byte, s, op, cp unsafe.Pointer, x, end, pairs, seg, step int) {
	var sw W
	db := unsafe.Pointer(unsafe.SliceData(d))
	for {
		w0 := loadWindows(s, op, 0)
		w1 := loadWindows(s, op, 1)
		kp := unsafe.Add(cp, pairs*128)
		a := shufMadd(w0, cp, kp, 0)
		b := shufMadd(w0, cp, kp, 1)
		c := shufMadd(w1, cp, kp, 2)
		e := shufMadd(w1, cp, kp, 3)
		for j := 1; j < pairs; j++ {
			cp = unsafe.Add(cp, 128)
			kp = unsafe.Add(kp, 128)
			a = a.Add(shufMadd(w0, cp, kp, 0))
			b = b.Add(shufMadd(w0, cp, kp, 1))
			c = c.Add(shufMadd(w1, cp, kp, 2))
			e = e.Add(shufMadd(w1, cp, kp, 3))
		}
		if len(sw) == 2 {
			storeGroup(d, db, x, pack32x8Wide(a, b, c, e))
		} else {
			storeGroup(d, db, x, pack32x8(a, b, c, e))
		}
		if x += 32; x >= end {
			return
		}
		op = unsafe.Add(op, 16)
		cp = unsafe.Add(kp, 128)
	}
}

// shufRow4 computes one row of h8scaleShufSIMD with one window per vector,
// reloaded every seg pairs.
func shufRow4[W sampleWidth](d []byte, s, op, cp unsafe.Pointer, x, end, pairs, seg, step int) {
	var sw W
	db := unsafe.Pointer(unsafe.SliceData(d))
	for {
		kp := unsafe.Add(cp, seg*128)
		var a, b, c, e archsimd.Int32x8
		so := s
		for j := 0; ; {
			w0 := loadWindows(so, op, 0)
			w1 := loadWindows(so, op, 1)
			w2 := loadWindows(so, op, 2)
			w3 := loadWindows(so, op, 3)
			ip := cp
			for r := min(pairs-j, seg); ; r-- {
				a = a.Add(shufMadd(w0, ip, kp, 0))
				b = b.Add(shufMadd(w1, ip, kp, 1))
				c = c.Add(shufMadd(w2, ip, kp, 2))
				e = e.Add(shufMadd(w3, ip, kp, 3))
				if r == 1 {
					break
				}
				ip = unsafe.Add(ip, 128)
				kp = unsafe.Add(kp, 128)
			}
			if j += seg; j >= pairs {
				break
			}
			kp = unsafe.Add(kp, 128)
			so = unsafe.Add(so, seg*step)
		}
		if len(sw) == 2 {
			storeGroup(d, db, x, pack32x8Wide(a, b, c, e))
		} else {
			storeGroup(d, db, x, pack32x8(a, b, c, e))
		}
		if x += 32; x >= end {
			return
		}
		op = unsafe.Add(op, 32)
		cp = unsafe.Add(kp, 128)
	}
}

// shufRow16 computes one row of h8scaleShufSIMD with 2 output bytes per
// lane, groups of 16 output bytes and one window per vector, reloaded every
// seg steps.
func shufRow16[W sampleWidth](d []byte, s, op, cp unsafe.Pointer, x, end, pairs, seg, step int) {
	var sw W
	db := unsafe.Pointer(unsafe.SliceData(d))
	for {
		kp := unsafe.Add(cp, seg*128)
		var a, b, c, e archsimd.Int32x8
		so := s
		for j := 0; ; {
			w0 := loadWindows(so, op, 0)
			w1 := loadWindows(so, op, 1)
			w2 := loadWindows(so, op, 2)
			w3 := loadWindows(so, op, 3)
			ip := cp
			for r := min(pairs-j, seg); ; r-- {
				a = a.Add(shufMadd(w0, ip, kp, 0))
				b = b.Add(shufMadd(w1, ip, kp, 1))
				c = c.Add(shufMadd(w2, ip, kp, 2))
				e = e.Add(shufMadd(w3, ip, kp, 3))
				if r == 1 {
					break
				}
				ip = unsafe.Add(ip, 128)
				kp = unsafe.Add(kp, 128)
			}
			if j += seg; j >= pairs {
				break
			}
			kp = unsafe.Add(kp, 128)
			so = unsafe.Add(so, seg*step)
		}
		ab, ce := a.ConcatAddPairsGrouped(b), c.ConcatAddPairsGrouped(e)
		var p archsimd.Uint8x16
		if len(sw) == 2 {
			p = pack16Wide(ab, ce)
		} else {
			p = pack16(ab, ce)
		}
		if len(d) >= 16 {
			p.StoreArray((*[16]uint8)(unsafe.Add(db, min(len(d)-16, x))))
		} else {
			p.StorePart(d)
		}
		if x += 16; x >= end {
			return
		}
		op = unsafe.Add(op, 32)
		cp = unsafe.Add(kp, 128)
	}
}

// h8scaleNSIMD scales horizontally with any even number of taps, reading the
// source bytes of each pixel in chunks of 16.
func h8scaleNSIMD(dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	hscaleNSIMD[sample8](dst, src, cof, off, taps, width, height, dp, sp)
}

// h16scaleNSIMD is h8scaleNSIMD for 16-bit samples, see widenKernel.
func h16scaleNSIMD(dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	hscaleNSIMD[sample16](dst, src, cof, off, taps, width, height, dp, sp)
}

func hscaleNSIMD[W sampleWidth](dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	chunk := padSize(taps, 16)
	n := padSize(width, simdPixels)
	off = off[:n]
	cof = cof[:n*chunk]
	reach := int(slices.Max(off)) + chunk
	hscale(dst, src, width, height, dp, sp, reach, simdPixels, off, simdPixels, cof, simdPixels*chunk, func(d []byte, s, op, cp unsafe.Pointer, x, end int) {
		nRow[W](d, s, op, cp, x, end, chunk)
	})
}

// nRow computes one row of h8scaleNSIMD.
func nRow[W sampleWidth](d []byte, s, op, cp unsafe.Pointer, x, end, chunk int) {
	var sw W
	db := unsafe.Pointer(unsafe.SliceData(d))
	for {
		var t [4]archsimd.Int32x8
		for q := 0; ; q++ {
			p0 := unsafe.Add(s, int(at32(op, 0)))
			p1 := unsafe.Add(s, int(at32(op, 1)))
			p2 := unsafe.Add(s, int(at32(op, 2)))
			p3 := unsafe.Add(s, int(at32(op, 3)))
			s0 := madd16(p0, loadCoeffs(cp, 0*chunk/16))
			s1 := madd16(p1, loadCoeffs(cp, 1*chunk/16))
			s2 := madd16(p2, loadCoeffs(cp, 2*chunk/16))
			s3 := madd16(p3, loadCoeffs(cp, 3*chunk/16))
			for i := 16; i < chunk; i += 16 {
				cp = unsafe.Add(cp, 32)
				s0 = s0.Add(madd16(unsafe.Add(p0, i), loadCoeffs(cp, 0*chunk/16)))
				s1 = s1.Add(madd16(unsafe.Add(p1, i), loadCoeffs(cp, 1*chunk/16)))
				s2 = s2.Add(madd16(unsafe.Add(p2, i), loadCoeffs(cp, 2*chunk/16)))
				s3 = s3.Add(madd16(unsafe.Add(p3, i), loadCoeffs(cp, 3*chunk/16)))
			}
			t[q] = hadd4(s0, s1, s2, s3)
			if q == 3 {
				break
			}
			op = unsafe.Add(op, 4*4)
			cp = unsafe.Add(cp, (4*chunk-chunk+16)*2)
		}
		a := t[0].ConcatPermute128Scalars(0, 2, t[2]).Add(t[0].ConcatPermute128Scalars(1, 3, t[2]))
		b := t[1].ConcatPermute128Scalars(0, 2, t[3]).Add(t[1].ConcatPermute128Scalars(1, 3, t[3]))
		var p archsimd.Uint8x16
		if len(sw) == 2 {
			p = pack16Wide(a, b)
		} else {
			p = pack16(a, b)
		}
		if x+simdPixels <= len(d) {
			p.StoreArray((*[16]uint8)(unsafe.Add(db, x)))
		} else {
			p.StorePart(d[x:])
		}
		if x += simdPixels; x >= end {
			return
		}
		op = unsafe.Add(op, 4*4)
		cp = unsafe.Add(cp, (4*chunk-chunk+16)*2)
	}
}

// hscale calls row for every row of a horizontal resize, in vertical strips of
// groups of output bytes so that the coefficients of a strip stay in the L1
// cache. row computes the output bytes from x to end of the row d, x < end,
// from the source s, op and cp pointing at the offsets and coefficients of the
// group starting at x. The layout has offStride offsets and cofStride
// coefficients per group. row may read up to reach bytes from s, rows with
// fewer bytes left in src are copied in a zero padded scratch row first.
func hscale(dst, src []byte, width, height, dp, sp, reach, group int, off []int32, offStride int, cof []int16, cofStride int,
	row func(d []byte, s, op, cp unsafe.Pointer, x, end int)) {
	var scratch []byte
	filled := -1 // the offset in src of the row in scratch
	strip := max(1, simdStripBytes/(cofStride*2)) * group
	op := unsafe.Pointer(&off[0])
	cp := unsafe.Pointer(&cof[0])
	for x := 0; x < width; x += strip {
		end := min(width, x+strip)
		o := unsafe.Add(op, x/group*offStride*4)
		c := unsafe.Add(cp, x/group*cofStride*2)
		di, si := 0, 0
		for range height {
			s := unsafe.Pointer(unsafe.SliceData(src[si:]))
			if len(src)-si < reach {
				if scratch == nil {
					scratch = make([]byte, reach)
				}
				if filled != si {
					clear(scratch[copy(scratch, src[si:]):])
					filled = si
				}
				s = unsafe.Pointer(&scratch[0])
			}
			row(dst[di:di+width], s, o, c, x, end)
			di += dp
			si += sp
		}
	}
}

// loadWindows loads the i'th pair of windows of a shuffle layout group, at the
// offsets op[2i] and op[2i+1] from s, into the two lanes of a vector. It reads
// 32 bytes at the first offset.
func loadWindows(s, op unsafe.Pointer, i int) archsimd.Uint8x32 {
	lo := unsafe.Add(s, int(at32(op, 2*i)))
	hi := unsafe.Add(s, int(at32(op, 2*i+1)))
	return archsimd.LoadUint8x32Array((*[32]uint8)(lo)).SetHi(archsimd.LoadUint8x16Array((*[16]uint8)(hi)))
}

// storeGroup stores the group of 32 output bytes starting at x of the row d, at
// db. Groups overlap the previous one at the end of the row.
func storeGroup(d []byte, db unsafe.Pointer, x int, p archsimd.Uint8x32) {
	if len(d) >= 32 {
		p.StoreArray((*[32]uint8)(unsafe.Add(db, min(len(d)-32, x))))
	} else {
		p.StorePart(d)
	}
}

// permBReduce8 sums the two dwords of each of the 8 output bytes of a and b and
// returns the 16 sums in order: a and b hold output bytes 0-7 and 8-15, output
// 2l+h of a vector owning dwords 4l+h and 4l+h+2.
func permBReduce8(a, b archsimd.Int32x16) archsimd.Int32x16 {
	r := a.InterleaveLoGrouped(b).Add(a.InterleaveHiGrouped(b))
	return r.Permute(archsimd.LoadUint32x16Array(&simdPermB8))
}

// permBReduce4 sums the four dwords of each of the 4 output bytes of a, b, c
// and e, one 128-bit lane per output byte, and returns the 16 sums in order: a
// holds output bytes 0-3, b 4-7, c 8-11 and e 12-15. The second interleave
// works on qword pairs of dwords, the add must stay 32-bit or a carry out of
// the low dword leaks into the high one.
func permBReduce4(a, b, c, e archsimd.Int32x16) archsimd.Int32x16 {
	ab := a.InterleaveLoGrouped(b).Add(a.InterleaveHiGrouped(b)).ToBits().ReshapeToUint64s().BitsToInt64()
	ce := c.InterleaveLoGrouped(e).Add(c.InterleaveHiGrouped(e)).ToBits().ReshapeToUint64s().BitsToInt64()
	r := ab.InterleaveLoGrouped(ce).ToBits().ReshapeToUint32s().BitsToInt32().Add(ab.InterleaveHiGrouped(ce).ToBits().ReshapeToUint32s().BitsToInt32())
	return r.Permute(archsimd.LoadUint32x16Array(&simdPermB4))
}

// permBMadd gathers source pixels from the 64 byte window at p with the indices
// idx, zero extending them to 16 bits, and multiplies them with the
// coefficients at kp.
func permBMadd(p unsafe.Pointer, idx archsimd.Uint8x64, words archsimd.Mask8x64, kp unsafe.Pointer) archsimd.Int32x16 {
	w := archsimd.LoadUint8x64Array((*[64]uint8)(p))
	k := archsimd.LoadInt16x32Array((*[32]int16)(kp))
	return w.Permute(idx).Masked(words).ReshapeToUint16s().BitsToInt16().DotProductPairs(k)
}

// shufMadd gathers the source pixels of the i'th vector of a shuffle layout
// group from the window w with the indices at ip and multiplies them with the
// coefficients at kp.
func shufMadd(w archsimd.Uint8x32, ip, kp unsafe.Pointer, i int) archsimd.Int32x8 {
	idx := archsimd.LoadInt8x32Array((*[32]int8)(unsafe.Add(ip, 32*i)))
	return w.PermuteOrZeroGrouped(idx).ReshapeToUint16s().BitsToInt16().DotProductPairs(loadCoeffs(kp, i))
}

// madd16 multiplies the 16 source bytes at p with the 16 coefficients in k and
// returns 8 partial sums.
func madd16(p unsafe.Pointer, k archsimd.Int16x16) archsimd.Int32x8 {
	return archsimd.LoadUint8x16Array((*[16]uint8)(p)).ExtendToUint16().BitsToInt16().DotProductPairs(k)
}

// hadd4 reduces the partial sums of r0..r3 to [r0 r1 r2 r3] in each lane.
func hadd4(r0, r1, r2, r3 archsimd.Int32x8) archsimd.Int32x8 {
	return r0.ConcatAddPairsGrouped(r1).ConcatAddPairsGrouped(r2.ConcatAddPairsGrouped(r3))
}

// v8scaleNSIMD scales vertically with any even number of taps, 32 pixels at a
// time, see vrow256.
func v8scaleNSIMD(dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	vscale(dst, src, cof, off, taps, width, height, dp, sp, 16, vrow256[sample8])
}

// v16scaleNSIMD is v8scaleNSIMD for 16-bit samples.
func v16scaleNSIMD(dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	vscale(dst, src, cof, off, taps, width, height, dp, sp, 16, vrow256[sample16])
}

// v8scaleNSIMD512 scales vertically with any even number of taps, 64 pixels at
// a time, see vrow512.
func v8scaleNSIMD512(dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	vscale(dst, src, cof, off, taps, width, height, dp, sp, 32, vrow512[sample8])
}

// v16scaleNSIMD512 is v8scaleNSIMD512 for 16-bit samples.
func v16scaleNSIMD512(dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	vscale(dst, src, cof, off, taps, width, height, dp, sp, 32, vrow512[sample16])
}

// vscale calls row for every row of a vertical resize with xwidth coefficients
// per pair of taps.
func vscale(dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp, xwidth int,
	row func(d []byte, s unsafe.Pointer, sp int, c []int16, pairs int)) {
	pairs := taps >> 1
	n := pairs * xwidth
	cof = cof[:height*n]
	di := 0
	for _, yoff := range off[:height] {
		src = src[sp*int(yoff):]
		rows := src[:sp*(taps-1)+width]
		row(dst[di:di+width], unsafe.Pointer(&rows[0]), sp, cof[:n], pairs)
		cof = cof[n:]
		di += dp
	}
}

// vrow256 computes one row of a vertical resize with AVX2. It widens 32 pixels
// of a source row with two VPSHUFB, to pixels 0-7 and 16-23 then 8-15 and
// 24-31, and interleaves the rows of each pair of taps, which yields the order
// of pack32x8. Rows narrower than 32 pixels are read and written partially, the
// last block of wider rows overlaps the previous one.
func vrow256[W sampleWidth](d []byte, s unsafe.Pointer, sp int, c []int16, pairs int) {
	var sw W
	const block = 32
	width := len(d)
	if width < block {
		vrow256Part[W](d, s, sp, c, pairs)
		return
	}
	db := unsafe.Pointer(&d[0])
	cb := unsafe.Pointer(&c[0])
	lo := archsimd.LoadInt8x32Array(&simdWidenLo)
	hi := archsimd.LoadInt8x32Array(&simdWidenHi)
	for x := 0; x < width; x += block {
		x = min(width-block, x)
		p := unsafe.Add(s, x)
		k := loadCoeffs(cb, 0)
		a, b, e, f := vmadd32(p, sp, lo, hi, k)
		cp := cb
		for j := 1; j < pairs; j++ {
			p = unsafe.Add(p, 2*sp)
			cp = unsafe.Add(cp, 32)
			a1, b1, e1, f1 := vmadd32(p, sp, lo, hi, loadCoeffs(cp, 0))
			a, b, e, f = a.Add(a1), b.Add(b1), e.Add(e1), f.Add(f1)
		}
		if len(sw) == 2 {
			pack32x8Wide(a, b, e, f).StoreArray((*[block]uint8)(unsafe.Add(db, x)))
		} else {
			pack32x8(a, b, e, f).StoreArray((*[block]uint8)(unsafe.Add(db, x)))
		}
	}
}

// vrow256Part is vrow256 for rows narrower than 32 pixels.
func vrow256Part[W sampleWidth](d []byte, s unsafe.Pointer, sp int, c []int16, pairs int) {
	var sw W
	width := len(d)
	lo := archsimd.LoadInt8x32Array(&simdWidenLo)
	hi := archsimd.LoadInt8x32Array(&simdWidenHi)
	var a, b, e, f archsimd.Int32x8
	for j := range pairs {
		r0, _ := archsimd.LoadUint8x32Part(unsafe.Slice((*byte)(unsafe.Add(s, 2*j*sp)), width))
		r1, _ := archsimd.LoadUint8x32Part(unsafe.Slice((*byte)(unsafe.Add(s, (2*j+1)*sp)), width))
		a1, b1, e1, f1 := vmadd32x(r0, r1, lo, hi, archsimd.LoadInt16x16(c[j*16:]))
		a, b, e, f = a.Add(a1), b.Add(b1), e.Add(e1), f.Add(f1)
	}
	if len(sw) == 2 {
		pack32x8Wide(a, b, e, f).StorePart(d)
	} else {
		pack32x8(a, b, e, f).StorePart(d)
	}
}

// vrow512 computes one row of a vertical resize with AVX-512, 64 pixels at a
// time then 32. Rows narrower than 32 pixels are read and written partially,
// the last block of wider rows overlaps the previous one.
func vrow512[W sampleWidth](d []byte, s unsafe.Pointer, sp int, c []int16, pairs int) {
	var sw W
	width := len(d)
	if width < 32 {
		vrow512Part[W](d, s, sp, c, pairs)
		return
	}
	db := unsafe.Pointer(&d[0])
	cb := unsafe.Pointer(&c[0])
	x := 0
	for ; x+64 <= width; x += 64 {
		p := unsafe.Add(s, x)
		k := archsimd.LoadInt16x32Array((*[32]int16)(cb))
		a, b := vmadd32z(p, sp, k)
		e, f := vmadd32z(unsafe.Add(p, 32), sp, k)
		cp := cb
		for j := 1; j < pairs; j++ {
			p = unsafe.Add(p, 2*sp)
			cp = unsafe.Add(cp, 64)
			k := archsimd.LoadInt16x32Array((*[32]int16)(cp))
			a1, b1 := vmadd32z(p, sp, k)
			e1, f1 := vmadd32z(unsafe.Add(p, 32), sp, k)
			a, b, e, f = a.Add(a1), b.Add(b1), e.Add(e1), f.Add(f1)
		}
		if len(sw) == 2 {
			pack32x16Wide(a, b, &simdWideLanes).StoreArray((*[32]uint8)(unsafe.Add(db, x)))
			pack32x16Wide(e, f, &simdWideLanes).StoreArray((*[32]uint8)(unsafe.Add(db, x+32)))
		} else {
			pack32x16(a, b).StoreArray((*[32]uint8)(unsafe.Add(db, x)))
			pack32x16(e, f).StoreArray((*[32]uint8)(unsafe.Add(db, x+32)))
		}
	}
	for ; x < width; x += 32 {
		x = min(width-32, x)
		p := unsafe.Add(s, x)
		a, b := vmadd32z(p, sp, archsimd.LoadInt16x32Array((*[32]int16)(cb)))
		cp := cb
		for j := 1; j < pairs; j++ {
			p = unsafe.Add(p, 2*sp)
			cp = unsafe.Add(cp, 64)
			a1, b1 := vmadd32z(p, sp, archsimd.LoadInt16x32Array((*[32]int16)(cp)))
			a, b = a.Add(a1), b.Add(b1)
		}
		if len(sw) == 2 {
			pack32x16Wide(a, b, &simdWideLanes).StoreArray((*[32]uint8)(unsafe.Add(db, x)))
		} else {
			pack32x16(a, b).StoreArray((*[32]uint8)(unsafe.Add(db, x)))
		}
	}
}

// vrow512Part is vrow512 for rows narrower than 32 pixels.
func vrow512Part[W sampleWidth](d []byte, s unsafe.Pointer, sp int, c []int16, pairs int) {
	var sw W
	width := len(d)
	var a, b archsimd.Int32x16
	for j := range pairs {
		r0, _ := archsimd.LoadUint8x32Part(unsafe.Slice((*byte)(unsafe.Add(s, 2*j*sp)), width))
		r1, _ := archsimd.LoadUint8x32Part(unsafe.Slice((*byte)(unsafe.Add(s, (2*j+1)*sp)), width))
		w0, w1 := r0.ExtendToUint16(), r1.ExtendToUint16()
		k := archsimd.LoadInt16x32(c[j*32:])
		a = a.Add(w0.InterleaveLoGrouped(w1).BitsToInt16().DotProductPairs(k))
		b = b.Add(w0.InterleaveHiGrouped(w1).BitsToInt16().DotProductPairs(k))
	}
	if len(sw) == 2 {
		pack32x16Wide(a, b, &simdWideLanes).StorePart(d)
	} else {
		pack32x16(a, b).StorePart(d)
	}
}

// vmadd32 multiplies 32 pixels of the rows at p and p+sp with the pair of
// coefficients in k, see vrow256.
func vmadd32(p unsafe.Pointer, sp int, lo, hi archsimd.Int8x32, k archsimd.Int16x16) (a, b, e, f archsimd.Int32x8) {
	r0 := archsimd.LoadUint8x32Array((*[32]uint8)(p))
	r1 := archsimd.LoadUint8x32Array((*[32]uint8)(unsafe.Add(p, sp)))
	return vmadd32x(r0, r1, lo, hi, k)
}

// vmadd32x is vmadd32 for rows already loaded.
func vmadd32x(r0, r1 archsimd.Uint8x32, lo, hi archsimd.Int8x32, k archsimd.Int16x16) (a, b, e, f archsimd.Int32x8) {
	l0 := r0.PermuteOrZeroGrouped(lo).ReshapeToUint16s().BitsToInt16()
	l1 := r1.PermuteOrZeroGrouped(lo).ReshapeToUint16s().BitsToInt16()
	h0 := r0.PermuteOrZeroGrouped(hi).ReshapeToUint16s().BitsToInt16()
	h1 := r1.PermuteOrZeroGrouped(hi).ReshapeToUint16s().BitsToInt16()
	a = l0.InterleaveLoGrouped(l1).DotProductPairs(k)
	b = l0.InterleaveHiGrouped(l1).DotProductPairs(k)
	e = h0.InterleaveLoGrouped(h1).DotProductPairs(k)
	f = h0.InterleaveHiGrouped(h1).DotProductPairs(k)
	return
}

// vmadd32z multiplies 32 pixels of the rows at p and p+sp with the pair of
// coefficients in k, in the order of pack32x16.
func vmadd32z(p unsafe.Pointer, sp int, k archsimd.Int16x32) (a, b archsimd.Int32x16) {
	r0 := archsimd.LoadUint8x32Array((*[32]uint8)(p)).ExtendToUint16()
	r1 := archsimd.LoadUint8x32Array((*[32]uint8)(unsafe.Add(p, sp))).ExtendToUint16()
	a = r0.InterleaveLoGrouped(r1).BitsToInt16().DotProductPairs(k)
	b = r0.InterleaveHiGrouped(r1).BitsToInt16().DotProductPairs(k)
	return
}

// loadCoeffs loads the i'th vector of 16 coefficients at p.
func loadCoeffs(p unsafe.Pointer, i int) archsimd.Int16x16 {
	return archsimd.LoadInt16x16Array((*[16]int16)(unsafe.Add(p, 32*i)))
}

// The pack functions round, shift and clamp pixel sums to bytes. They use (s +
// 1<<13) >> 14 == ((s >> 13) + 1) >> 1, which lets VPAVGW against zero do the
// final rounding shift on 16-bit values after VPACKUSDW: the clamp of s >> 13
// to [0, 65535] never changes the result clamped to [0, 255].

// pack16 packs 16 pixel sums in the order of VPACKUSDW: a holds pixels 0-3
// and 8-11, b holds pixels 4-7 and 12-15.
func pack16(a, b archsimd.Int32x8) archsimd.Uint8x16 {
	var zero archsimd.Uint16x16
	w := a.ShiftAllRight(bits - 1).SaturateToUint16ConcatGrouped(b.ShiftAllRight(bits - 1)).Average(zero)
	if useAVX512 {
		return w.SaturateToUint8()
	}
	w = w.Min(archsimd.LoadUint16x16Array(&simdMax255))
	p := w.ReshapeToUint8s().BitsToInt8().PermuteOrZeroGrouped(archsimd.LoadInt8x32Array(&simdPackBytes))
	return p.ToBits().ReshapeToUint32s().Permute(archsimd.LoadUint32x8Array(&simdPackLanes)).ReshapeToUint8s().GetLo()
}

// pack32x8 packs 32 pixel sums with AVX2. The 128-bit lane l of the result
// holds lane l of a, b, c and e in this order.
func pack32x8(a, b, c, e archsimd.Int32x8) archsimd.Uint8x32 {
	var zero archsimd.Uint16x16
	max255 := archsimd.LoadUint16x16Array(&simdMax255)
	lo := a.ShiftAllRight(bits - 1).SaturateToUint16ConcatGrouped(b.ShiftAllRight(bits - 1)).Average(zero).Min(max255)
	hi := c.ShiftAllRight(bits - 1).SaturateToUint16ConcatGrouped(e.ShiftAllRight(bits - 1)).Average(zero).Min(max255)
	w := lo.Or(hi.ShiftAllLeft(8)).ReshapeToUint8s()
	return w.PermuteOrZeroGrouped(archsimd.LoadInt8x32Array(&simdUninterleave))
}

// pack32x16 packs 32 pixel sums in the order of VPACKUSDW: 128-bit lane l of a
// then b.
func pack32x16(a, b archsimd.Int32x16) archsimd.Uint8x32 {
	var zero archsimd.Uint16x32
	return a.ShiftAllRight(bits - 1).SaturateToUint16ConcatGrouped(b.ShiftAllRight(bits - 1)).Average(zero).SaturateToUint8()
}

// pack32x16Seq packs 32 pixel sums, pixels 0-15 in a and 16-31 in b.
func pack32x16Seq(a, b archsimd.Int32x16) archsimd.Uint8x32 {
	var zero archsimd.Uint16x32
	w := a.ShiftAllRight(bits - 1).SaturateToUint16ConcatGrouped(b.ShiftAllRight(bits - 1))
	w = w.ReshapeToUint64s().Permute(archsimd.LoadUint64x8Array(&simdPackQwords)).ReshapeToUint16s()
	return w.Average(zero).SaturateToUint8()
}

// The wide pack functions take the byte sums of 16-bit samples, the high byte
// of a sample in an even lane and its low byte in the following odd lane, and
// pack them to big endian u16: (hi << 8) + lo is the sample sum, rounded,
// shifted by bits and clamped to [0, 65535] by VPACKUSDW.

// pack16Wide packs the byte sums of 8 samples in the order of pack16.
func pack16Wide(a, b archsimd.Int32x8) archsimd.Uint8x16 {
	sh := archsimd.LoadUint32x8Array(&simdShiftEven8)
	s := a.ShiftLeft(sh).ConcatAddPairsGrouped(b.ShiftLeft(sh)).Add(archsimd.LoadInt32x8Array(&simdRound8)).ShiftAllRight(bits)
	w := s.SaturateToUint16ConcatGrouped(s).ReshapeToUint32s().Permute(archsimd.LoadUint32x8Array(&simdPackLanes))
	return w.ReshapeToUint8s().PermuteOrZeroGrouped(archsimd.LoadInt8x32Array(&simdSwapBytes)).GetLo()
}

// pack32x8Wide packs the byte sums of 16 samples in the order of pack32x8.
func pack32x8Wide(a, b, c, e archsimd.Int32x8) archsimd.Uint8x32 {
	sh := archsimd.LoadUint32x8Array(&simdShiftEven8)
	r := archsimd.LoadInt32x8Array(&simdRound8)
	lo := a.ShiftLeft(sh).ConcatAddPairsGrouped(b.ShiftLeft(sh)).Add(r).ShiftAllRight(bits)
	hi := c.ShiftLeft(sh).ConcatAddPairsGrouped(e.ShiftLeft(sh)).Add(r).ShiftAllRight(bits)
	return lo.SaturateToUint16ConcatGrouped(hi).ReshapeToUint8s().PermuteOrZeroGrouped(archsimd.LoadInt8x32Array(&simdSwapBytes))
}

// pack32x16Wide packs the byte sums of 16 samples held in a and b, idx being
// the even lanes of the two vectors in sample order: simdWideSeq for the order
// of pack32x16Seq, simdWideLanes for the order of pack32x16.
func pack32x16Wide(a, b archsimd.Int32x16, idx *[16]uint32) archsimd.Uint8x32 {
	var zero archsimd.Int32x16
	sh := archsimd.LoadUint32x16Array(&simdShiftEven16)
	qa := a.ShiftLeft(sh).ToBits().ReshapeToUint64s().BitsToInt64()
	qb := b.ShiftLeft(sh).ToBits().ReshapeToUint64s().BitsToInt64()
	qa = qa.Add(qa.ShiftAllRight(32))
	qb = qb.Add(qb.ShiftAllRight(32))
	s := qa.ToBits().ReshapeToUint32s().BitsToInt32().ConcatPermute(qb.ToBits().ReshapeToUint32s().BitsToInt32(), archsimd.LoadUint32x16Array(idx))
	s = s.Add(archsimd.LoadInt32x16Array(&simdRound16)).ShiftAllRight(bits).Max(zero)
	w := s.ToBits().SaturateToUint16()
	return w.ReshapeToUint8s().PermuteOrZeroGrouped(archsimd.LoadInt8x32Array(&simdSwapBytes))
}
