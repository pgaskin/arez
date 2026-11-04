// Copyright 2026 Patrick Gaskin, 2014 Benoît Amiaux. All rights reserved. Use
// of this source code is governed by a MIT-style license that can be found in
// the LICENSE file.

//go:build arm64 && goexperiment.simd

package arez

import (
	"simd/archsimd"
	"unsafe"
)

var (
	neonShift     = [4]int32{-bits, -bits, -bits, -bits}
	neonShiftEven = [4]int32{8, 0, 8, 0}
	neonRound     = [4]int32{1 << (bits - 1), 1 << (bits - 1), 1 << (bits - 1), 1 << (bits - 1)}
	neonSwapBytes = [16]uint8{1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14}
)

// packConsts holds the vector constants pack16 and pack16Wide need, loaded once
// per scaler call by loadPackConsts.
type packConsts struct {
	shift archsimd.Int32x4
}

type packWideConsts struct {
	shl8  archsimd.Int32x4
	round archsimd.Int32x4
	swap  archsimd.Uint8x16
}

type reduceConsts struct{}

func loadPackConsts(wide bool) packConsts {
	return packConsts{archsimd.LoadInt32x4Array(&neonShift)}
}

func loadPackWideConsts() packWideConsts {
	return packWideConsts{
		archsimd.LoadInt32x4Array(&neonShiftEven),
		archsimd.LoadInt32x4Array(&neonRound),
		archsimd.LoadUint8x16Array(&neonSwapBytes),
	}
}

func loadReduceConsts() reduceConsts {
	return reduceConsts{}
}

// lookup16 gathers the bytes of t at the indices in i, zero for indices out of
// range.
func lookup16(t, i archsimd.Uint8x16) archsimd.Uint8x16 {
	return t.LookupOrZero(i)
}

// maddLookup multiplies the source bytes of two slots of 8 pixels in v with the
// coefficients after the indices at c and accumulates the sums of pixels 0-3 in
// a and 4-7 in b.
func maddLookup(v archsimd.Uint8x16, c unsafe.Pointer, a, b archsimd.Int32x4) (archsimd.Int32x4, archsimd.Int32x4) {
	return madd2(v, loadCoeffs8(c, 1), loadCoeffs8(c, 2), a, b)
}

// madd2 multiplies the low 8 bytes of v with k0 and the high 8 with k1 and
// accumulates the products of bytes 0-3 of each half in a and of bytes 4-7 in
// b.
func madd2(v archsimd.Uint8x16, k0, k1 archsimd.Int16x8, a, b archsimd.Int32x4) (archsimd.Int32x4, archsimd.Int32x4) {
	lo := v.ExtendLo8ToUint16().BitsToInt16()
	hi := v.HiToLo().ExtendLo8ToUint16().BitsToInt16()
	a = a.Add(lo.MulWidenLo(k0).Add(hi.MulWidenLo(k1)))
	b = b.Add(lo.HiToLo().MulWidenLo(k0.HiToLo()).Add(hi.HiToLo().MulWidenLo(k1.HiToLo())))
	return a, b
}

// madd16 multiplies the 16 source bytes at p with the 16 coefficients at c and
// adds the 4 partial sums to acc.
func madd16(p, c unsafe.Pointer, acc archsimd.Int32x4) archsimd.Int32x4 {
	v := archsimd.LoadUint8x16Array((*[16]uint8)(p))
	k0 := archsimd.LoadInt16x8Array((*[8]int16)(c))
	k1 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(c, 16)))
	lo := v.ExtendLo8ToUint16().BitsToInt16()
	hi := v.HiToLo().ExtendLo8ToUint16().BitsToInt16()
	s := lo.MulWidenLo(k0).Add(lo.HiToLo().MulWidenLo(k0.HiToLo()))
	t := hi.MulWidenLo(k1).Add(hi.HiToLo().MulWidenLo(k1.HiToLo()))
	return acc.Add(s.Add(t))
}

// madd8 multiplies the 8 source bytes at p with the 8 coefficients at c and
// adds the 4 partial sums to acc. It reads 16 source bytes.
func madd8(p, c unsafe.Pointer, acc archsimd.Int32x4) archsimd.Int32x4 {
	v := archsimd.LoadUint8x16Array((*[16]uint8)(p))
	k := archsimd.LoadInt16x8Array((*[8]int16)(c))
	lo := v.ExtendLo8ToUint16().BitsToInt16()
	return acc.Add(lo.MulWidenLo(k).Add(lo.HiToLo().MulWidenLo(k.HiToLo())))
}

// mul16 multiplies 16 source bytes with the 8 coefficients in k, which are
// expected to be all equal, and returns the 16 products.
func mul16(v archsimd.Uint8x16, k archsimd.Int16x8) (archsimd.Int32x4, archsimd.Int32x4, archsimd.Int32x4, archsimd.Int32x4) {
	lo := v.ExtendLo8ToUint16().BitsToInt16()
	hi := v.HiToLo().ExtendLo8ToUint16().BitsToInt16()
	kh := k.HiToLo()
	return lo.MulWidenLo(k), lo.HiToLo().MulWidenLo(kh), hi.MulWidenLo(k), hi.HiToLo().MulWidenLo(kh)
}

// reduce4 sums the 4 lanes of each of s0..s3 into the 4 lanes of the result, an
// ADDP tree.
func reduce4(s0, s1, s2, s3 archsimd.Int32x4, _ reduceConsts) archsimd.Int32x4 {
	return s0.ConcatAddPairs(s1).ConcatAddPairs(s2.ConcatAddPairs(s3))
}

// pack16 shifts and clamps 16 rounded pixel sums in natural order to bytes.
func pack16(a, b, c, d archsimd.Int32x4, pc packConsts) archsimd.Uint8x16 {
	shift := pc.shift
	lo := concatLo(a.Shift(shift).SaturateToInt16().ToBits().ReshapeToUint64s(), b.Shift(shift).SaturateToInt16().ToBits().ReshapeToUint64s())
	hi := concatLo(c.Shift(shift).SaturateToInt16().ToBits().ReshapeToUint64s(), d.Shift(shift).SaturateToInt16().ToBits().ReshapeToUint64s())
	l := lo.ReshapeToUint16s().BitsToInt16().SaturateToUint8().ReshapeToUint64s()
	h := hi.ReshapeToUint16s().BitsToInt16().SaturateToUint8().ReshapeToUint64s()
	return concatLo(l, h).ReshapeToUint8s()
}

// pack16Wide packs the byte sums of 8 big endian 16-bit samples, the high byte
// of a sample in an even lane and its low byte in the odd lane after it: (hi <<
// 8) + lo is the sample sum, rounded here, shifted and clamped by SQXTUN, then
// the bytes are swapped.
func pack16Wide(a, b, c, d archsimd.Int32x4, pc packConsts, pw packWideConsts) archsimd.Uint8x16 {
	ab := a.Shift(pw.shl8).ConcatAddPairs(b.Shift(pw.shl8)).Add(pw.round).Shift(pc.shift)
	cd := c.Shift(pw.shl8).ConcatAddPairs(d.Shift(pw.shl8)).Add(pw.round).Shift(pc.shift)
	w := concatLo(ab.SaturateToUint16().ReshapeToUint64s(), cd.SaturateToUint16().ReshapeToUint64s())
	return w.ReshapeToUint8s().LookupOrZero(pw.swap)
}

// concatLo returns the low 64 bits of a followed by the low 64 bits of b. It is
// written so that the narrowing operation producing b becomes its high half
// variant (SQXTN2, SQXTUN2) writing into a.
func concatLo(a, b archsimd.Uint64x2) archsimd.Uint64x2 {
	return a.BitsToFloat64().SetElem(1, b.BitsToFloat64().GetElem(0)).ToBits()
}
