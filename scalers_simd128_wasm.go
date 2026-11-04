// Copyright 2026 Patrick Gaskin, 2014 Benoît Amiaux. All rights reserved. Use
// of this source code is governed by a MIT-style license that can be found in
// the LICENSE file.

//go:build wasm && goexperiment.simd

package arez

import (
	"simd/archsimd"
	"unsafe"
)

// Wasm has the widening multiplies (extmul) and the table lookup
// (i8x16.swizzle) of NEON but no pairwise add and no narrowing, so the
// reduction of 4 pixel sums and the pack to bytes use bitselect, shifts and
// swizzles instead.

var (
	wasmMax255     = [4]int32{255, 255, 255, 255}
	wasmPackBytes  = [16]int8{0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15}
	wasmSwapHalves = [16]int8{8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7}
	wasmSwapPairs  = [16]int8{4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11}
	wasmLanes01    = [4]int32{-1, -1, 0, 0}
	wasmLanes02    = [4]int32{-1, 0, -1, 0}
	wasmMul256     = [4]int32{256, 1, 256, 1}
	wasmMax65535   = [4]int32{65535, 65535, 65535, 65535}
	wasmRound      = [4]int32{1 << (bits - 1), 1 << (bits - 1), 1 << (bits - 1), 1 << (bits - 1)}
	wasmWideBytes  = [16]int8{1, 0, 9, 8, 5, 4, 13, 12, 3, 2, 11, 10, 7, 6, 15, 14}
)

// packConsts holds the vector constants pack16 or pack16Wide need, loaded once
// per scaler call by loadPackConsts: the clamp and the byte order of the sample
// width in use.
type packConsts struct {
	max archsimd.Int32x4
	idx archsimd.Int8x16
}

// packWideConsts holds the vector constants pack16Wide needs besides
// packConsts.
type packWideConsts struct {
	mul     archsimd.Int32x4
	round   archsimd.Int32x4
	lanes02 archsimd.Int32x4
}

type reduceConsts struct {
	halves  archsimd.Int8x16
	pairs   archsimd.Int8x16
	lanes01 archsimd.Int32x4
	lanes02 archsimd.Int32x4
}

func loadPackConsts(wide bool) packConsts {
	if wide {
		return packConsts{archsimd.LoadInt32x4Array(&wasmMax65535), archsimd.LoadInt8x16Array(&wasmWideBytes)}
	}
	return packConsts{archsimd.LoadInt32x4Array(&wasmMax255), archsimd.LoadInt8x16Array(&wasmPackBytes)}
}

func loadPackWideConsts() packWideConsts {
	return packWideConsts{
		archsimd.LoadInt32x4Array(&wasmMul256),
		archsimd.LoadInt32x4Array(&wasmRound),
		archsimd.LoadInt32x4Array(&wasmLanes02),
	}
}

func loadReduceConsts() reduceConsts {
	return reduceConsts{
		archsimd.LoadInt8x16Array(&wasmSwapHalves),
		archsimd.LoadInt8x16Array(&wasmSwapPairs),
		archsimd.LoadInt32x4Array(&wasmLanes01),
		archsimd.LoadInt32x4Array(&wasmLanes02),
	}
}

// lookup16 gathers the bytes of t at the indices in i, zero for indices out of
// range.
func lookup16(t, i archsimd.Uint8x16) archsimd.Uint8x16 {
	return t.BitsToInt8().LookupOrZero(i.BitsToInt8()).ToBits()
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
	hi := v.ExtendHi8ToUint16().BitsToInt16()
	a = a.Add(lo.MulWidenLo(k0).Add(hi.MulWidenLo(k1)))
	b = b.Add(lo.MulWidenHi(k0).Add(hi.MulWidenHi(k1)))
	return a, b
}

// madd16 multiplies the 16 source bytes at p with the 16 coefficients at c and
// adds the 4 partial sums to acc.
func madd16(p, c unsafe.Pointer, acc archsimd.Int32x4) archsimd.Int32x4 {
	v := archsimd.LoadUint8x16Array((*[16]uint8)(p))
	k0 := archsimd.LoadInt16x8Array((*[8]int16)(c))
	k1 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(c, 16)))
	lo := v.ExtendLo8ToUint16().BitsToInt16()
	hi := v.ExtendHi8ToUint16().BitsToInt16()
	s := lo.MulWidenLo(k0).Add(lo.MulWidenHi(k0))
	t := hi.MulWidenLo(k1).Add(hi.MulWidenHi(k1))
	return acc.Add(s.Add(t))
}

// madd8 multiplies the 8 source bytes at p with the 8 coefficients at c and
// adds the 4 partial sums to acc. It reads 16 source bytes.
func madd8(p, c unsafe.Pointer, acc archsimd.Int32x4) archsimd.Int32x4 {
	v := archsimd.LoadUint8x16Array((*[16]uint8)(p))
	k := archsimd.LoadInt16x8Array((*[8]int16)(c))
	lo := v.ExtendLo8ToUint16().BitsToInt16()
	return acc.Add(lo.MulWidenLo(k).Add(lo.MulWidenHi(k)))
}

// mul16 multiplies 16 source bytes with the 8 coefficients in k, which are
// expected to be all equal, and returns the 16 products.
func mul16(v archsimd.Uint8x16, k archsimd.Int16x8) (archsimd.Int32x4, archsimd.Int32x4, archsimd.Int32x4, archsimd.Int32x4) {
	lo := v.ExtendLo8ToUint16().BitsToInt16()
	hi := v.ExtendHi8ToUint16().BitsToInt16()
	return lo.MulWidenLo(k), lo.MulWidenHi(k), hi.MulWidenLo(k), hi.MulWidenHi(k)
}

// reduce4 sums the 4 lanes of each of s0..s3 into the 4 lanes of the result.
// Pairs of vectors are first reduced to 2 lanes each by exchanging their high
// halves (lanes 2-3 of one for lanes 0-1 of the other, then adding), then the
// two results to 1 lane each by exchanging their odd lanes the same way.
func reduce4(s0, s1, s2, s3 archsimd.Int32x4, rc reduceConsts) archsimd.Int32x4 {
	u := reducePairs(s0, s2, rc.lanes01, rc.halves)
	v := reducePairs(s1, s3, rc.lanes01, rc.halves)
	return reducePairs(u, v, rc.lanes02, rc.pairs)
}

// reducePairs adds the lanes of x selected by mask to the others and returns
// the sums in the selected lanes, and the same for y in the other lanes, swap
// being the byte permutation exchanging the two sets of lanes.
func reducePairs(x, y, mask archsimd.Int32x4, swap archsimd.Int8x16) archsimd.Int32x4 {
	a := x.BitSelect(y, mask)
	b := y.BitSelect(x, mask).ToBits().ReshapeToUint8s().BitsToInt8().LookupOrZero(swap)
	return a.Add(b.ToBits().ReshapeToUint32s().BitsToInt32())
}

// pack16Wide packs the byte sums of 8 big endian 16-bit samples, the high byte
// of a sample in an even lane and its low byte in the odd lane after it. The
// high lanes are multiplied by 256 and the pairs added as qwords, x into its
// low dwords and y into its high dwords so that one bitselect gathers the four
// sums as (x0, y0, x1, y1), which the final swizzle, which also swaps the
// bytes, puts back in order with the samples of cd.
func pack16Wide(a, b, c, d archsimd.Int32x4, pc packConsts, pw packWideConsts) archsimd.Uint8x16 {
	var zero archsimd.Int32x4
	ab := addPairsWide(a.Mul(pw.mul), b.Mul(pw.mul), pw.lanes02).Add(pw.round).ShiftAllRight(bits).Max(zero).Min(pc.max)
	cd := addPairsWide(c.Mul(pw.mul), d.Mul(pw.mul), pw.lanes02).Add(pw.round).ShiftAllRight(bits).Max(zero).Min(pc.max)
	x := ab.Or(cd.ShiftAllLeft(16)).ToBits().ReshapeToUint8s().BitsToInt8()
	return x.LookupOrZero(pc.idx).ToBits()
}

// addPairsWide returns (x0+x1, y0+y1, x2+x3, y2+y3).
func addPairsWide(x, y, lanes02 archsimd.Int32x4) archsimd.Int32x4 {
	qx := x.ToBits().ReshapeToUint64s().BitsToInt64()
	qy := y.ToBits().ReshapeToUint64s().BitsToInt64()
	qx = qx.Add(qx.ShiftAllRight(32))
	qy = qy.Add(qy.ShiftAllLeft(32))
	sx := qx.ToBits().ReshapeToUint32s().BitsToInt32()
	sy := qy.ToBits().ReshapeToUint32s().BitsToInt32()
	return sx.BitSelect(sy, lanes02)
}

// pack16 shifts and clamps 16 rounded pixel sums in natural order to bytes:
// each sum is clamped to 0..255 in its 32-bit lane, the 4 vectors are merged
// into the 4 bytes of each lane and one swizzle restores the pixel order.
func pack16(a, b, c, d archsimd.Int32x4, pc packConsts) archsimd.Uint8x16 {
	var zero archsimd.Int32x4
	a = a.ShiftAllRight(bits).Max(zero).Min(pc.max)
	b = b.ShiftAllRight(bits).Max(zero).Min(pc.max)
	c = c.ShiftAllRight(bits).Max(zero).Min(pc.max)
	d = d.ShiftAllRight(bits).Max(zero).Min(pc.max)
	ab := a.Or(b.ShiftAllLeft(16))
	cd := c.Or(d.ShiftAllLeft(16))
	x := ab.Or(cd.ShiftAllLeft(8)).ToBits().ReshapeToUint8s().BitsToInt8()
	return x.LookupOrZero(pc.idx).ToBits()
}
