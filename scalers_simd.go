// Copyright 2026 Patrick Gaskin, 2014 Benoît Amiaux. All rights reserved. Use
// of this source code is governed by a MIT-style license that can be found in
// the LICENSE file.

//go:build goexperiment.simd && (amd64 || arm64 || wasm)

package arez

import "unsafe"

// sample8 and sample16 are the type arguments of the row kernels, which pack
// 8-bit or 16-bit samples. The two types have different sizes so each kernel is
// compiled once per sample width when used as a generic param (and len becomes
// a constant).
type (
	sample8  [1]byte
	sample16 [2]byte
)

type sampleWidth interface {
	sample8 | sample16
}

// padKernel returns the horizontal kernel k with its pixels padded to a
// multiple of pixels by repeating the last one, and the coefficients of every
// pixel in natural order, zero padded to a multiple of taps.
func padKernel(k kernel, taps, pixels int) kernel {
	size := len(k.offsets)
	chunk := padSize(k.size, taps)
	n := padSize(size, pixels)
	cof := make([]int16, n*chunk)
	off := make([]int32, n)
	for p := range n {
		q := min(p, size-1)
		copy(cof[p*chunk:], k.coeffs[q*k.size:q*k.size+k.size])
		off[p] = k.offsets[q]
	}
	k.coeffs, k.offsets = cof, off
	return k
}

// widenKernel converts a horizontal 16-bit kernel, taps coefficients per output
// sample with byte offsets, to the 8-bit kernel of the same bytes: the high and
// low bytes of sample x are two output bytes with the offset of x, the high one
// with the coefficients of x in its even taps and the low one in its odd taps,
// like unpack with twice the pack.
func widenKernel(k kernel) kernel {
	size := len(k.offsets)
	cof := make([]int16, 2*size*2*k.size)
	off := make([]int32, 2*size)
	for p := range size {
		for b := range 2 {
			off[2*p+b] = k.offsets[p]
			d := cof[(2*p+b)*2*k.size:]
			for t, c := range k.coeffs[p*k.size : (p+1)*k.size] {
				d[2*t+b] = c
			}
		}
	}
	k.coeffs, k.offsets, k.size = cof, off, 2*k.size
	return k
}

func padSize(n, m int) int {
	return (n + m - 1) / m * m
}

// at32 returns the i'th int32 at p.
func at32(p unsafe.Pointer, i int) int32 {
	return *(*int32)(unsafe.Add(p, 4*i))
}
