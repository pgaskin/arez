// Copyright 2013 Benoît Amiaux. All rights reserved. Use of this source code is
// governed by a MIT-style license that can be found in the LICENSE file.

package arez

import (
	"cmp"
	"fmt"
	"math"
	"slices"
)

type kernel struct {
	coeffs   []int16
	offsets  []int32
	size     int
	cofscale int  // how many more coeffs do we have
	layout   int  // backend specific coefficient layout
	wide     bool // 16-bit samples
}

func bin(v bool) uint {
	if v {
		return 1
	}
	return 0
}

func clip(v, lo, hi int) int {
	return min(max(v, lo), hi)
}

func makeDoubleKernel(cfg *ResizerConfig, filter Filter, field, idx uint) ([]int, []float64, []float64, int, int) {
	scale := float64(cfg.Output) / float64(cfg.Input)
	step := min(1, scale)
	support := filter.Support() / step
	taps := int(min(math.Ceil(support), float64(cfg.Input))) * 2
	taps = min(taps, (cfg.Input>>field)&^1)
	offsets := make([]int, cfg.Output)
	sums := make([]float64, cfg.Output)
	weights := make([]float64, cfg.Output*taps)
	xmid := float64(cfg.Input-cfg.Output) / float64(cfg.Output*2)
	xstep := 1 / scale
	// interlaced resize see only one field but still use full res pixel positions
	ftaps := taps << field
	size := (cfg.Output + int(field*(1-idx))) >> field
	step /= float64(1 + field)
	xmid += xstep * float64(field*idx)
	for i := range size {
		left := int(math.Ceil(xmid)) - ftaps>>1
		x := clip(left, 0, max(0, cfg.Input-ftaps))
		offsets[i] = x
		for j := range ftaps {
			src := left + j
			if field != 0 && idx^uint(src&1) != 0 {
				continue
			}
			weight := filter.Get(math.Abs(xmid-float64(src)) * step)
			src = clip(src, x, cfg.Input-1) - x
			src >>= field
			weights[i*taps+src] += weight
			sums[i] += weight
		}
		xmid += xstep * float64(1+field)
	}
	return offsets, sums, weights, taps, size
}

type weight struct {
	weight float64
	offset int
}

// makeIntegerKernel scales the weights of every output pixel to fixed point of
// bits fractional bits, normalized by their sum, spreading the rounding error
// from the largest weights to the smallest. It returns an error for weights
// that do not sum to a finite non-zero value (so for NaN or infinite weights)
// or scale beyond int16.
func makeIntegerKernel(taps, size int, cof, sums []float64, pos []int, field, idx uint) ([]int16, []int, error) {
	coeffs := make([]int16, taps*size)
	offsets := make([]int, size)
	weights := make([]weight, taps)
	for i, sum := range sums[:size] {
		if sum == 0 || math.IsNaN(sum) || math.IsInf(sum, 0) {
			return nil, nil, fmt.Errorf("weights of output pixel %v sum to %v", i, sum)
		}
		for j, w := range cof[:taps] {
			weights[j] = weight{w, j}
		}
		// largest weights first
		slices.SortFunc(weights, func(a, b weight) int {
			return cmp.Compare(math.Abs(b.weight), math.Abs(a.weight))
		})
		diff := float64(0)
		scale := 1 << bits / sum
		for _, it := range weights {
			w := it.weight*scale + diff
			iw := math.Floor(w + 0.5)
			if iw > math.MaxInt16 || iw < math.MinInt16 {
				return nil, nil, fmt.Errorf("weight %v of output pixel %v is %.3g times their sum", it.weight, i, it.weight/sum)
			}
			coeffs[i*taps+it.offset] = int16(iw)
			diff = w - iw
		}
		cof = cof[taps:]
		offsets[i] = (pos[i] + int(field-idx)) >> field
	}
	return coeffs, offsets, nil
}

// makeKernel builds the kernel of the field idx of a resize, in the natural
// layout of the Go scalers unless the backend prepares it. Offsets are int32:
// horizontal ones are bytes, vertical ones are steps between output rows, both
// below the input size checkResize bounds.
func makeKernel(cfg *ResizerConfig, filter Filter, idx uint) (kernel, error) {
	if s := filter.Support(); !(s > 0) || math.IsInf(s, 1) {
		return kernel{}, fmt.Errorf("%w: %v has a support of %v", ErrInvalidFilter, filter.Name(), s)
	}
	field := bin(cfg.Interlaced)
	pos, sums, cof, taps, size := makeDoubleKernel(cfg, filter, field, idx)
	coeffs, offsets, err := makeIntegerKernel(taps, size, cof, sums, pos, field, idx)
	if err != nil {
		return kernel{}, fmt.Errorf("%w: %v %v", ErrInvalidFilter, filter.Name(), err)
	}
	if cfg.Vertical {
		for i := len(offsets) - 1; i > 0; i-- {
			offsets[i] -= offsets[i-1]
		}
	}
	off := make([]int32, len(offsets))
	for i, o := range offsets {
		off[i] = int32(o)
	}
	if !cfg.Vertical {
		if cfg.Pack > 1 {
			// unpacked to pack*pack coefficients per tap
			if cfg.Pack > math.MaxInt32/cfg.Pack || len(coeffs) > math.MaxInt32/(cfg.Pack*cfg.Pack) {
				return kernel{}, fmt.Errorf("%w: kernel of %v taps of %v samples for %v pixels", ErrTooLarge, taps, cfg.Pack, len(offsets))
			}
			coeffs, off, taps = unpack(coeffs, off, taps, cfg.Pack)
		}
		if cfg.Depth == 16 {
			// horizontal offsets are in bytes
			for i, o := range off {
				off[i] = o * 2
			}
		}
	}
	k := kernel{coeffs: coeffs, offsets: off, size: taps, cofscale: 1, wide: cfg.Depth == 16}
	if !hasSIMD() || cfg.DisableSIMD {
		return k, nil
	}
	return prepareKernel(cfg, k), nil
}

func unpack(coeffs []int16, offsets []int32, taps, pack int) ([]int16, []int32, int) {
	cof := make([]int16, len(coeffs)*pack*pack)
	off := make([]int32, len(offsets)*pack)
	di := 0
	ci := 0
	oi := 0
	next := make([]int16, pack*taps)
	for _, offset := range offsets {
		clear(next)
		for i := range taps {
			next[i*pack] = coeffs[ci+i]
		}
		for i := range pack {
			off[oi+i] = offset * int32(pack)
			copy(cof[di+pack*taps*i:], next)
			copy(next[i+1:], next[i:])
			clear(next[:i+1])
		}
		di += taps * pack * pack
		ci += taps
		oi += pack
	}
	return cof, off, taps * pack
}
