// Copyright 2026 Patrick Gaskin, 2013 Benoît Amiaux. All rights reserved. Use
// of this source code is governed by a MIT-style license that can be found in
// the LICENSE file.

package arez

import (
	"fmt"
	"math"
	"runtime"
	"sync"
)

// ResizerConfig configures a [Resizer], see [NewResizer].
//
// Rows must not be more than 2^31-1 bytes, or an error matching [ErrTooLarge]
// will be returned.
type ResizerConfig struct {
	Depth       int  // bits per sample, 8 (the default, also 0) or 16
	Input       int  // input size in pixels (must be at least 2, or 2 per field when interlaced)
	Output      int  // output size in pixels (must be at least 1)
	Vertical    bool // resize the height instead of the width
	Interlaced  bool // whether the rows are two interlaced fields, vertical resizes only
	Pack        int  // samples per pixel, 1 (the default, also 0) or more (must not be negative)
	Threads     int  // number of goroutines, 0 for GOMAXPROCS
	DisableSIMD bool // use the Go scalers only
}

// Resizer resizes planes of raw samples in one direction, see [NewResizer].
type Resizer interface {
	// Resize resizes the plane src of width x height pixels into dst. The rows
	// of dst and src start dstPitch and srcPitch bytes apart, and 16-bit
	// samples are big endian, as in the image package.
	//
	// The resized dimension of the source, width for a horizontal resize and
	// height for a vertical one, must be the configured Input, and the other
	// one is kept.
	//
	// The pitches must not be negative and the slices must be sized correctly
	// for the planes (src at least srcPitch*(height-1) + width*bpp bytes, dst
	// at least dstPitch*(rows-1) + columns*bpp bytes, where rows and columns
	// are the destination dimensions, and bpp = Pack*Depth/8 the bytes per
	// pixel).
	//
	// The rows of dst are written from several goroutines and must not overlap.
	// When there is more than one goroutine, dstPitch must be at least
	// columns*bpp. The rows of src may overlap.
	//
	// Resize will panic if any of these requirements are not met.
	Resize(dst, src []byte, width, height, dstPitch, srcPitch int)
}

type scaler func(dst, src []byte, cof []int16, off []int32,
	taps, width, height, dstPitch, srcPitch int)

type resizer struct {
	cfg     ResizerConfig
	kernels []kernel
	scaler  scaler
}

// NewResizer returns a [Resizer] for the resize cfg describes, with the weights
// of filter. It copies cfg. NewResizer returns an error matching
// [ErrInvalidDescriptor] if cfg/filter is not valid, or
// [ErrTooSmall]/[ErrTooLarge] if unsupported.
func NewResizer(cfg *ResizerConfig, filter Filter) (Resizer, error) {
	if filter == nil {
		return nil, fmt.Errorf("%w: nil filter", ErrInvalidFilter)
	}
	ctx := resizer{
		cfg: *cfg,
	}
	if ctx.cfg.Depth == 0 {
		ctx.cfg.Depth = 8
	}
	if ctx.cfg.Pack == 0 {
		ctx.cfg.Pack = 1
	}
	if ctx.cfg.Threads <= 0 {
		ctx.cfg.Threads = runtime.GOMAXPROCS(0)
	}
	// interlaced rows don't matter for horizontal resizes
	ctx.cfg.Interlaced = ctx.cfg.Interlaced && ctx.cfg.Vertical
	if err := checkResize(&ctx.cfg); err != nil {
		return nil, err
	}
	k, err := makeKernel(&ctx.cfg, filter, 0)
	if err != nil {
		return nil, err
	}
	ctx.kernels = []kernel{k}
	if !cfg.Vertical {
		ctx.scaler = getHorizontalScaler(&ctx.kernels[0], !cfg.DisableSIMD)
		return &ctx, nil
	}
	ctx.scaler = getVerticalScaler(&ctx.kernels[0], !cfg.DisableSIMD)
	if cfg.Interlaced {
		k, err := makeKernel(&ctx.cfg, filter, 1)
		if err != nil {
			return nil, err
		}
		ctx.kernels = append(ctx.kernels, k)
	}
	return &ctx, nil
}

func checkResize(cfg *ResizerConfig) error {
	if cfg.Depth != 8 && cfg.Depth != 16 {
		return fmt.Errorf("%w: depth %v", ErrInvalidDescriptor, cfg.Depth)
	}
	if cfg.Pack < 1 {
		return fmt.Errorf("%w: pack %v", ErrInvalidDescriptor, cfg.Pack)
	}
	if cfg.Pack > math.MaxInt32/(cfg.Depth/8) {
		return fmt.Errorf("%w: pixels of %v %v-bit samples", ErrTooLarge, cfg.Pack, cfg.Depth)
	}
	if cfg.Input>>bin(cfg.Interlaced) < 2 {
		return fmt.Errorf("%w: input %v", ErrTooSmall, cfg.Input)
	}
	if cfg.Output < 1 {
		return fmt.Errorf("%w: output %v", ErrTooSmall, cfg.Output)
	}
	bpp := 1
	if !cfg.Vertical {
		bpp = cfg.Pack * (cfg.Depth / 8)
	}
	if cfg.Input > math.MaxInt32/bpp {
		return fmt.Errorf("%w: input %v of %v bytes", ErrTooLarge, cfg.Input, bpp)
	}
	if cfg.Output > math.MaxInt32/bpp {
		return fmt.Errorf("%w: output %v of %v bytes", ErrTooLarge, cfg.Output, bpp)
	}
	return nil
}

// planeBytes returns the bytes a plane of width x height pixels of bpp bytes
// occupies at a pitch of pitch bytes, pitch*(height-1) + width*bpp, and
// whether it fits an int. A pitch may be smaller than a row, rows overlap.
func planeBytes(width, height, bpp, pitch int) (int, bool) {
	row := width * bpp
	if row/bpp != width || pitch > 0 && height > 1 && pitch > (math.MaxInt-row)/(height-1) {
		return 0, false
	}
	return pitch*(height-1) + row, true
}

// jobs manages resize/convert jobs, managing goroutines if more than one,
// rethrowing panics from goroutines.
type jobs struct {
	group    sync.WaitGroup
	mu       sync.Mutex
	panicked bool
	value    any
}

func (j *jobs) Wait() {
	j.group.Wait()
	if j.panicked {
		panic(j.value)
	}
}

func (j *jobs) run(job func()) {
	defer func() {
		if v := recover(); v != nil {
			j.mu.Lock()
			if !j.panicked {
				j.panicked, j.value = true, v
			}
			j.mu.Unlock()
		}
	}()
	job()
}

func dispatch(j *jobs, threads int, job func()) {
	if threads == 1 {
		j.run(job)
		return
	}
	j.group.Go(func() { j.run(job) })
}

func scaleSlice(group *jobs, threads int, scaler scaler,
	dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	dispatch(group, threads, func() {
		scaler(dst, src, cof, off, taps, width, height, dp, sp)
	})
}

func scaleSlices(group *jobs, scaler scaler,
	vertical bool, threads, taps, width, height, dp, sp int,
	dst, src []byte, cof []int16, cofscale int, off []int32) {
	dispatch(group, threads, func() {
		nh := max(height/threads, 1)
		di := 0
		si := 0
		oi := 0
		ci := 0
		for i := range threads {
			last := i+1 == threads
			ih := nh
			if last {
				ih = height - nh*(threads-1)
			}
			if ih == 0 {
				continue
			}
			c, o := cof, off
			if vertical {
				c = cof[ci : ci+ih*taps*cofscale]
				o = off[oi : oi+ih]
			}
			scaleSlice(group, threads, scaler,
				dst[di:di+dp*(ih-1)+width],
				src[si:],
				c, o,
				taps, width, ih, dp, sp)
			if last {
				break
			}
			di += ih * dp
			if vertical {
				ci += ih * taps * cofscale
				for _, o := range off[oi : oi+ih] {
					si += sp * int(o)
				}
				oi += ih
			} else {
				si += sp * ih
			}
		}
	})
}

func (c *resizer) Resize(dst, src []byte, width, height, dp, sp int) {
	field := bin(c.cfg.Interlaced)
	dwidth, dheight, input := c.cfg.Output, height, width
	if c.cfg.Vertical {
		dwidth, dheight, input = width, c.cfg.Output, height
	}
	pk := c.cfg.Pack * (c.cfg.Depth / 8)
	if input != c.cfg.Input || width < 1 || height < 1 {
		panic(fmt.Sprintf("arez: resize of a %vx%v plane with a resizer of %v to %v", width, height, c.cfg.Input, c.cfg.Output))
	}
	if sp < 0 || dp < 0 {
		panic(fmt.Sprintf("arez: resize with a source pitch of %v and a destination pitch of %v", sp, dp))
	}
	if n, ok := planeBytes(width, height, pk, sp); !ok || len(src) < n {
		panic(fmt.Sprintf("arez: resize source of %v bytes, %vx%v pixels of %v bytes with a pitch of %v need %v", len(src), width, height, pk, sp, n))
	}
	if n, ok := planeBytes(dwidth, dheight, pk, dp); !ok || len(dst) < n {
		panic(fmt.Sprintf("arez: resize destination of %v bytes, %vx%v pixels of %v bytes with a pitch of %v need %v", len(dst), dwidth, dheight, pk, dp, n))
	}
	if dheight > 1 && dp < dwidth*pk {
		panic(fmt.Sprintf("arez: resize destination with overlapping rows, a pitch of %v for rows of %v bytes", dp, dwidth*pk))
	}
	var group jobs
	for i, k := range c.kernels[:1+field] {
		rows := dheight
		if c.cfg.Vertical {
			rows = (c.cfg.Output + (1-i)*int(field)) >> field
		}
		if rows == 0 {
			// the bottom field of a single row (dst may end before it)
			continue
		}
		// scaleSlices gives every thread at least one row
		threads := clip(c.cfg.Threads, 1, max(rows, 1))
		scaleSlices(&group, c.scaler, c.cfg.Vertical, threads,
			k.size, dwidth*pk, rows, dp<<field, sp<<field,
			dst[dp*i:], src[sp*i:], k.coeffs, k.cofscale, k.offsets)
	}
	group.Wait()
}
