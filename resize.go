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
//
// ReducingGap speeds up large downscales by resizing in two steps, like
// [Pillow]. First, the input is averaged like [Box] in blocks of n =
// floor(Input/Output/ReducingGap) pixels, if n is at least 2. Second, the
// averaged pixels are resized to Output with the filter, which changes the size
// by no less than ReducingGap times. If set, ReducingGap must be at least 1.
// The bigger ReducingGap, the closer the result is to a single resize. The
// smaller ReducingGap, the faster the resize. With a ReducingGap of 3 or more,
// the result is indistinguishable from a single resize in most cases.
// ReducingGap has no effect with [Box] and [Nearest] (or other filters with a
// support independent of the scale), which are not faster in two steps.
//
// [Pillow]: https://pillow.readthedocs.io/en/latest/reference/Image.html#PIL.Image.Image.resize
type ResizerConfig struct {
	Depth       int     // bits per sample, 8 (the default, also 0) or 16
	Input       int     // input size in pixels (must be at least 2, or 2 per field when interlaced)
	Output      int     // output size in pixels (must be at least 1)
	Vertical    bool    // resize the height instead of the width
	Interlaced  bool    // whether the rows are two interlaced fields, vertical resizes only
	Pack        int     // samples per pixel, 1 (the default, also 0) or more (must not be negative)
	Threads     int     // number of goroutines, 0 for GOMAXPROCS
	DisableSIMD bool    // use the Go scalers only
	ReducingGap float64 // average large downscales first, see above (0 disables it)
}

type scaler func(dst, src []byte, cof []int16, off []int32,
	taps, width, height, dstPitch, srcPitch int)

// Resizer resizes planes of raw samples in one direction, see [NewResizer]. A
// resizer is safe for concurrent use.
type Resizer struct {
	cfg     ResizerConfig
	kernels []kernel
	scaler  scaler

	// for ReducingGap, the box average and the filter steps
	reduce, resize *Resizer
	buf            sync.Pool // *[]byte for the output of reduce
}

// NewResizer returns a [Resizer] for the resize cfg describes, with the weights
// of filter. It copies cfg. NewResizer returns an error matching
// [ErrInvalidDescriptor] if cfg/filter is not valid, or
// [ErrTooSmall]/[ErrTooLarge] if unsupported.
func NewResizer(cfg *ResizerConfig, filter Filter) (*Resizer, error) {
	if filter == nil {
		return nil, fmt.Errorf("%w: nil filter", ErrInvalidFilter)
	}
	ctx := Resizer{
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
	if mid, ok := reducedInput(&ctx.cfg, filter); ok {
		step := ctx.cfg
		step.ReducingGap = 0
		step.Output = mid
		reduce, err := NewResizer(&step, Box())
		if err != nil {
			return nil, err
		}
		step.Input, step.Output = mid, ctx.cfg.Output
		resize, err := NewResizer(&step, filter)
		if err != nil {
			return nil, err
		}
		ctx.reduce, ctx.resize = reduce, resize
		return &ctx, nil
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

// reducedInput returns the size of the box average of a resize with
// ReducingGap, and whether to average first.
func reducedInput(cfg *ResizerConfig, filter Filter) (int, bool) {
	if cfg.ReducingGap == 0 {
		return 0, false
	}
	ratio := float64(cfg.Input) / float64(cfg.Output)
	n := int(min(ratio/cfg.ReducingGap, math.MaxInt32)) // pixels per block
	if n < 2 {
		return 0, false
	}
	mid := (cfg.Input + n - 1) / n
	// a box average already reads every input pixel once
	if mid>>bin(cfg.Interlaced) < 2 || filter == Box() || filter.Support(1/ratio) <= filter.Support(float64(n)/ratio) {
		return 0, false
	}
	return mid, true
}

func checkResize(cfg *ResizerConfig) error {
	if g := cfg.ReducingGap; g != 0 && !(g >= 1) {
		return fmt.Errorf("%w: reducing gap %v", ErrInvalidDescriptor, g)
	}
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

// resizeReduced resizes src into dst through a buffer of the output of
// c.reduce, with pixels of pk bytes.
func (c *Resizer) resizeReduced(dst, src []byte, width, height, dstPitch, srcPitch, pk int) {
	w, h := c.resize.cfg.Input, height
	if c.cfg.Vertical {
		w, h = width, c.resize.cfg.Input
	}
	pitch := w * pk
	buf, _ := c.buf.Get().(*[]byte)
	if buf == nil || len(*buf) < pitch*h {
		b := make([]byte, pitch*h)
		buf = &b
	}
	c.reduce.Resize(*buf, src, width, height, pitch, srcPitch)
	c.resize.Resize(dst, *buf, w, h, dstPitch, pitch)
	c.buf.Put(buf)
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

// Resize resizes the plane src of width x height pixels into dst. The rows of
// dst and src start dstPitch and srcPitch bytes apart, and 16-bit samples are
// big endian, as in the image package.
//
// The resized dimension of the source, width for a horizontal resize and height
// for a vertical one, must be the configured Input, and the other one is kept.
//
// The pitches must not be negative and the slices must be sized correctly for
// the planes (src at least srcPitch*(height-1) + width*bpp bytes, dst at least
// dstPitch*(rows-1) + columns*bpp bytes, where rows and columns are the
// destination dimensions, and bpp = Pack*Depth/8 the bytes per pixel).
//
// The rows of dst are written from several goroutines and must not overlap.
// When there is more than one goroutine, dstPitch must be at least columns*bpp.
// The rows of src may overlap.
//
// Resize will panic if any of these requirements are not met.
func (c *Resizer) Resize(dst, src []byte, width, height, dstPitch, srcPitch int) {
	field := bin(c.cfg.Interlaced)
	dwidth, dheight, input := c.cfg.Output, height, width
	if c.cfg.Vertical {
		dwidth, dheight, input = width, c.cfg.Output, height
	}
	pk := c.cfg.Pack * (c.cfg.Depth / 8)
	if input != c.cfg.Input || width < 1 || height < 1 {
		panic(fmt.Sprintf("arez: resize of a %vx%v plane with a resizer of %v to %v", width, height, c.cfg.Input, c.cfg.Output))
	}
	if srcPitch < 0 || dstPitch < 0 {
		panic(fmt.Sprintf("arez: resize with a source pitch of %v and a destination pitch of %v", srcPitch, dstPitch))
	}
	if n, ok := planeBytes(width, height, pk, srcPitch); !ok || len(src) < n {
		panic(fmt.Sprintf("arez: resize source of %v bytes, %vx%v pixels of %v bytes with a pitch of %v need %v", len(src), width, height, pk, srcPitch, n))
	}
	if n, ok := planeBytes(dwidth, dheight, pk, dstPitch); !ok || len(dst) < n {
		panic(fmt.Sprintf("arez: resize destination of %v bytes, %vx%v pixels of %v bytes with a pitch of %v need %v", len(dst), dwidth, dheight, pk, dstPitch, n))
	}
	if dheight > 1 && dstPitch < dwidth*pk {
		panic(fmt.Sprintf("arez: resize destination with overlapping rows, a pitch of %v for rows of %v bytes", dstPitch, dwidth*pk))
	}
	if c.reduce != nil {
		c.resizeReduced(dst, src, width, height, dstPitch, srcPitch, pk)
		return
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
			k.size, dwidth*pk, rows, dstPitch<<field, srcPitch<<field,
			dst[dstPitch*i:], src[srcPitch*i:], k.coeffs, k.cofscale, k.offsets)
	}
	group.Wait()
}
