// Copyright 2026 Patrick Gaskin, 2013 Benoît Amiaux. All rights reserved. Use
// of this source code is governed by a MIT-style license that can be found in
// the LICENSE file.

package arez

import (
	"bytes"
	"fmt"
	"image"
	"math"
	"runtime"
	"testing"
)

func TestFilterSupport(t *testing.T) {
	for _, f := range append(filters, Lanczos(4), Lanczos(2.5), CustomBicubic(1.0/3, 1.0/3)) {
		support := f.Support()
		// lanczos is sin(pi*alpha)*..., about 1e-33 rather than 0
		expect(t, math.Abs(f.Get(support)) < 1e-9, true)
		expect(t, math.Abs(f.Get(support-0.25)) > 1e-3, true)
		for _, tt := range []struct{ input, output int }{{100, 200}, {100, 100}, {300, 100}, {250, 100}, {1920, 640}} {
			k, err := makeKernel(&ResizerConfig{Input: tt.input, Output: tt.output, Pack: 1, Depth: 8}, f, 0)
			expect(t, err, nil)
			r := max(1, float64(tt.input)/float64(tt.output))
			expect(t, k.size, 2*int(math.Ceil(r*support)))
		}
	}
	expect(t, Bilinear().Support(), 1)
	expect(t, Bicubic().Support(), 2)
	expect(t, Lanczos(3).Support(), 3)
}

// TestFractionalSupport checks the kernel sizes of filters with a support
// between whole pixels, which round up to whole taps on each side.
func TestFractionalSupport(t *testing.T) {
	src := []byte{64, 64, 64, 64, 64, 64, 64, 64}
	for _, support := range []float64{0.25, 0.5, 1.5, 2.25} {
		for _, out := range []int{16, 8, 5, 3} {
			k, err := makeKernel(&ResizerConfig{Input: 8, Output: out, Pack: 1, Depth: 8}, supportFilter{support}, 0)
			expect(t, err, nil)
			r := max(1, 8/float64(out))
			expect(t, k.size, min(2*int(math.Ceil(r*support)), 8))
			rz, err := NewResizer(&ResizerConfig{Input: 8, Output: out}, supportFilter{support})
			expect(t, err, nil)
			dst := make([]byte, out)
			rz.Resize(dst, src, 8, 1, out, 8)
			expect(t, string(dst), string(bytes.Repeat([]byte{64}, out)))
		}
	}
}

type constFilter struct{ weight float64 }

func (constFilter) Support() float64      { return 1 }
func (constFilter) Name() string          { return "const" }
func (f constFilter) Get(float64) float64 { return f.weight }

// spikeFilter gives a large weight to the nearest pixel and a nearly opposite
// one to the others, so the weights sum to a small fraction of the largest and
// the fixed point coefficients overflow.
type spikeFilter struct{}

func (spikeFilter) Support() float64 { return 1 }
func (spikeFilter) Name() string     { return "spike" }
func (spikeFilter) Get(x float64) float64 {
	if x < 0.5 {
		return 1000
	}
	return -999
}

// supportFilter is a box filter of the given support, which NewResizer must
// reject unless positive and finite instead of allocating a kernel of negative
// size.
type supportFilter struct{ support float64 }

func (f supportFilter) Support() float64  { return f.support }
func (supportFilter) Name() string        { return "support" }
func (supportFilter) Get(float64) float64 { return 1 }

// TestHugeSupport resizes with filters whose tap counts overflow int before
// they are clamped to the input.
func TestHugeSupport(t *testing.T) {
	src := []byte{0, 16, 32, 48, 64, 80, 96, 112}
	for _, out := range []int{16, 3} {
		var outs [][]byte
		for _, f := range []Filter{supportFilter{8}, supportFilter{math.MaxInt}, supportFilter{math.MaxInt / 2}} {
			r, err := NewResizer(&ResizerConfig{Input: 8, Output: out}, f)
			expect(t, err, nil)
			dst := make([]byte, out)
			r.Resize(dst, src, 8, 1, out, 8)
			outs = append(outs, dst)
		}
		for _, dst := range outs[1:] {
			if !bytes.Equal(dst, outs[0]) {
				t.Errorf("8 to %v: got %v, want %v", out, dst, outs[0])
			}
		}
	}
	for _, f := range []Filter{Lanczos(math.MaxInt / 2), supportFilter{math.MaxInt}} {
		for _, size := range []int{16, 4} {
			expect(t, Convert(image.NewGray(image.Rect(0, 0, size, size)), image.NewGray(image.Rect(0, 0, 8, 8)), f), nil)
		}
	}
}

type panicFilter struct{ value string }

func (panicFilter) Support() float64      { return 1 }
func (panicFilter) Name() string          { return "panic" }
func (f panicFilter) Get(float64) float64 { panic(f.value) }

func TestPanics(t *testing.T) {
	src := newImage(t, ycbcr(image.YCbCrSubsampleRatio420), image.Rect(0, 0, 32, 32))
	dst := newImage(t, ycbcr(image.YCbCrSubsampleRatio420), image.Rect(0, 0, 48, 40))
	recovered := func(what string, f func()) any {
		t.Helper()
		var v any
		func() {
			defer func() { v = recover() }()
			f()
		}()
		if v == nil {
			t.Errorf("%v: no panic", what)
		}
		return v
	}
	for _, threads := range []int{1, 4} {
		what := fmt.Sprintf("threads=%v", threads)
		filter := panicFilter{what}
		cfg, err := Prepare(dst, src)
		expect(t, err, nil)
		cfg.Threads = threads
		if v := recovered(what+" NewConverter", func() { NewConverter(cfg, filter) }); v != nil && v != what {
			t.Errorf("%v NewConverter: got %v, want %v", what, v, what)
		}
		if v := recovered(what+" nil bicubic", func() { NewConverter(cfg, (*bicubic)(nil)) }); v != nil {
			if _, ok := v.(runtime.Error); !ok {
				t.Errorf("%v nil bicubic: got %v, want a runtime error", what, v)
			}
		}
		c, err := NewConverter(cfg, Bilinear())
		expect(t, err, nil)
		c.wrez[1].scaler = func([]byte, []byte, []int16, []int32, int, int, int, int, int) { panic(what) }
		if v := recovered(what+" Convert", func() { c.Convert(dst, src) }); v != nil && v != what {
			t.Errorf("%v Convert: got %v, want %v", what, v, what)
		}
		r, err := NewResizer(&ResizerConfig{Input: 32, Output: 16, Vertical: true, Threads: threads}, Bilinear())
		expect(t, err, nil)
		r.scaler = func([]byte, []byte, []int16, []int32, int, int, int, int, int) { panic(what) }
		if v := recovered(what+" Resize", func() { r.Resize(make([]byte, 16*16), make([]byte, 32*16), 16, 32, 16, 16) }); v != nil && v != what {
			t.Errorf("%v Resize: got %v, want %v", what, v, what)
		}
	}
	if v := recovered("Convert", func() { Convert(dst, src, panicFilter{"Convert"}) }); v != nil && v != "Convert" {
		t.Errorf("Convert: got %v, want Convert", v)
	}
}
