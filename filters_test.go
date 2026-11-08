// Copyright 2026 Patrick Gaskin, 2013 Benoît Amiaux. All rights reserved. Use
// of this source code is governed by a MIT-style license that can be found in
// the LICENSE file.

package arez

import (
	"bytes"
	"fmt"
	"image"
	"math"
	"math/rand"
	"runtime"
	"testing"
)

func TestFilterSupport(t *testing.T) {
	for _, f := range append(filters, Lanczos(4), Lanczos(2.5), CustomBicubic(1.0/3, 1.0/3)) {
		support := f.Support(1)
		expect(t, f.Weight(support, 1), 0)
		expect(t, f.Weight(-support, 1), 0)
		expect(t, math.Abs(f.Weight(support-0.25, 1)) > 1e-3, true)
		// stretched over the output pixels when downscaling
		expect(t, f.Support(2), support)
		expect(t, f.Support(0.5), 2*support)
		for _, dx := range []float64{0, 0.25, 0.5, 1.25, support - 0.25} {
			expect(t, f.Weight(dx, 2), f.Weight(dx, 1))
			expect(t, f.Weight(-2*dx, 0.5), f.Weight(dx, 1))
		}
		for _, tt := range []struct{ input, output int }{{100, 200}, {100, 100}, {300, 100}, {250, 100}, {1920, 640}} {
			k, err := makeKernel(&ResizerConfig{Input: tt.input, Output: tt.output, Pack: 1, Depth: 8}, f, 0)
			expect(t, err, nil)
			r := max(1, float64(tt.input)/float64(tt.output))
			expect(t, k.size, 2*int(math.Ceil(r*support)))
		}
	}
	expect(t, Bilinear().Support(1), 1)
	expect(t, Bicubic().Support(1), 2)
	expect(t, Lanczos(3).Support(1), 3)
}

// TestKernel checks that Kernel filters are stretched like the built-in ones
// and are zero outside their support.
func TestKernel(t *testing.T) {
	linear := Kernel(1, func(x float64) float64 { return 1 - x })
	expect(t, filterString(linear), "kernel(support=1)")
	rng := rand.New(rand.NewSource(1))
	for _, s := range []struct{ input, output int }{{8, 16}, {16, 8}, {100, 37}, {37, 100}, {64, 3}} {
		src := make([]byte, s.input*3)
		rng.Read(src)
		var outs [2][]byte
		for i, f := range []Filter{Bilinear(), linear} {
			r, err := NewResizer(&ResizerConfig{Input: s.input, Output: s.output}, f)
			expect(t, err, nil)
			outs[i] = make([]byte, s.output*3)
			r.Resize(outs[i], src, s.input, 3, s.output, s.input)
		}
		if !bytes.Equal(outs[0], outs[1]) {
			t.Errorf("%v to %v: got %v, want %v", s.input, s.output, outs[1], outs[0])
		}
	}
	one := Kernel(1.5, func(float64) float64 { return 1 })
	expect(t, one.Weight(1.25, 1), 1)
	expect(t, one.Weight(-1.25, 1), 1)
	expect(t, one.Weight(1.5, 1), 0)
	expect(t, one.Weight(2.75, 0.5), 1)
	expect(t, one.Weight(3, 0.5), 0)
}

func TestFilterString(t *testing.T) {
	for _, tt := range []struct {
		filter     Filter
		str, gostr string
	}{
		{Bilinear(), "bilinear", "arez.Bilinear()"},
		{Bicubic(), "bicubic(b=0, c=0.5)", "arez.Bicubic()"},
		{CustomBicubic(0, 0.5), "bicubic(b=0, c=0.5)", "arez.Bicubic()"},
		{CustomBicubic(1.0/3, 1.0/3), "bicubic(b=0.3333333333333333, c=0.3333333333333333)", "arez.CustomBicubic(0.3333333333333333, 0.3333333333333333)"},
		{CustomBicubic(1, 0), "bicubic(b=1, c=0)", "arez.CustomBicubic(1, 0)"},
		{Lanczos(3), "lanczos(a=3)", "arez.Lanczos(3)"},
		{Lanczos(2.5), "lanczos(a=2.5)", "arez.Lanczos(2.5)"},
		{Nearest(), "nearest", "arez.Nearest()"},
		{Box(), "box", "arez.Box()"},
	} {
		expect(t, fmt.Sprint(tt.filter), tt.str)
		expect(t, fmt.Sprintf("%#v", tt.filter), tt.gostr)
		expect(t, filterString(tt.filter), tt.str)
	}
	expect(t, filterString(supportFilter{1}), "arez.supportFilter")
	expect(t, filterString(&supportFilter{0.5}), "arez.supportFilter")
	expect(t, filterString((*supportFilter)(nil)), "arez.supportFilter")
}

// TestNearestBox checks the outputs of the scale-dependent filters, including
// output pixels with centers exactly between two input pixels.
func TestNearestBox(t *testing.T) {
	for _, tt := range []struct {
		filter Filter
		src    []byte
		out    int
		want   []byte
	}{
		{Nearest(), []byte{10, 20}, 3, []byte{10, 10, 20}},
		{Nearest(), []byte{0, 10, 20, 30, 40, 50, 60, 70}, 3, []byte{10, 30, 60}},
		{Nearest(), []byte{0, 10, 20, 30, 40, 50, 60, 70}, 2, []byte{10, 50}},
		{Nearest(), []byte{0, 10, 20, 30, 40, 50, 60, 70}, 16, []byte{0, 0, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70}},
		{Box(), []byte{10, 20}, 3, []byte{10, 15, 20}},
		{Box(), []byte{0, 10, 20, 30, 40, 50, 60, 70}, 2, []byte{15, 55}},
		{Box(), []byte{0, 30, 60, 90, 120, 150}, 4, []byte{10, 50, 100, 140}},
		{Box(), []byte{0, 10, 20, 30, 40, 50, 60, 70}, 16, []byte{0, 0, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70}},
		{Box(), []byte{0, 30, 60, 90}, 6, []byte{0, 15, 30, 60, 75, 90}},
	} {
		for _, vertical := range []bool{false, true} {
			r, err := NewResizer(&ResizerConfig{Input: len(tt.src), Output: tt.out, Vertical: vertical}, tt.filter)
			expect(t, err, nil)
			dst := make([]byte, tt.out)
			if vertical {
				r.Resize(dst, tt.src, 1, len(tt.src), 1, 1)
			} else {
				r.Resize(dst, tt.src, len(tt.src), 1, tt.out, len(tt.src))
			}
			if !bytes.Equal(dst, tt.want) {
				t.Errorf("%v %v to %v vertical=%v: got %v, want %v", tt.filter, len(tt.src), tt.out, vertical, dst, tt.want)
			}
		}
	}
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

func (constFilter) Support(float64) float64           { return 1 }
func (f constFilter) Weight(float64, float64) float64 { return f.weight }

// spikeFilter gives a large weight to the nearest pixel and a nearly opposite
// one to the others, so the weights sum to a small fraction of the largest and
// the fixed point coefficients overflow.
type spikeFilter struct{}

func (spikeFilter) Support(float64) float64 { return 1 }
func (spikeFilter) Weight(dx, _ float64) float64 {
	if math.Abs(dx) < 0.5 {
		return 1000
	}
	return -999
}

// supportFilter is a box filter of the given support, which NewResizer must
// reject unless positive and finite instead of allocating a kernel of negative
// size.
type supportFilter struct{ support float64 }

func (f supportFilter) Support(scale float64) float64 { return f.support / min(1, scale) }
func (supportFilter) Weight(float64, float64) float64 { return 1 }

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

func (panicFilter) Support(float64) float64           { return 1 }
func (f panicFilter) Weight(float64, float64) float64 { panic(f.value) }

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
