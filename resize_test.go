// Copyright 2026 Patrick Gaskin, 2013 Benoît Amiaux. All rights reserved. Use
// of this source code is governed by a MIT-style license that can be found in
// the LICENSE file.

package arez

import (
	"bytes"
	"errors"
	"fmt"
	"image"
	"math"
	"math/rand"
	"strings"
	"sync"
	"testing"
)

// TestResizeThreads resizes a few rows with more threads than rows, and with
// the default thread count, and checks the result against one thread.
func TestResizeThreads(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const n = 16
	src := make([]byte, n*n)
	rng.Read(src)
	for _, vertical := range []bool{false, true} {
		for _, interlaced := range []bool{false, vertical} {
			for _, size := range []int{1, 2, 3, 5} {
				var want []byte
				for _, threads := range []int{1, 0, 2, 7, 64} {
					cfg := ResizerConfig{Input: n, Output: 2 * n, Vertical: vertical, Interlaced: interlaced, Threads: threads}
					width, height, dp := n, size, 2*n
					if vertical {
						cfg.Output = size
						width, height, dp = n, n, n
					}
					dst := make([]byte, dp*cfg.Output)
					if vertical {
						dst = make([]byte, dp*size)
					}
					r, err := NewResizer(&cfg, Bicubic())
					expect(t, err, nil)
					r.Resize(dst, src, width, height, dp, n)
					if threads == 1 {
						want = dst
					} else if !bytes.Equal(dst, want) {
						t.Errorf("vertical=%v interlaced=%v size=%v threads=%v: output differs from one thread", vertical, interlaced, size, threads)
					}
				}
			}
		}
	}
}

// TestHorizontalInterlaced checks that a horizontal resize ignores the
// interlaced flag, which describes rows.
func TestHorizontalInterlaced(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	src := make([]byte, 16*4)
	rng.Read(src)
	var want []byte
	for _, interlaced := range []bool{false, true} {
		dst := make([]byte, 32*4)
		r, err := NewResizer(&ResizerConfig{Input: 16, Output: 32, Interlaced: interlaced}, Bicubic())
		expect(t, err, nil)
		r.Resize(dst, src, 16, 4, 32, 16)
		if interlaced && !bytes.Equal(dst, want) {
			t.Error("interlaced horizontal resize differs from progressive")
		}
		want = dst
	}
}

// TestNewResizer checks the configurations NewResizer rejects.
func TestNewResizer(t *testing.T) {
	tooTall := math.MaxInt32
	tooTall++
	huge := math.MaxInt32
	huge <<= 19
	for _, tt := range []struct {
		cfg ResizerConfig
		err string
	}{
		{ResizerConfig{Input: 2, Output: 1}, ""},
		{ResizerConfig{Input: 4, Output: 1, Vertical: true, Interlaced: true}, ""},
		{ResizerConfig{Input: 1, Output: 8}, "too small"},
		{ResizerConfig{Input: 8, Output: 0}, "too small"},
		{ResizerConfig{Input: 3, Output: 8, Vertical: true, Interlaced: true}, "too small"},
		// rows beyond int32 byte offsets, checked before any allocation
		{ResizerConfig{Input: math.MaxInt32/8 + 1, Output: 2, Pack: 4, Depth: 16}, "too large"},
		{ResizerConfig{Input: math.MaxInt32/4 + 1, Output: 2, Pack: 4}, "too large"},
		{ResizerConfig{Input: tooTall, Output: 2, Vertical: true}, "too large"},
		{ResizerConfig{Input: 8, Output: tooTall, Vertical: true}, "too large"},
		{ResizerConfig{Input: 8, Output: math.MaxInt32/4 + 1, Pack: 4}, "too large"},
		{ResizerConfig{Input: 8, Output: huge, Vertical: true}, "too large"},
		{ResizerConfig{Input: 8, Output: huge}, "too large"},
		// depths and packs of Descriptor.Check, 0 for the defaults, and
		// packs above 4
		{ResizerConfig{Input: 8, Output: 16, Depth: 8, Pack: 1}, ""},
		{ResizerConfig{Input: 8, Output: 16, Depth: 16, Pack: 4}, ""},
		{ResizerConfig{Input: 8, Output: 16, Pack: 7}, ""},
		{ResizerConfig{Input: 8, Output: 16, Pack: 7, Vertical: true}, ""},
		{ResizerConfig{Input: 8, Output: 16, Depth: 12}, "depth 12"},
		{ResizerConfig{Input: 8, Output: 16, Depth: 32, Vertical: true}, "depth 32"},
		{ResizerConfig{Input: 8, Output: 16, Depth: -8}, "depth -8"},
		{ResizerConfig{Input: 8, Output: 16, Depth: 1}, "depth 1"},
		{ResizerConfig{Input: 8, Output: 16, Pack: -1}, "pack -1"},
		{ResizerConfig{Input: 8, Output: 16, Pack: -1, Vertical: true}, "pack -1"},
		// bytes per pixel overflowing int, and int32 offsets
		{ResizerConfig{Input: 8, Output: 16, Pack: math.MaxInt}, "too large"},
		{ResizerConfig{Input: 8, Output: 16, Pack: math.MaxInt, Depth: 16, Vertical: true}, "too large"},
		{ResizerConfig{Input: 8, Output: 16, Pack: math.MaxInt/2 + 1, Depth: 16, Vertical: true}, "too large"},
		{ResizerConfig{Input: 8, Output: 16, Pack: math.MaxInt32/2 + 1, Depth: 16, Vertical: true}, "too large"},
		{ResizerConfig{Input: 8, Output: 16, Pack: math.MaxInt32 / 8}, "too large"},
		// horizontal kernels of p*p coefficients per tap beyond int32
		{ResizerConfig{Input: 2, Output: 2, Pack: 1 << 29}, "too large"},
		{ResizerConfig{Input: 2, Output: 2, Pack: 1 << 29, Vertical: true}, ""},
		{ResizerConfig{Input: 2, Output: 2, Pack: 1 << 15}, "too large"},
		{ResizerConfig{Input: 2, Output: 2, Pack: 1 << 11}, ""},
	} {
		if tt.cfg.Input < 0 || tt.cfg.Output < 0 {
			// 32-bit int
			continue
		}
		r, err := NewResizer(&tt.cfg, Bilinear())
		if tt.err == "" {
			expect(t, err, nil)
			expect(t, r != nil, true)
			continue
		}
		if r != nil || !errors.Is(err, ErrTooSmall) && !errors.Is(err, ErrTooLarge) && !errors.Is(err, ErrInvalidDescriptor) || !strings.Contains(err.Error(), tt.err) {
			t.Errorf("%+v: got %v, want %q", tt.cfg, err, tt.err)
		}
	}
	if huge > 0 {
		// NewConverter builds its resizers in goroutines
		g8 := Descriptor{Rect: image.Rect(0, 0, 8, 8), Ratio: Ratio444, Pack: 1, Planes: 1}
		for _, r := range []image.Rectangle{image.Rect(0, 0, huge, 8), image.Rect(0, 0, 8, huge)} {
			_, err := NewConverter(&ConverterConfig{Input: g8, Output: Descriptor{Rect: r, Ratio: Ratio444, Pack: 1, Planes: 1}}, Bilinear())
			if !errors.Is(err, ErrTooLarge) {
				t.Errorf("converter to %v: got %v, want ErrTooLarge", r, err)
			}
		}
	}
}

// TestResizePanics checks that Resize panics on planes, pitches and slices that
// do not match its configuration, and accepts slices of exactly the documented
// sizes.
func TestResizePanics(t *testing.T) {
	panics := func(f func()) (msg string) {
		defer func() {
			if r := recover(); r != nil {
				msg = fmt.Sprint(r)
			}
		}()
		f()
		return ""
	}
	for _, tt := range []struct {
		vertical, interlaced bool
		pack, depth          int
	}{
		{false, false, 1, 8}, {true, false, 1, 8}, {true, true, 1, 8}, {false, false, 4, 16}, {true, true, 4, 16},
	} {
		cfg := ResizerConfig{Input: 16, Output: 24, Vertical: tt.vertical, Interlaced: tt.interlaced, Pack: tt.pack, Depth: tt.depth}
		r, err := NewResizer(&cfg, Bicubic())
		expect(t, err, nil)
		bpp := tt.pack * tt.depth / 8
		width, height, columns, rows := 16, 5, 24, 5
		if tt.vertical {
			width, height, columns, rows = 5, 16, 5, 24
		}
		sp, dp := width*bpp+3, columns*bpp+7
		src := make([]byte, sp*(height-1)+width*bpp)
		dst := make([]byte, dp*(rows-1)+columns*bpp)
		expect(t, panics(func() { r.Resize(dst, src, width, height, dp, sp) }), "")
		// the resized dimension off by one, and the kept one zero
		wrongWidth, wrongHeight, zeroWidth, zeroHeight := width+1, height, width, 0
		if tt.vertical {
			wrongWidth, wrongHeight, zeroWidth, zeroHeight = width, height+1, 0, height
		}
		for _, c := range []struct {
			name          string
			dst, src      []byte
			width, height int
			dp, sp        int
		}{
			{"short source", dst, src[:len(src)-1], width, height, dp, sp},
			{"short destination", dst[:len(dst)-1], src, width, height, dp, sp},
			{"wrong input", dst, src, wrongWidth, wrongHeight, dp, sp},
			{"zero size", dst, src, zeroWidth, zeroHeight, dp, sp},
			// a negative pitch passes the size check but reads before
			// the slice, a huge one overflows it
			{"negative source pitch", dst, src, width, height, dp, -sp},
			{"negative destination pitch", dst, src, width, height, -dp, sp},
			{"huge source pitch", dst, src, width, height, dp, math.MaxInt / 2},
			{"huge destination pitch", dst, src, width, height, math.MaxInt / 2, sp},
			{"overlapping destination rows", dst, src, width, height, columns*bpp - 1, sp},
			{"zero destination pitch", dst, src, width, height, 0, sp},
		} {
			if msg := panics(func() { r.Resize(c.dst, c.src, c.width, c.height, c.dp, c.sp) }); !strings.HasPrefix(msg, "arez: resize") {
				t.Errorf("%+v %v: got %q, want an arez: resize panic", tt, c.name, msg)
			}
		}
	}
	// a zero source pitch is one row repeated
	r, err := NewResizer(&ResizerConfig{Input: 4, Output: 2, Vertical: true}, Bilinear())
	expect(t, err, nil)
	dst := make([]byte, 6)
	r.Resize(dst, []byte{1, 2, 3}, 3, 4, 3, 0)
	expect(t, [6]byte(dst), [6]byte{1, 2, 3, 1, 2, 3})
	// an interlaced single row has an empty bottom field, starting past
	// the end of a destination of one row shorter than its pitch
	r, err = NewResizer(&ResizerConfig{Input: 4, Output: 1, Vertical: true, Interlaced: true}, Bilinear())
	expect(t, err, nil)
	dst = make([]byte, 3)
	r.Resize(dst, []byte{10, 20, 30, 0, 200, 200, 200, 0, 10, 20, 30, 0, 200, 200, 200}, 3, 4, 8, 4)
	expect(t, [3]byte(dst), [3]byte{10, 20, 30})
}

func TestResizeOverlappingDestination(t *testing.T) {
	for _, tt := range []struct {
		vertical, disableSIMD bool
		dst, width            int
	}{
		{false, false, 8, 16}, {true, false, 64, 64}, {false, true, 8, 16}, {true, true, 64, 64},
	} {
		r, err := NewResizer(&ResizerConfig{Input: 16, Output: 8, Vertical: tt.vertical, Threads: 4, DisableSIMD: tt.disableSIMD}, Bilinear())
		expect(t, err, nil)
		msg := func() (msg string) {
			defer func() {
				msg = fmt.Sprint(recover())
			}()
			r.Resize(make([]byte, tt.dst), make([]byte, tt.width*16), tt.width, 16, 0, tt.width)
			return
		}()
		if !strings.HasPrefix(msg, "arez: resize destination with overlapping rows") {
			t.Errorf("%+v: got %q, want an overlapping rows panic", tt, msg)
		}
	}
	// a single destination row needs no pitch
	r, err := NewResizer(&ResizerConfig{Input: 4, Output: 1, Vertical: true, Threads: 4}, Bilinear())
	expect(t, err, nil)
	dst := make([]byte, 3)
	r.Resize(dst, []byte{1, 2, 3}, 3, 4, 0, 0)
	expect(t, [3]byte(dst), [3]byte{1, 2, 3})
}

// TestResizeOverlappingSourceAllocs checks that the allocations of a horizontal
// resize do not grow with the height when the source rows overlap, every row
// then ending before the reach of the kernels.
func TestResizeOverlappingSourceAllocs(t *testing.T) {
	for _, tt := range []struct {
		pack, depth int
	}{
		{1, 8}, {4, 8}, {1, 16}, {4, 16},
	} {
		r, err := NewResizer(&ResizerConfig{Input: 64, Output: 32, Pack: tt.pack, Depth: tt.depth, Threads: 1}, Bicubic())
		expect(t, err, nil)
		bpp := tt.pack * tt.depth / 8
		src := make([]byte, 64*bpp)
		allocs := func(height int) float64 {
			dst := make([]byte, 32*bpp*height)
			return testing.AllocsPerRun(10, func() { r.Resize(dst, src, 64, height, 32*bpp, 0) })
		}
		if few, many := allocs(4), allocs(1000); many > few {
			t.Errorf("%+v: %v allocations for 1000 rows, %v for 4", tt, many, few)
		}
	}
}

// TestReducingGap checks that resizes with a reducing gap are a box average to
// the size the gap gives followed by the filter, or a single resize when they
// do not reduce, the same with and without SIMD.
func TestReducingGap(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	sizes := []struct{ input, output int }{{1000, 37}, {640, 100}, {99, 10}, {64, 3}, {4000, 1}, {50, 30}, {17, 8}}
	for _, s := range sizes {
		for _, gap := range []float64{1, 2, 2.5, 3, math.Inf(1)} {
			for _, filter := range []Filter{Bilinear(), Lanczos(3), Box(), Nearest(), CustomBicubic(1.0/3, 1.0/3)} {
				for _, mode := range []struct {
					vertical, interlaced bool
					pack, depth          int
				}{{false, false, 1, 8}, {false, false, 4, 8}, {false, false, 3, 16}, {true, false, 2, 8}, {true, true, 1, 16}} {
					cfg := ResizerConfig{Input: s.input, Output: s.output, Vertical: mode.vertical, Interlaced: mode.interlaced, Pack: mode.pack, Depth: mode.depth, Threads: 3}
					what := fmt.Sprintf("%v gap=%v %+v", filter, gap, cfg)
					field := bin(mode.interlaced)
					// the size of the box average, if any
					mid := 0
					if n := int(float64(s.input) / float64(s.output) / gap); n >= 2 && (s.input+n-1)/n>>field >= 2 && filter != Nearest() && filter != Box() {
						mid = (s.input + n - 1) / n
					}
					bpp := mode.pack * mode.depth / 8
					rows := 5
					width, height, dw, dh, mw, mh := s.input, rows, s.output, rows, mid, rows
					if mode.vertical {
						width, height, dw, dh, mw, mh = rows, s.input, rows, s.output, rows, mid
					}
					src := make([]byte, width*height*bpp)
					rng.Read(src)
					want := make([]byte, dw*dh*bpp)
					if mid == 0 {
						r, err := NewResizer(&cfg, filter)
						expect(t, err, nil)
						r.Resize(want, src, width, height, dw*bpp, width*bpp)
					} else {
						step := cfg
						step.Output = mid
						r, err := NewResizer(&step, Box())
						expect(t, err, nil)
						buf := make([]byte, mw*mh*bpp)
						r.Resize(buf, src, width, height, mw*bpp, width*bpp)
						step.Input, step.Output = mid, s.output
						r, err = NewResizer(&step, filter)
						expect(t, err, nil)
						r.Resize(want, buf, mw, mh, dw*bpp, mw*bpp)
					}
					for _, simd := range []bool{true, false} {
						c := cfg
						c.ReducingGap = gap
						c.DisableSIMD = !simd
						r, err := NewResizer(&c, filter)
						expect(t, err, nil)
						expect(t, r.reduce != nil, mid != 0)
						got := make([]byte, dw*dh*bpp)
						r.Resize(got, src, width, height, dw*bpp, width*bpp)
						if !bytes.Equal(got, want) {
							t.Fatalf("%v simd=%v: output differs from the two steps", what, simd)
						}
					}
				}
			}
		}
	}
	for _, gap := range []float64{0.5, -1, math.NaN(), math.Inf(-1)} {
		_, err := NewResizer(&ResizerConfig{Input: 100, Output: 10, ReducingGap: gap}, Bilinear())
		if !errors.Is(err, ErrInvalidDescriptor) {
			t.Errorf("gap=%v NewResizer: got %v, want ErrInvalidDescriptor", gap, err)
		}
		cfg := &ConverterConfig{
			Input:       Descriptor{Rect: image.Rect(0, 0, 8, 8), Ratio: Ratio444, Pack: 1, Planes: 1},
			Output:      Descriptor{Rect: image.Rect(0, 0, 8, 8), Ratio: Ratio444, Pack: 1, Planes: 1},
			ReducingGap: gap,
		}
		if _, err := NewConverter(cfg, Bilinear()); !errors.Is(err, ErrInvalidDescriptor) {
			t.Errorf("gap=%v NewConverter: got %v, want ErrInvalidDescriptor", gap, err)
		}
	}
}

// TestReducingGapConvert checks the quality of conversions with a reducing gap
// against single resizes, and concurrent conversions sharing the buffers.
func TestReducingGapConvert(t *testing.T) {
	yuv := readImage(t, "testdata/lenna.jpg")
	for _, src := range []image.Image{yuv, toRGB(t, yuv)} {
		for _, tt := range []struct {
			size image.Point
			gap  float64
			psnr float64
		}{{image.Pt(64, 64), 2, 45}, {image.Pt(64, 64), 3, 47}, {image.Pt(37, 100), 2, 45}, {image.Pt(20, 20), 1, 30}} {
			ctor := func(r image.Rectangle) image.Image { return image.NewRGBA(r) }
			if _, ok := src.(*image.YCbCr); ok {
				ctor = func(r image.Rectangle) image.Image { return image.NewYCbCr(r, image.YCbCrSubsampleRatio420) }
			}
			ref := newImage(t, ctor, image.Rectangle{Max: tt.size})
			convert(t, ref, src, true, false, Lanczos(3))
			var outs [][]byte
			for _, simd := range []bool{true, false} {
				dst := newImage(t, ctor, image.Rectangle{Max: tt.size})
				c := prepare(t, dst, src, simd, false, Lanczos(3), 0, func(cfg *ConverterConfig) { cfg.ReducingGap = tt.gap })
				dsts := make([]image.Image, 4)
				var wg sync.WaitGroup
				for i := range dsts {
					dsts[i] = newImage(t, ctor, image.Rectangle{Max: tt.size})
					wg.Go(func() {
						if err := c.Convert(dsts[i], src); err != nil {
							t.Error(err)
						}
					})
				}
				wg.Wait()
				for _, dst := range dsts {
					outs = append(outs, imageBytes(dst))
				}
				psnrs := psnr(t, ref, dsts[0])
				t.Logf("%T to %v gap=%v simd=%v: psnr %.1f", src, tt.size, tt.gap, simd, psnrs)
				for i, v := range psnrs {
					if v < tt.psnr {
						t.Errorf("%T to %v gap=%v simd=%v: plane %v psnr %.1f, want at least %v", src, tt.size, tt.gap, simd, i, v, tt.psnr)
					}
				}
			}
			for _, out := range outs[1:] {
				if !bytes.Equal(out, outs[0]) {
					t.Errorf("%T to %v gap=%v: outputs differ", src, tt.size, tt.gap)
				}
			}
		}
	}
}
