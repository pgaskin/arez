// Copyright 2026 Patrick Gaskin, 2013 Benoît Amiaux. All rights reserved. Use
// of this source code is governed by a MIT-style license that can be found in
// the LICENSE file.

package arez

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"image/color"
	"math"
	"math/rand"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"
)

// note: claude helped write a lot of these test cases from short descriptions
// of each

func TestConvertYCbCr(t *testing.T) { testConvertWith(t, false) }
func TestConvertRGB(t *testing.T)   { testConvertWith(t, true) }

type TestCase struct {
	file       string
	src        image.Rectangle
	dst        image.Rectangle
	rgb        bool
	interlaced bool
	filter     Filter
	threads    int
	psnrs      []float64
	psnrRect   image.Rectangle
	dump       string
}

func TestCopy(t *testing.T) {
	tc := NewTestCase(512, 512, false)
	tc.psnrs = []float64{math.Inf(1), math.Inf(1), math.Inf(1)}
	runTestCase(t, tc, 1)
	tc = NewTestCase(512, 512, false)
	tc.rgb = true
	tc.psnrs = []float64{math.Inf(1)}
	runTestCase(t, tc, 1)
}
func TestDegradations(t *testing.T) {
	interlaced := []bool{false, true}
	rgb := []bool{false, true}
	for _, f := range filters {
		for _, ii := range interlaced {
			offset := 1
			if ii {
				offset = 2
			}
			for _, rgb := range rgb {
				tc := NewTestCase(256+offset, 256+offset, ii)
				tc.filter = f
				tc.rgb = rgb
				tc.psnrs = []float64{22, 35, 35}
				runTestCase(t, tc, 32)
			}
		}
	}
}
func TestTooManyThreads(t *testing.T) {
	sizes := []struct{ w, h int }{{128, 16}, {16, 128}, {16, 16}}
	interlaced := []bool{false, true}
	for _, s := range sizes {
		for _, ii := range interlaced {
			tc := NewTestCase(s.w, s.h, ii)
			tc.threads = 32
			runTestCase(t, tc, 1)
		}
	}
}
func TestSmallSizes(t *testing.T) {
	if testing.Short() {
		return
	}
	interlaced := []bool{false, true}
	// we need at least 2 taps per pixel, so 4 pixels in 4:2:0
	for w := 4; w < 24; w++ {
		for _, ii := range interlaced {
			// we need at least 2 taps per field/pixel
			// so 4/8 pixels with 4:2:0 progressive/interlaced
			for h := 4 + int(bin(ii))*4; h < 24; h++ {
				tc := NewTestCase(w, h, ii)
				tc.src = image.Rect(0, 0, 32, 32)
				runTestCase(t, tc, 1)
			}
		}
	}
}
func TestSaturatedRightBorder(t *testing.T) {
	tc := NewTestCase(171, 500, false)
	tc.file = "bug3img.jpg"
	tc.rgb = true
	tc.psnrs = []float64{16}
	tc.psnrRect = image.Rect(280, 0, 286, 500)
	runTestCase(t, tc, 1)
}
func TestGrayPlanes(t *testing.T) {
	src := readImage(t, "testdata/gray.png")
	testPlanes(t, src, func(r image.Rectangle) image.Image { return image.NewGray(r) }, []float64{35})
}
func TestAlphaPlanes(t *testing.T) {
	gray := readImage(t, "testdata/gray.png").(*image.Gray)
	src := &image.Alpha{Pix: gray.Pix, Stride: gray.Stride, Rect: gray.Rect}
	testPlanes(t, src, func(r image.Rectangle) image.Image { return image.NewAlpha(r) }, []float64{35})
}
func TestNRGBAPlanes(t *testing.T) {
	src := readImage(t, "testdata/nrgba.png")
	testPlanes(t, src, func(r image.Rectangle) image.Image { return image.NewNRGBA(r) }, []float64{37})
}
func TestNYCbCrAPlanes(t *testing.T) {
	yuv := readImage(t, "testdata/lenna.jpg").(*image.YCbCr)
	src := &image.NYCbCrA{YCbCr: *yuv, A: make([]byte, len(yuv.Y)), AStride: yuv.YStride}
	for i, v := range yuv.Y {
		src.A[i] = 255 - v
	}
	guardImage(t, src)
	testPlanes(t, src, func(r image.Rectangle) image.Image { return image.NewNYCbCrA(r, yuv.SubsampleRatio) }, []float64{30, 39, 39, 30})
}
func TestCMYKPlanes(t *testing.T) {
	src := drawInto(t, readImage(t, "testdata/lenna.jpg"), image.NewCMYK)
	testPlanes(t, src, func(r image.Rectangle) image.Image { return image.NewCMYK(r) }, []float64{31})
}
func TestGray16Planes(t *testing.T) {
	src := drawInto(t, readImage(t, "testdata/lenna.jpg"), image.NewGray16)
	testPlanes(t, src, func(r image.Rectangle) image.Image { return image.NewGray16(r) }, []float64{30})
}
func TestAlpha16Planes(t *testing.T) {
	gray := drawInto(t, readImage(t, "testdata/lenna.jpg"), image.NewGray16)
	src := &image.Alpha16{Pix: gray.Pix, Stride: gray.Stride, Rect: gray.Rect}
	testPlanes(t, src, func(r image.Rectangle) image.Image { return image.NewAlpha16(r) }, []float64{30})
}
func TestRGBA64Planes(t *testing.T) {
	src := drawInto(t, readImage(t, "testdata/lenna.jpg"), image.NewRGBA64)
	testPlanes(t, src, func(r image.Rectangle) image.Image { return image.NewRGBA64(r) }, []float64{30})
}
func TestNRGBA64Planes(t *testing.T) {
	src := drawInto(t, readImage(t, "testdata/nrgba.png"), image.NewNRGBA64)
	testPlanes(t, src, func(r image.Rectangle) image.Image { return image.NewNRGBA64(r) }, []float64{37})
}
func TestUnsupported(t *testing.T) {
	r := image.Rect(0, 0, 64, 64)
	gray := newImage(t, image.NewGray, r)
	for _, img := range []image.Image{
		newImage(t, paletted, r),
		image.NewUniform(color.Gray{}),
		r,
	} {
		_, err := Prepare(gray, img)
		if !errors.Is(err, ErrUnsupportedImage) {
			t.Errorf("%T: got %v, want ErrUnsupportedImage", img, err)
		}
	}
	_, err := Prepare(gray, newImage(t, image.NewGray16, r))
	if !errors.Is(err, ErrIncompatible) || !strings.Contains(err.Error(), "16-bit") {
		t.Errorf("Gray16 to Gray: got %v, want a depth error", err)
	}
}

// TestMixedTypes checks that conversions between images of different types
// of the same layout, whose samples mean different things, are rejected.
func TestMixedTypes(t *testing.T) {
	r := image.Rect(0, 0, 8, 8)
	big := image.Rect(0, 0, 16, 16)
	ctors := map[string]func(image.Rectangle) image.Image{}
	for _, it := range testImageTypes {
		ctors[it.name] = it.new
	}
	for _, pair := range [][2]string{
		{"RGBA", "NRGBA"}, {"RGBA", "CMYK"}, {"NRGBA", "CMYK"}, {"Gray", "Alpha"},
		{"RGBA64", "NRGBA64"}, {"Gray16", "Alpha16"},
	} {
		for _, names := range [][2]string{pair, {pair[1], pair[0]}} {
			dstNew, srcNew := ctors[names[0]], ctors[names[1]]
			dst, src := newImage(t, dstNew, big), newImage(t, srcNew, r)
			what := fmt.Sprintf("%T from %T", dst, src)
			if _, err := Prepare(dst, src); !errors.Is(err, ErrIncompatible) {
				t.Errorf("Prepare %v: got %v, want ErrIncompatible", what, err)
			}
			if err := Convert(dst, src, Bilinear()); !errors.Is(err, ErrIncompatible) {
				t.Errorf("Convert %v: got %v, want ErrIncompatible", what, err)
			}
			if err := Convert(newImage(t, dstNew, r), src, Bilinear()); !errors.Is(err, ErrIncompatible) {
				t.Errorf("Convert copy %v: got %v, want ErrIncompatible", what, err)
			}
			// converters prepared for either type, which descriptors describe
			for _, ctor := range []func(image.Rectangle) image.Image{dstNew, srcNew} {
				cfg, err := Prepare(newImage(t, ctor, big), newImage(t, ctor, r))
				expect(t, err, nil)
				c, err := NewConverter(cfg, Bilinear())
				expect(t, err, nil)
				if err := c.Convert(dst, src); !errors.Is(err, ErrIncompatible) {
					t.Errorf("Converter.Convert %v: got %v, want ErrIncompatible", what, err)
				}
			}
		}
	}
}

func TestOverlappingRows(t *testing.T) {
	r := image.Rect(0, 0, 8, 8)
	gray := func(stride, h int) *image.Gray {
		return &image.Gray{Pix: make([]byte, 64), Stride: stride, Rect: image.Rect(0, 0, 8, h)}
	}
	overlapping := func(f func(*image.YCbCr), ctor func(image.Rectangle) image.Image) image.Image {
		img := newImage(t, ctor, r)
		if a, ok := img.(*image.NYCbCrA); ok {
			f(&a.YCbCr)
			a.AStride = 4
			return a
		}
		f(img.(*image.YCbCr))
		return img
	}
	yuv := func(r image.Rectangle) image.Image { return image.NewYCbCr(r, image.YCbCrSubsampleRatio420) }
	yuva := func(r image.Rectangle) image.Image { return image.NewNYCbCrA(r, image.YCbCrSubsampleRatio420) }
	for _, tt := range []struct {
		name     string
		dst, src image.Image
	}{
		{"gray", gray(4, 8), newImage(t, image.NewGray, image.Rect(0, 0, 4, 4))},
		{"gray copy", gray(7, 8), newImage(t, image.NewGray, r)},
		{"gray zero stride", gray(0, 8), newImage(t, image.NewGray, image.Rect(0, 0, 4, 4))},
		{"rgba", &image.RGBA{Pix: make([]byte, 256), Stride: 16, Rect: r}, newImage(t, image.NewRGBA, image.Rect(0, 0, 4, 4))},
		{"luma", overlapping(func(img *image.YCbCr) { img.YStride = 4 }, yuv), newImage(t, yuv, image.Rect(0, 0, 4, 4))},
		{"chroma", overlapping(func(img *image.YCbCr) { img.CStride = 2 }, yuv), newImage(t, yuv, image.Rect(0, 0, 4, 4))},
		{"alpha", overlapping(func(*image.YCbCr) {}, yuva), newImage(t, yuva, image.Rect(0, 0, 4, 4))},
	} {
		if _, err := Prepare(tt.dst, tt.src); !errors.Is(err, ErrUnsupportedImage) {
			t.Errorf("%v Prepare: got %v, want ErrUnsupportedImage", tt.name, err)
		}
		if err := Convert(tt.dst, tt.src, Bilinear()); !errors.Is(err, ErrUnsupportedImage) {
			t.Errorf("%v Convert: got %v, want ErrUnsupportedImage", tt.name, err)
		}
		cfg, err := Prepare(tt.src, tt.src)
		expect(t, err, nil)
		cfg.Output.Rect = tt.dst.Bounds()
		c, err := NewConverter(cfg, Bilinear())
		expect(t, err, nil)
		if err := c.Convert(tt.dst, tt.src); !errors.Is(err, ErrUnsupportedImage) {
			t.Errorf("%v Converter.Convert: got %v, want ErrUnsupportedImage", tt.name, err)
		}
	}
	expect(t, Convert(newImage(t, image.NewGray, image.Rect(0, 0, 16, 16)), gray(4, 8), Bilinear()), nil)
	expect(t, Convert(newImage(t, image.NewGray, r), gray(0, 8), Bilinear()), nil)
	expect(t, Convert(gray(0, 1), gray(8, 8), Bilinear()), nil)
}

func TestNilImages(t *testing.T) {
	r := image.Rect(0, 0, 8, 8)
	catch := func(what string, f func() error) {
		t.Helper()
		defer func() {
			if v := recover(); v != nil {
				t.Errorf("%v: panic: %v", what, v)
			}
		}()
		if err := f(); !errors.Is(err, ErrUnsupportedImage) {
			t.Errorf("%v: got %v, want ErrUnsupportedImage", what, err)
		}
	}
	types := append(testImageTypes, struct {
		name string
		new  func(image.Rectangle) image.Image
	}{"Paletted", func(r image.Rectangle) image.Image { return paletted(r) }})
	for _, it := range types {
		img := newImage(t, it.new, r)
		null := reflect.Zero(reflect.TypeOf(img)).Interface().(image.Image)
		catch(it.name+" Prepare input", func() error { _, err := Prepare(img, null); return err })
		catch(it.name+" Prepare output", func() error { _, err := Prepare(null, img); return err })
		catch(it.name+" Convert input", func() error { return Convert(img, null, Bilinear()) })
		catch(it.name+" Convert output", func() error { return Convert(null, img, Bilinear()) })
		cfg, err := Prepare(img, img)
		if err != nil {
			continue
		}
		c, err := NewConverter(cfg, Bilinear())
		expect(t, err, nil)
		catch(it.name+" Converter.Convert input", func() error { return c.Convert(img, null) })
		catch(it.name+" Converter.Convert output", func() error { return c.Convert(null, img) })
	}
	catch("untyped nil", func() error { return Convert(newImage(t, image.NewGray, r), nil, Bilinear()) })
}

func TestChromaRatioOf(t *testing.T) {
	for want, r := range []image.YCbCrSubsampleRatio{
		image.YCbCrSubsampleRatio410,
		image.YCbCrSubsampleRatio411,
		image.YCbCrSubsampleRatio420,
		image.YCbCrSubsampleRatio422,
		image.YCbCrSubsampleRatio440,
		image.YCbCrSubsampleRatio444,
	} {
		got, ok := ChromaRatioOf(r)
		expect(t, ok, true)
		expect(t, got, ChromaRatio(want))
	}
	_, ok := ChromaRatioOf(image.YCbCrSubsampleRatio(99))
	expect(t, ok, false)
	img := newImage(t, ycbcr(image.YCbCrSubsampleRatio(99)), image.Rect(0, 0, 8, 8))
	_, err := Prepare(img, img)
	if !errors.Is(err, ErrUnsupportedImage) || !strings.Contains(err.Error(), "ratio") {
		t.Errorf("unknown ratio: got %v, want a ratio error", err)
	}
}

func TestBigKernels(t *testing.T) {
	interlaced := []bool{false, true}
	for _, ii := range interlaced {
		tc := NewTestCase(256, 256, ii)
		tc.src = image.Rect(0, 0, 32, 32)
		tc.filter = Lanczos(128)
		runTestCase(t, tc, 1)
	}
}

func TestLanczos3(t *testing.T) {
	srcs := []image.Rectangle{
		{},
		image.Rect(0, 0, 100, 100),
		image.Rect(7, 9, 7+301, 9+203),
		image.Rect(0, 0, 33, 512),
	}
	dsts := []struct{ w, h int }{
		{1920, 1080},
		{640, 480},
		{333, 777},
		{1023, 18},
		{64, 64},
		{129, 257},
		{17, 2000},
		{2000, 20},
	}
	for _, src := range srcs {
		for _, d := range dsts {
			for _, ii := range []bool{false, true} {
				for _, rgb := range []bool{false, true} {
					tc := NewTestCase(d.w, d.h, ii)
					tc.src = src
					tc.rgb = rgb
					tc.filter = Lanczos(3)
					runTestCase(t, tc, 1)
				}
			}
		}
	}
}

// TestLimits checks the sizes NewConverter rejects, input fields of less
// than 2 rows, that outputs of one pixel per plane convert, and that tall
// images convert.
func TestLimits(t *testing.T) {
	gray := func(w, h int) image.Image { return newImage(t, image.NewGray, image.Rect(0, 0, w, h)) }
	yuv := func(w, h int) image.Image {
		return newImage(t, ycbcr(image.YCbCrSubsampleRatio420), image.Rect(0, 0, w, h))
	}
	tests := []struct {
		src, dst   image.Image
		interlaced bool
		err        string
	}{
		{gray(8, 2), gray(8, 8), true, "too small"},
		{gray(8, 3), gray(8, 8), true, "too small"},
		{gray(8, 4), gray(8, 8), true, ""},
		{yuv(8, 4), yuv(8, 8), true, "too small"},
		{yuv(8, 8), yuv(8, 16), true, ""},
		{gray(1, 8), gray(2, 8), false, "too small"},
		{gray(8, 8), gray(1, 8), false, ""},
		{gray(8, 8), gray(8, 1), true, ""},
		{yuv(8, 8), yuv(2, 2), true, ""},
		{yuv(8, 8), yuv(1, 1), false, ""},
	}
	for _, tt := range tests {
		cfg, err := Prepare(tt.dst, tt.src)
		expect(t, err, nil)
		cfg.Input.Interlaced, cfg.Output.Interlaced = tt.interlaced, tt.interlaced
		c, err := NewConverter(cfg, Bilinear())
		if tt.err == "" {
			expect(t, err, nil)
			expect(t, c.Convert(tt.dst, tt.src), nil)
			continue
		}
		if !errors.Is(err, ErrTooSmall) || !strings.Contains(err.Error(), tt.err) {
			t.Errorf("%v %v interlaced=%v: got %v, want %q", tt.src.Bounds(), tt.dst.Bounds(), tt.interlaced, err, tt.err)
		}
	}
	// a tall image, whose vertical offsets are large before the conversion
	// to steps
	src := gray(2, 40000).(*image.Gray)
	for y := range 40000 {
		src.Pix[y*src.Stride], src.Pix[y*src.Stride+1] = byte(y%251), byte(y%251)
	}
	dst := gray(2, 20000).(*image.Gray)
	expect(t, Convert(dst, src, Bilinear()), nil)
	// output row y averages source rows 2y-1 to 2y+2
	for _, y := range []int{1, 10000, 19998} {
		want := 0
		for r := 2*y - 1; r <= 2*y+2; r++ {
			want += r % 251
		}
		if got := int(dst.Pix[y*dst.Stride]); got != want>>2 && got != (want+3)>>2 {
			t.Errorf("row %v: got %v, want about %v", y, got, want>>2)
		}
	}
}

// TestWideRows resizes rows wider than 32767 bytes, which the int16 offsets
// of earlier versions rejected, through the Go scalers and the accelerated
// ones: a horizontal ramp must come out as a ramp and both paths must agree
// bit for bit.
func TestWideRows(t *testing.T) {
	tests := []struct {
		name      string
		new       func(image.Rectangle) image.Image
		win, wout int
	}{
		{"Gray", func(r image.Rectangle) image.Image { return image.NewGray(r) }, 40000, 20000},
		{"Gray", func(r image.Rectangle) image.Image { return image.NewGray(r) }, 40000, 45000},
		{"RGBA", func(r image.Rectangle) image.Image { return image.NewRGBA(r) }, 9000, 4500},
		{"RGBA64", func(r image.Rectangle) image.Image { return image.NewRGBA64(r) }, 5000, 2500},
		{"Gray16", func(r image.Rectangle) image.Image { return image.NewGray16(r) }, 20000, 10000},
	}
	const h = 3
	for _, tt := range tests {
		src := newImage(t, tt.new, image.Rect(0, 0, tt.win, h))
		d, planes, err := inspect(src, false)
		expect(t, err, nil)
		p := planes[0]
		ramp := func(x, w int) int {
			if d.depth() == 16 {
				return x * 65535 / (w - 1)
			}
			return x * 255 / (w - 1)
		}
		for y := range h {
			for x := range tt.win {
				v := ramp(x, tt.win)
				for c := range d.Pack {
					if d.depth() == 16 {
						binary.BigEndian.PutUint16(p.Data[y*p.Pitch+(x*d.Pack+c)*2:], uint16(v))
					} else {
						p.Data[y*p.Pitch+x*d.Pack+c] = byte(v)
					}
				}
			}
		}
		var outs [][]byte
		for _, simd := range []bool{true, false} {
			if !simd && !hasSIMD() {
				break
			}
			dst := newImage(t, tt.new, image.Rect(0, 0, tt.wout, h))
			convert(t, dst, src, simd, false, Bilinear())
			_, planes, err := inspect(dst, false)
			expect(t, err, nil)
			p := planes[0]
			// output pixel x is centred on source pixel (x+0.5)*win/wout-0.5
			for _, x := range []int{0, 1, tt.wout / 3, tt.wout / 2, tt.wout - 2, tt.wout - 1} {
				sx := (float64(x)+0.5)*float64(tt.win)/float64(tt.wout) - 0.5
				want := min(max(sx, 0), float64(tt.win-1)) * float64(ramp(tt.win-1, tt.win)) / float64(tt.win-1)
				got := int(p.Data[x*d.Pack*d.depth()/8])
				if d.depth() == 16 {
					got = int(binary.BigEndian.Uint16(p.Data[x*d.Pack*2:]))
				}
				tol := 2.0
				if d.depth() == 16 {
					tol = 8
				}
				if math.Abs(float64(got)-want) > tol {
					t.Errorf("%v %v -> %v simd=%v: pixel %v is %v, want about %.1f", tt.name, tt.win, tt.wout, simd, x, got, want)
				}
			}
			outs = append(outs, imageBytes(dst))
		}
		if len(outs) == 2 && !bytes.Equal(outs[0], outs[1]) {
			t.Errorf("%v %v -> %v: simd differs from go", tt.name, tt.win, tt.wout)
		}
	}
}

// TestConverterConfigCopy checks that NewConverter leaves the configuration
// it is given alone and defaults the thread count in its own copy.
func TestConverterConfigCopy(t *testing.T) {
	for _, threads := range []int{0, -1} {
		cfg := ConverterConfig{
			Input:   Descriptor{Rect: image.Rect(0, 0, 50, 50), Ratio: Ratio444, Pack: 1, Planes: 1},
			Output:  Descriptor{Rect: image.Rect(0, 0, 100, 100), Ratio: Ratio444, Pack: 1, Planes: 1},
			Threads: threads,
		}
		want := cfg
		c, err := NewConverter(&cfg, Bicubic())
		expect(t, err, nil)
		expect(t, cfg, want)
		expect(t, c.cfg.Threads, runtime.GOMAXPROCS(0))
		expect(t, c.Convert(newImage(t, image.NewGray, image.Rect(0, 0, 100, 100)), newImage(t, image.NewGray, image.Rect(0, 0, 50, 50))), nil)
	}
}

// TestConvertConcurrent runs concurrent conversions on one converter, two
// pass resizes which share its intermediate planes without the pool, and
// compares them with a sequential one; run it with -race.
func TestConvertConcurrent(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, it := range []struct {
		name string
		new  func(image.Rectangle) image.Image
		opts []func(*ConverterConfig)
	}{
		{"YCbCr420", func(r image.Rectangle) image.Image { return image.NewYCbCr(r, image.YCbCrSubsampleRatio420) }, nil},
		{"RGBA", func(r image.Rectangle) image.Image { return image.NewRGBA(r) }, nil},
		{"Gray16", func(r image.Rectangle) image.Image { return image.NewGray16(r) }, nil},
		{"NRGBA premultiplied", func(r image.Rectangle) image.Image { return image.NewNRGBA(r) }, []func(*ConverterConfig){premultiplied}},
	} {
		src := newImage(t, it.new, image.Rect(0, 0, 320, 240))
		fillImage(rng, src)
		for _, simd := range []bool{true, false} {
			if !simd && !hasSIMD() {
				break
			}
			dst := newImage(t, it.new, image.Rect(0, 0, 200, 150))
			c := prepare(t, dst, src, simd, false, Lanczos(3), 2, it.opts...)
			expect(t, c.Convert(dst, src), nil)
			want := imageBytes(dst)
			var group sync.WaitGroup
			for range 8 {
				group.Go(func() {
					for range 4 {
						dst := newImage(t, it.new, image.Rect(0, 0, 200, 150))
						if err := c.Convert(dst, src); err != nil {
							t.Error(err)
							return
						}
						if !bytes.Equal(imageBytes(dst), want) {
							t.Errorf("%v simd=%v: concurrent conversion differs", it.name, simd)
							return
						}
					}
				})
			}
			group.Wait()
		}
	}
}

// TestConvertMismatch checks that a converter rejects images of other sizes
// or formats than the ones it was created for.
func TestConvertMismatch(t *testing.T) {
	gray := func(w, h int) image.Image { return newImage(t, image.NewGray, image.Rect(0, 0, w, h)) }
	yuv := func(w, h int, r image.YCbCrSubsampleRatio) image.Image {
		return newImage(t, ycbcr(r), image.Rect(0, 0, w, h))
	}
	c, err := NewConverter(&ConverterConfig{
		Input:  Descriptor{Rect: image.Rect(0, 0, 50, 50), Ratio: Ratio444, Pack: 1, Planes: 1},
		Output: Descriptor{Rect: image.Rect(0, 0, 100, 100), Ratio: Ratio444, Pack: 1, Planes: 1},
	}, Bicubic())
	expect(t, err, nil)
	expect(t, c.Convert(gray(100, 100), gray(50, 50)), nil)
	for _, tt := range []struct{ dst, src image.Image }{
		{gray(100, 100), gray(30, 30)},
		{gray(100, 100), gray(70, 50)},
		{gray(80, 100), gray(50, 50)},
		{gray(100, 100), newImage(t, image.NewGray16, image.Rect(0, 0, 50, 50))},
		{yuv(100, 100, image.YCbCrSubsampleRatio444), gray(50, 50)},
	} {
		err := c.Convert(tt.dst, tt.src)
		if !errors.Is(err, ErrMismatch) {
			t.Errorf("%T%v from %T%v: got %v, want ErrMismatch", tt.dst, tt.dst.Bounds(), tt.src, tt.src.Bounds(), err)
		}
	}
	src := yuv(50, 50, image.YCbCrSubsampleRatio420)
	c, err = NewConverter(&ConverterConfig{
		Input:  Descriptor{Rect: image.Rect(0, 0, 50, 50), Ratio: Ratio420, Pack: 1, Planes: 3},
		Output: Descriptor{Rect: image.Rect(0, 0, 100, 100), Ratio: Ratio420, Pack: 1, Planes: 3},
	}, Bicubic())
	expect(t, err, nil)
	expect(t, c.Convert(yuv(100, 100, image.YCbCrSubsampleRatio420), src), nil)
	err = c.Convert(yuv(100, 100, image.YCbCrSubsampleRatio444), src)
	if !errors.Is(err, ErrMismatch) {
		t.Errorf("4:4:4 output: got %v, want ErrMismatch", err)
	}
	// the chroma planes of an image at an odd x have one more column, at an
	// even origin the same size
	odd := newImage(t, ycbcr(image.YCbCrSubsampleRatio420), image.Rect(1, 0, 51, 50))
	err = c.Convert(yuv(100, 100, image.YCbCrSubsampleRatio420), odd)
	if !errors.Is(err, ErrMismatch) || !strings.Contains(err.Error(), "50x50 at (1,0)") {
		t.Errorf("odd origin: got %v, want ErrMismatch", err)
	}
	even := newImage(t, ycbcr(image.YCbCrSubsampleRatio420), image.Rect(-2, 2, 48, 52))
	expect(t, c.Convert(yuv(100, 100, image.YCbCrSubsampleRatio420), even), nil)
}

// TestSinglePlaneRatio converts packed images with converters from
// descriptors whose Ratio is not the Ratio444 Prepare gives them.
func TestSinglePlaneRatio(t *testing.T) {
	for _, it := range []struct {
		pack, depth int
		new         func(image.Rectangle) image.Image
	}{
		{1, 8, func(r image.Rectangle) image.Image { return image.NewGray(r) }},
		{4, 8, func(r image.Rectangle) image.Image { return image.NewRGBA(r) }},
		{4, 16, func(r image.Rectangle) image.Image { return image.NewNRGBA64(r) }},
	} {
		for _, ratio := range []ChromaRatio{Ratio410, Ratio420, Ratio444} {
			c, err := NewConverter(&ConverterConfig{
				Input:  Descriptor{Rect: image.Rect(0, 0, 8, 8), Pack: it.pack, Planes: 1, Depth: it.depth, Ratio: ratio},
				Output: Descriptor{Rect: image.Rect(0, 0, 16, 16), Pack: it.pack, Planes: 1, Depth: it.depth},
			}, Bilinear())
			expect(t, err, nil)
			dst, src := newImage(t, it.new, image.Rect(0, 0, 16, 16)), newImage(t, it.new, image.Rect(0, 0, 8, 8))
			if err := c.Convert(dst, src); err != nil {
				t.Errorf("%T with ratio %v: %v", src, ratio, err)
			}
		}
	}
}
func TestDescriptorCheck(t *testing.T) {
	for _, tt := range []struct {
		d   Descriptor
		err string
	}{
		{Descriptor{Rect: image.Rect(0, 0, 8, 8), Pack: 1, Planes: 1}, ""},
		{Descriptor{Rect: image.Rect(0, 0, 8, 8), Pack: 1, Planes: 4, Ratio: Ratio420}, ""},
		{Descriptor{Rect: image.Rect(0, 0, 8, 8), Pack: 0, Planes: 1}, "pack"},
		{Descriptor{Rect: image.Rect(0, 0, 8, 8), Pack: 5, Planes: 1}, "pack"},
		{Descriptor{Rect: image.Rect(0, 0, 8, 8), Pack: 1, Planes: 1, Depth: 12}, "depth"},
		{Descriptor{Rect: image.Rect(0, 0, 8, 8), Pack: 1, Planes: 0}, "planes"},
		{Descriptor{Rect: image.Rect(0, 0, 8, 8), Pack: 1, Planes: 5}, "planes"},
		{Descriptor{Rect: image.Rect(0, 0, 8, 8), Pack: 1, Planes: 3, Ratio: -1}, "ratio"},
		{Descriptor{Rect: image.Rect(0, 0, 8, 8), Pack: 1, Planes: 3, Ratio: Ratio444 + 1}, "ratio"},
		{Descriptor{Rect: image.Rect(0, 0, 8, 7), Pack: 1, Planes: 3, Ratio: Ratio420, Interlaced: true}, ""},
		{Descriptor{Rect: image.Rect(0, 0, 8, 8), Pack: 1, Planes: 3, Ratio: Ratio420, Interlaced: true}, ""},
	} {
		err := tt.d.Check()
		if tt.err == "" {
			expect(t, err, nil)
		} else if !errors.Is(err, ErrInvalidDescriptor) || !strings.Contains(err.Error(), tt.err) {
			t.Errorf("%+v: got %v, want %q", tt.d, err, tt.err)
		}
	}
}
func TestDescriptorString(t *testing.T) {
	for _, tt := range []struct {
		d    Descriptor
		want string
	}{
		{Descriptor{Rect: image.Rect(0, 0, 640, 480), Ratio: Ratio420, Pack: 1, Planes: 3, Interlaced: true}, "640x480 3 planes 4:2:0 1-packed 8-bit interlaced"},
		{Descriptor{Rect: image.Rect(0, 0, 8, 8), Ratio: Ratio444, Pack: 4, Planes: 1, Depth: 16}, "8x8 1 plane 4:4:4 4-packed 16-bit progressive"},
		{Descriptor{Rect: image.Rect(0, 0, 8, 8), Ratio: 7, Pack: 1, Planes: 1}, "8x8 1 plane ChromaRatio(7) 1-packed 8-bit progressive"},
		{Descriptor{Rect: image.Rect(-1, 0, 16, 16), Ratio: Ratio420, Pack: 1, Planes: 3}, "17x16 at (-1,0) 3 planes 4:2:0 1-packed 8-bit progressive"},
	} {
		expect(t, tt.d.String(), tt.want)
		expect(t, fmt.Sprint(&tt.d), tt.want)
	}
	c, err := NewConverter(&ConverterConfig{
		Input:  Descriptor{Rect: image.Rect(0, 0, 50, 50), Ratio: Ratio444, Pack: 1, Planes: 1},
		Output: Descriptor{Rect: image.Rect(0, 0, 100, 100), Ratio: Ratio444, Pack: 1, Planes: 1},
	}, Bicubic())
	expect(t, err, nil)
	err = c.Convert(newImage(t, image.NewGray, image.Rect(0, 0, 100, 100)), newImage(t, image.NewGray, image.Rect(0, 0, 30, 30)))
	if err == nil || !strings.Contains(err.Error(), "input image 30x30 1 plane 4:4:4 1-packed 8-bit progressive, configured 50x50") {
		t.Errorf("got %v, want the descriptor strings", err)
	}
}

// TestErrors checks that every error wraps its sentinel.
func TestErrors(t *testing.T) {
	gray := func(w, h int) image.Image { return newImage(t, image.NewGray, image.Rect(0, 0, w, h)) }
	cfg := func(src, dst Descriptor, interlaced bool) *ConverterConfig {
		src.Interlaced, dst.Interlaced = interlaced, interlaced
		return &ConverterConfig{Input: src, Output: dst}
	}
	g8 := Descriptor{Rect: image.Rect(0, 0, 8, 8), Ratio: Ratio444, Pack: 1, Planes: 1}
	g16 := Descriptor{Rect: image.Rect(0, 0, 8, 8), Ratio: Ratio444, Pack: 1, Planes: 1, Depth: 16}
	rgba := Descriptor{Rect: image.Rect(0, 0, 8, 8), Ratio: Ratio444, Pack: 4, Planes: 1}
	yuv := Descriptor{Rect: image.Rect(0, 0, 8, 8), Ratio: Ratio420, Pack: 1, Planes: 3}
	newConverter := func(cfg *ConverterConfig, f Filter) error {
		_, err := NewConverter(cfg, f)
		return err
	}
	for _, tt := range []struct {
		name string
		err  func() error
		want error
	}{
		{"paletted", func() error {
			_, err := Prepare(gray(8, 8), newImage(t, paletted, image.Rect(0, 0, 8, 8)))
			return err
		}, ErrUnsupportedImage},
		{"ratio", func() error {
			img := newImage(t, ycbcr(image.YCbCrSubsampleRatio(99)), image.Rect(0, 0, 8, 8))
			_, err := Prepare(img, img)
			return err
		}, ErrUnsupportedImage},
		{"check", func() error { return (&Descriptor{Rect: image.Rect(0, 0, 8, 8), Pack: 5, Planes: 1}).Check() }, ErrInvalidDescriptor},
		{"converter check", func() error {
			return newConverter(cfg(g8, Descriptor{Rect: image.Rect(0, 0, 8, 8), Planes: 9}, false), Bilinear())
		}, ErrInvalidDescriptor},
		{"depth", func() error { return newConverter(cfg(g8, g16, false), Bilinear()) }, ErrIncompatible},
		{"pack", func() error { return newConverter(cfg(g8, rgba, false), Bilinear()) }, ErrIncompatible},
		{"planes", func() error { return newConverter(cfg(yuv, g8, false), Bilinear()) }, ErrIncompatible},
		{"interlacing", func() error {
			c := cfg(g8, g8, false)
			c.Output.Interlaced = true
			return newConverter(c, Bilinear())
		}, ErrIncompatible},
		{"converter too small", func() error {
			return newConverter(cfg(Descriptor{Rect: image.Rect(0, 0, 1, 8), Pack: 1, Planes: 1, Ratio: Ratio444}, g8, false), Bilinear())
		}, ErrTooSmall},
		{"interlaced too small", func() error {
			return newConverter(cfg(Descriptor{Rect: image.Rect(0, 0, 8, 3), Pack: 1, Planes: 1, Ratio: Ratio444}, g8, true), Bilinear())
		}, ErrTooSmall},
		{"resize too small", func() error {
			_, err := NewResizer(&ResizerConfig{Input: 1, Output: 8}, Bilinear())
			return err
		}, ErrTooSmall},
		{"empty image", func() error {
			// a row of padding and no rows, which the standard constructors
			// do not build but SubImage can describe
			img := &image.Gray{Stride: 16, Rect: image.Rect(0, 0, 10, 0)}
			_, err := Prepare(gray(8, 8), img)
			return err
		}, ErrTooSmall},
		{"empty yuv", func() error {
			img := &image.YCbCr{YStride: 16, CStride: 8, Rect: image.Rect(0, 0, 0, 10), SubsampleRatio: image.YCbCrSubsampleRatio420}
			_, err := Prepare(img, img)
			return err
		}, ErrTooSmall},
		{"negative stride", func() error {
			img := &image.Gray{Pix: make([]byte, 64), Stride: -1, Rect: image.Rect(0, 0, 8, 8)}
			return Convert(gray(16, 16), img, Bilinear())
		}, ErrUnsupportedImage},
		{"negative stride copy", func() error {
			img := &image.Gray{Pix: make([]byte, 64), Stride: -1, Rect: image.Rect(0, 0, 8, 8)}
			return Convert(gray(8, 8), img, Bilinear())
		}, ErrUnsupportedImage},
		{"negative output stride", func() error {
			img := &image.RGBA{Pix: make([]byte, 256), Stride: -4, Rect: image.Rect(0, 0, 8, 8)}
			return Convert(img, newImage(t, image.NewRGBA, image.Rect(0, 0, 4, 4)), Bilinear())
		}, ErrUnsupportedImage},
		{"negative chroma stride", func() error {
			img := newImage(t, ycbcr(image.YCbCrSubsampleRatio420), image.Rect(0, 0, 8, 8))
			img.CStride = -1
			return Convert(newImage(t, ycbcr(image.YCbCrSubsampleRatio420), image.Rect(0, 0, 16, 16)), img, Bilinear())
		}, ErrUnsupportedImage},
		{"negative alpha stride", func() error {
			img := newImage(t, nycbcra(image.YCbCrSubsampleRatio420), image.Rect(0, 0, 8, 8))
			img.AStride = -1
			return Convert(newImage(t, nycbcra(image.YCbCrSubsampleRatio420), image.Rect(0, 0, 16, 16)), img, Bilinear())
		}, ErrUnsupportedImage},
		{"overflowing stride", func() error {
			// stride*(height-1) wraps to 0
			img := &image.Gray{Pix: make([]byte, 64), Stride: math.MaxInt/2 + 1, Rect: image.Rect(0, 0, 8, 5)}
			return Convert(gray(16, 16), img, Bilinear())
		}, ErrUnsupportedImage},
		{"short output pixels", func() error {
			// Pix shorter than its rows with the capacity for them
			buf := bytes.Repeat([]byte{0xA5}, 128)
			img := &image.Gray{Pix: buf[:60], Stride: 8, Rect: image.Rect(0, 0, 8, 8)}
			err := Convert(img, gray(4, 4), Bilinear())
			if !bytes.Equal(buf[60:], bytes.Repeat([]byte{0xA5}, 68)) {
				t.Errorf("short output pixels: wrote past Pix")
			}
			return err
		}, ErrUnsupportedImage},
		{"short chroma pixels", func() error {
			img := newImage(t, ycbcr(image.YCbCrSubsampleRatio420), image.Rect(0, 0, 8, 8))
			img.Cb = append(make([]byte, 0, 16), img.Cb[:15]...)
			return Convert(newImage(t, ycbcr(image.YCbCrSubsampleRatio420), image.Rect(0, 0, 16, 16)), img, Bilinear())
		}, ErrUnsupportedImage},
		{"nil alpha", func() error {
			img := newImage(t, nycbcra(image.YCbCrSubsampleRatio420), image.Rect(0, 0, 8, 8))
			img.A = nil
			return Convert(newImage(t, nycbcra(image.YCbCrSubsampleRatio420), image.Rect(0, 0, 16, 16)), img, Bilinear())
		}, ErrUnsupportedImage},
		{"nil output alpha", func() error {
			img := newImage(t, nycbcra(image.YCbCrSubsampleRatio420), image.Rect(0, 0, 16, 16))
			img.A = nil
			return Convert(img, newImage(t, nycbcra(image.YCbCrSubsampleRatio420), image.Rect(0, 0, 8, 8)), Bilinear())
		}, ErrUnsupportedImage},
		{"nil alpha copy", func() error {
			img := newImage(t, nycbcra(image.YCbCrSubsampleRatio420), image.Rect(0, 0, 8, 8))
			img.A = nil
			_, err := Prepare(img, img)
			return err
		}, ErrUnsupportedImage},
		{"nil alpha converter", func() error {
			src := newImage(t, nycbcra(image.YCbCrSubsampleRatio420), image.Rect(0, 0, 8, 8))
			dst := newImage(t, nycbcra(image.YCbCrSubsampleRatio420), image.Rect(0, 0, 16, 16))
			cfg, err := Prepare(dst, src)
			if err != nil {
				return err
			}
			c, err := NewConverter(cfg, Bilinear())
			if err != nil {
				return err
			}
			src.A = nil
			return c.Convert(dst, src)
		}, ErrUnsupportedImage},
		{"short alpha", func() error {
			img := newImage(t, nycbcra(image.YCbCrSubsampleRatio420), image.Rect(0, 0, 8, 8))
			img.A = append(make([]byte, 0, 64), img.A[:63]...)
			return Convert(newImage(t, nycbcra(image.YCbCrSubsampleRatio420), image.Rect(0, 0, 16, 16)), img, Bilinear())
		}, ErrUnsupportedImage},
		{"mismatch", func() error {
			c, err := NewConverter(cfg(g8, g8, false), Bilinear())
			if err != nil {
				return err
			}
			return c.Convert(gray(8, 8), gray(9, 8))
		}, ErrMismatch},
		{"zero weights resize", func() error { _, err := NewResizer(&ResizerConfig{Input: 8, Output: 16}, constFilter{0}); return err }, ErrInvalidFilter},
		{"zero weights converter", func() error {
			return newConverter(cfg(g8, Descriptor{Rect: image.Rect(0, 0, 16, 16), Ratio: Ratio444, Pack: 1, Planes: 1}, false), constFilter{0})
		}, ErrInvalidFilter},
		{"nan weights", func() error {
			_, err := NewResizer(&ResizerConfig{Input: 8, Output: 16}, constFilter{math.NaN()})
			return err
		}, ErrInvalidFilter},
		{"inf weights", func() error {
			_, err := NewResizer(&ResizerConfig{Input: 8, Output: 16}, constFilter{math.Inf(1)})
			return err
		}, ErrInvalidFilter},
		{"spike weights", func() error { _, err := NewResizer(&ResizerConfig{Input: 8, Output: 16}, spikeFilter{}); return err }, ErrInvalidFilter},
		{"negative support", func() error {
			_, err := NewResizer(&ResizerConfig{Input: 8, Output: 16}, supportFilter{-1})
			return err
		}, ErrInvalidFilter},
		{"zero support", func() error { _, err := NewResizer(&ResizerConfig{Input: 8, Output: 16}, supportFilter{0}); return err }, ErrInvalidFilter},
		{"nan support", func() error {
			_, err := NewResizer(&ResizerConfig{Input: 8, Output: 16}, supportFilter{math.NaN()})
			return err
		}, ErrInvalidFilter},
		{"inf support", func() error {
			_, err := NewResizer(&ResizerConfig{Input: 8, Output: 16}, supportFilter{math.Inf(1)})
			return err
		}, ErrInvalidFilter},
		{"nil filter resize", func() error { _, err := NewResizer(&ResizerConfig{Input: 8, Output: 16}, nil); return err }, ErrInvalidFilter},
		{"nil filter converter", func() error {
			return newConverter(cfg(g8, Descriptor{Rect: image.Rect(0, 0, 16, 16), Ratio: Ratio444, Pack: 1, Planes: 1}, false), nil)
		}, ErrInvalidFilter},
		{"nil filter convert", func() error { return Convert(gray(16, 16), gray(8, 8), nil) }, ErrInvalidFilter},
		{"nil filter copy", func() error { return Convert(gray(8, 8), gray(8, 8), nil) }, ErrInvalidFilter},
	} {
		err := tt.err()
		if !errors.Is(err, tt.want) {
			t.Errorf("%v: got %v, want %v", tt.name, err, tt.want)
		} else {
			t.Logf("%v: %v", tt.name, err)
		}
	}
	// a constant positive weight is a valid box filter
	r, err := NewResizer(&ResizerConfig{Input: 8, Output: 16}, constFilter{0.5})
	expect(t, err, nil)
	dst := make([]byte, 16)
	r.Resize(dst, []byte{0, 16, 32, 48, 64, 80, 96, 112}, 8, 1, 16, 8)
	expect(t, dst[1], byte(8))
}

// TestNarrowPlanes resizes every image type between narrow sizes and 24x24
// through sub-images, progressive and interlaced, and checks that the
// accelerated scalers match the Go ones bit for bit.
func TestNarrowPlanes(t *testing.T) {
	if !hasSIMD() {
		t.Skip("no simd")
	}
	rng := rand.New(rand.NewSource(1))
	big := image.Rect(0, 0, 24, 24)
	for _, it := range testImageTypes {
		for _, w := range []int{1, 2, 3, 5, 8, 15, 16, 17} {
			for _, h := range []int{1, 2, 3, 4, 5, 8, 9} {
				small := image.Rect(0, 0, w, h)
				for _, interlaced := range []bool{false, true} {
					for _, up := range []bool{true, false} {
						srcRect, dstRect := small, big
						if !up {
							srcRect, dstRect = big, small
						}
						src := newImage(t, it.new, srcRect)
						fillImage(rng, src)
						var outs [][]byte
						for _, simd := range []bool{true, false} {
							raw := newImage(t, it.new, image.Rect(0, 0, dstRect.Dx()+9, dstRect.Dy()+9))
							sub := dstRect.Add(image.Pt(7, 7))
							dst := raw.(subImager).SubImage(sub)
							check := checkOutside(t, raw, sub, 0xA5)
							cfg, err := Prepare(dst, src)
							expect(t, err, nil)
							cfg.Input.Interlaced, cfg.Output.Interlaced = interlaced, interlaced
							cfg.DisableSIMD = !simd
							c, err := NewConverter(cfg, Lanczos(3))
							if err != nil && strings.Contains(err.Error(), "too small") {
								break
							}
							expect(t, err, nil)
							expect(t, c.Convert(dst, src), nil)
							check()
							outs = append(outs, imageBytes(dst))
						}
						if len(outs) == 2 && !bytes.Equal(outs[0], outs[1]) {
							t.Errorf("%v %v -> %v interlaced=%v: simd differs from go", it.name, srcRect.Size(), dstRect.Size(), interlaced)
						}
					}
				}
			}
		}
	}
}

// TestInterlacedChromaRows converts interlaced 4:2:0 images whose chroma
// planes have an odd number of rows, which are split unevenly between the
// fields, and rejects the ones with a single row in a field.
func TestInterlacedChromaRows(t *testing.T) {
	for _, tt := range []struct {
		h   int
		err string
	}{
		{5, "too small"}, {6, "too small"}, {9, ""}, {10, ""}, {12, ""}, {13, ""},
	} {
		src := newImage(t, ycbcr(image.YCbCrSubsampleRatio420), image.Rect(0, 0, 16, tt.h))
		dst := newImage(t, ycbcr(image.YCbCrSubsampleRatio420), image.Rect(0, 0, 16, 24))
		for _, img := range []*image.YCbCr{src, dst} {
			cfg, err := Prepare(img, src)
			expect(t, err, nil)
			cfg.Input.Interlaced, cfg.Output.Interlaced = true, true
			c, err := NewConverter(cfg, Bicubic())
			if tt.err != "" {
				if err == nil || !strings.Contains(err.Error(), tt.err) {
					t.Errorf("height %v: got %v, want %q", tt.h, err, tt.err)
				}
				continue
			}
			expect(t, err, nil)
			expect(t, c.Convert(img, src), nil)
		}
	}
}

// TestYCbCrOrigins converts YCbCr and NYCbCrA images of every ratio whose
// rectangles start at negative even and odd coordinates, whole images and
// sub-images, progressive and interlaced, against the reference and Go
// against SIMD. image.YCbCr sizes its chroma planes from the coordinates,
// with truncating division.
func TestYCbCrOrigins(t *testing.T) {
	origins := []image.Point{{-1, 0}, {0, -1}, {-1, -1}, {-2, -2}, {-2, -1}, {-1, -2}, {-3, -5}, {-4, -3}, {-6, -7}}
	sizes := []struct{ wsrc, hsrc, wdst, hdst int }{{17, 16, 24, 21}, {16, 17, 9, 10}}
	for typ, it := range testImageTypes {
		if !strings.Contains(it.name, "YCbCr") {
			continue
		}
		for _, o := range origins {
			for off := range 3 {
				checkYCbCrPlanes(t, subImageEnding(t, it.new, 17, 16, o.X-off, o.Y-off, off, off))
				for _, s := range sizes {
					for _, interlaced := range []bool{false, true} {
						c := convertCase{
							typ:  typ,
							wsrc: s.wsrc, hsrc: s.hsrc,
							wdst: s.wdst, hdst: s.hdst,
							ox: off, oy: off,
							px: o.X - off, py: o.Y - off,
							interlaced: interlaced,
							filter:     2,
							threads:    2,
							seed:       int64(typ),
						}
						if !runConvertCase(t, c) {
							t.Fatalf("%v %+v: too small", it.name, c)
						}
					}
				}
			}
		}
	}
	// chroma planes without columns or rows
	for _, tt := range []struct {
		r     image.Rectangle
		ratio image.YCbCrSubsampleRatio
	}{
		{image.Rect(0, -1, 8, 0), image.YCbCrSubsampleRatio420},
		{image.Rect(-3, 0, 0, 8), image.YCbCrSubsampleRatio411},
	} {
		img := newImage(t, ycbcr(tt.ratio), tt.r)
		if _, err := Prepare(img, img); !errors.Is(err, ErrTooSmall) {
			t.Errorf("%v %v: got %v, want ErrTooSmall", tt.r, tt.ratio, err)
		}
	}
}

// checkYCbCrPlanes checks that the chroma planes inspect describes for a
// YCbCr or NYCbCrA image hold exactly the samples its pixels read.
func checkYCbCrPlanes(t *testing.T, img image.Image) {
	t.Helper()
	yuv, ok := img.(*image.YCbCr)
	if !ok {
		yuv = &img.(*image.NYCbCrA).YCbCr
	}
	_, planes, err := inspect(img, false)
	expect(t, err, nil)
	p := planes[1]
	expect(t, planes[2].Width, p.Width)
	expect(t, planes[2].Height, p.Height)
	read := make([]bool, p.Width*p.Height)
	for y := yuv.Rect.Min.Y; y < yuv.Rect.Max.Y; y++ {
		for x := yuv.Rect.Min.X; x < yuv.Rect.Max.X; x++ {
			ci := yuv.COffset(x, y)
			cx, cy := ci%p.Pitch, ci/p.Pitch
			if cx >= p.Width || cy >= p.Height {
				t.Fatalf("%v %v: pixel %v,%v reads chroma %v,%v outside the %vx%v plane", yuv.Rect, yuv.SubsampleRatio, x, y, cx, cy, p.Width, p.Height)
			}
			read[cy*p.Width+cx] = true
		}
	}
	for i, ok := range read {
		if !ok {
			t.Fatalf("%v %v: no pixel reads chroma %v,%v of the %vx%v plane", yuv.Rect, yuv.SubsampleRatio, i%p.Width, i/p.Width, p.Width, p.Height)
		}
	}
}
