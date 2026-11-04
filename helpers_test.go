// Copyright 2026 Patrick Gaskin, 2013 Benoît Amiaux. All rights reserved. Use
// of this source code is governed by a MIT-style license that can be found in
// the LICENSE file.

package arez

import (
	"bytes"
	"encoding/binary"
	"flag"
	"fmt"
	"image"
	"image/draw"
	_ "image/jpeg"
	"image/png"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

func expect[T comparable](t testing.TB, got, want T) {
	t.Helper()
	if got != want {
		t.Fatalf("got %T(%v), want %T(%v)", got, got, want, want)
	}
}

// newImage calls ctor(r), but moves the image data if -arez.guard is set. It
// should be used for all tests.
func newImage[T image.Image](tb testing.TB, ctor func(image.Rectangle) T, r image.Rectangle) T {
	tb.Helper()
	img := ctor(r)
	guardImage(tb, img)
	return img
}

func ycbcr(ratio image.YCbCrSubsampleRatio) func(image.Rectangle) *image.YCbCr {
	return func(r image.Rectangle) *image.YCbCr { return image.NewYCbCr(r, ratio) }
}

func nycbcra(ratio image.YCbCrSubsampleRatio) func(image.Rectangle) *image.NYCbCrA {
	return func(r image.Rectangle) *image.NYCbCrA { return image.NewNYCbCrA(r, ratio) }
}

func paletted(r image.Rectangle) *image.Paletted { return image.NewPaletted(r, nil) }

func readImage(t testing.TB, name string) image.Image {
	t.Helper()
	file, err := os.Open(name)
	expect(t, err, nil)
	defer file.Close()
	raw, _, err := image.Decode(file)
	expect(t, err, nil)
	guardImage(t, raw)
	return raw
}

func writeImage(t testing.TB, name string, img image.Image) {
	t.Helper()
	if f := flag.Lookup("test.artifacts"); f == nil || f.Value.String() != "true" {
		return
	}
	file, err := os.Create(filepath.Join(t.ArtifactDir(), name))
	expect(t, err, nil)
	defer file.Close()
	err = png.Encode(file, img)
	expect(t, err, nil)
}

func prepare(t testing.TB, dst, src image.Image, simd, interlaced bool, filter Filter, threads int, opts ...func(*ConverterConfig)) Converter {
	t.Helper()
	cfg, err := Prepare(dst, src)
	expect(t, err, nil)
	cfg.Input.Interlaced = interlaced
	cfg.Output.Interlaced = interlaced
	cfg.Threads = threads
	cfg.DisableSIMD = !simd
	for _, opt := range opts {
		opt(cfg)
	}
	converter, err := NewConverter(cfg, filter)
	expect(t, err, nil)
	return converter
}

func convert(t testing.TB, dst, src image.Image, simd, interlaced bool, filter Filter) {
	t.Helper()
	converter := prepare(t, dst, src, simd, interlaced, filter, 0)
	err := converter.Convert(dst, src)
	expect(t, err, nil)
}

// may be overridden by guard_test.go
var allocPlane = func(n int) ([]byte, func()) { return make([]byte, n), func() {} }
var (
	filters = []Filter{
		Bilinear(),
		Bicubic(),
		Lanczos(3),
	}
)
var _ = allocPlane // so go doesn't complain about it being unused

func toRGB(tb testing.TB, src image.Image) *image.RGBA {
	tb.Helper()
	b := src.Bounds()
	dst := newImage(tb, image.NewRGBA, image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(dst, b, src, image.Point{}, draw.Src)
	return dst
}

// testConvertWith converts lenna.jpg, as RGBA when rgb is set and YCbCr
// otherwise, to sub-images of several sizes, progressive and interlaced, and
// compares the Go and accelerated scalers with each other and with the float
// reference. The outputs are written with go test -artifacts.
func testConvertWith(t *testing.T, rgb bool) {
	sizes := []struct{ w, h int }{
		{128, 128},
		{640, 480},
		{256, 256},
		{720, 576},
		{1920, 1080},
	}
	if testing.Short() {
		sizes = sizes[:2]
	}
	suffix := "yuv"
	src := readImage(t, "testdata/lenna.jpg")
	if rgb {
		suffix = "rgb"
		src = toRGB(t, src)
	}
	for _, f := range filters {
		for _, s := range sizes {
			for _, interlaced := range []bool{false, true} {
				name := fmt.Sprintf("%vx%v-%v-%v-%v", s.w, s.h, f.Name(), suffix, toInterlacedString(interlaced))
				var ref []refPlane
				var outs [][]byte
				for _, simd := range []bool{true, false} {
					if !simd && !hasSIMD() {
						break
					}
					raw := newImage(t, ycbcr(image.YCbCrSubsampleRatio420), image.Rect(0, 0, s.w*2, s.h*2))
					dst := raw.SubImage(image.Rect(7, 7, 7+s.w, 7+s.h))
					if rgb {
						dst = newImage(t, image.NewRGBA, image.Rect(0, 0, s.w*2, s.h*2)).SubImage(image.Rect(7, 7, 7+s.w, 7+s.h))
					}
					convert(t, dst, src, simd, interlaced, f)
					if ref == nil {
						ref = refConvert(t, dst, src, interlaced, false, f)
						writeImage(t, "output-"+name+".png", dst)
					}
					checkReference(t, dst, ref, 3, false, name)
					outs = append(outs, imageBytes(dst))
				}
				if len(outs) == 2 && !bytes.Equal(outs[0], outs[1]) {
					t.Errorf("%v: simd differs from go", name)
				}
			}
		}
	}
}

func NewTestCase(w, h int, interlaced bool) *TestCase {
	return &TestCase{
		file:       "lenna.jpg",
		filter:     Bicubic(),
		interlaced: interlaced,
		dst:        image.Rect(0, 0, w, h),
	}
}

type subImager interface {
	SubImage(r image.Rectangle) image.Image
}

// psnr computes the PSNR between two images of the same format, one value per
// plane.
func psnr(t testing.TB, a, b image.Image) []float64 {
	t.Helper()
	id, src, err := inspect(a, false)
	expect(t, err, nil)
	od, dst, err := inspect(b, false)
	expect(t, err, nil)
	expect(t, *id, *od)
	var psnrs []float64
	for i := range dst {
		psnrs = append(psnrs, psnrPlane(src[i].Data, dst[i].Data, src[i].Width*src[i].Bpp, src[i].Height, src[i].Pitch, dst[i].Pitch, id.depth()))
	}
	return psnrs
}

// psnrPlane computes the PSNR between two planes of width bytes, of 8-bit or
// big endian 16-bit samples.
func psnrPlane(dst, src []byte, width, height, dp, sp, depth int) float64 {
	mse := 0
	di := 0
	si := 0
	step := depth / 8
	for range height {
		for x := 0; x < width; x += step {
			var n int
			if depth == 16 {
				n = int(binary.BigEndian.Uint16(src[si+x:])) - int(binary.BigEndian.Uint16(dst[di+x:]))
			} else {
				n = int(src[si+x]) - int(dst[di+x])
			}
			mse += n * n
		}
		di += dp
		si += sp
	}
	peak := float64(int(1)<<depth - 1)
	fmse := float64(mse) / float64(width/step*height)
	return 10 * math.Log10(peak*peak/fmse)
}

func checkPSNRs(t *testing.T, ref, img image.Image, sub image.Rectangle, want []float64) {
	t.Helper()
	if !sub.Empty() {
		ref = ref.(subImager).SubImage(sub)
		img = img.(subImager).SubImage(sub)
	}
	for i, v := range psnr(t, ref, img) {
		if v < want[i] {
			t.Fatalf("invalid psnr %v < %v", v, want[i])
		}
	}
}

func runTestCaseWith(t *testing.T, tc *TestCase, simd bool, cycles int) image.Image {
	srcRaw := readImage(t, "testdata/"+tc.file).(*image.YCbCr)
	dstRaw := newImage(t, ycbcr(srcRaw.SubsampleRatio), image.Rect(0, 0, tc.dst.Max.X*2, tc.dst.Max.Y*2))
	var src, dst, ref image.Image
	if tc.src.Empty() {
		tc.src = srcRaw.Bounds()
	}
	suffix := "yuv"
	if tc.rgb {
		suffix = "rgb"
		src = toRGB(t, srcRaw).SubImage(tc.src)
		ref = toRGB(t, srcRaw).SubImage(tc.src)
		dst = toRGB(t, dstRaw).SubImage(tc.dst)
	} else {
		src = srcRaw.SubImage(tc.src)
		refRaw := newImage(t, ycbcr(srcRaw.SubsampleRatio), src.Bounds())
		err := Convert(refRaw, src, Bilinear())
		expect(t, err, nil)
		ref = refRaw.SubImage(tc.src)
		dst = dstRaw.SubImage(tc.dst)
	}
	fwd := prepare(t, dst, src, simd, tc.interlaced, tc.filter, tc.threads)
	bwd := prepare(t, src, dst, simd, tc.interlaced, tc.filter, tc.threads)
	for range cycles {
		err := fwd.Convert(dst, src)
		expect(t, err, nil)
		err = bwd.Convert(src, dst)
		expect(t, err, nil)
	}
	if len(tc.psnrs) > 0 {
		checkPSNRs(t, ref, src, tc.psnrRect, tc.psnrs)
	}
	if len(tc.dump) > 0 {
		sb := src.Bounds()
		db := dst.Bounds()
		simdSuffix := "go"
		if simd {
			simdSuffix = "simd"
		}
		name := fmt.Sprintf("%v-%vx%v-%vx%v-%v-%v-%v-%v.png",
			tc.dump, sb.Dx(), sb.Dy(), db.Dx(), db.Dy(), suffix,
			toInterlacedString(tc.interlaced), simdSuffix, tc.filter.Name())
		writeImage(t, name, src)
	}
	return src
}

func runTestCase(t *testing.T, tc *TestCase, cycles int) {
	simd := runTestCaseWith(t, tc, true, cycles)
	if !hasSIMD() || testing.Short() {
		return
	}
	nosimd := runTestCaseWith(t, tc, false, cycles)
	checkPSNRs(t, nosimd, simd, image.Rectangle{}, []float64{math.Inf(1), math.Inf(1), math.Inf(1)})
}

var planeSizes = []struct {
	w, h       int
	interlaced bool
}{
	{256, 256, false},
	{256, 256, true},
	{171, 333, false},
	{170, 334, true},
}

// testPlanes resizes src to all planeSizes and back, with the Go scalers and
// the accelerated ones, checks the result against src with the psnrs and that
// both paths result in identical images.
func testPlanes(t *testing.T, src image.Image, ctor func(image.Rectangle) image.Image, psnrs []float64) {
	t.Helper()
	ref := newImage(t, ctor, src.Bounds())
	err := Convert(ref, src, Bilinear())
	expect(t, err, nil)
	for _, s := range planeSizes {
		var outs []image.Image
		for _, simd := range []bool{true, false} {
			if !simd && (!hasSIMD() || testing.Short()) {
				break
			}
			img := newImage(t, ctor, src.Bounds())
			raw := newImage(t, ctor, image.Rect(0, 0, s.w*2, s.h*2))
			dst := raw.(subImager).SubImage(image.Rect(7, 7, 7+s.w, 7+s.h))
			convert(t, dst, src, simd, s.interlaced, Bicubic())
			convert(t, img, dst, simd, s.interlaced, Bicubic())
			t.Logf("%T %vx%v %v simd=%v: psnr %.1f", src, s.w, s.h, toInterlacedString(s.interlaced), simd, psnr(t, ref, img))
			checkPSNRs(t, ref, img, image.Rectangle{}, psnrs)
			outs = append(outs, img)
		}
		if len(outs) == 2 {
			inf := make([]float64, len(psnrs))
			for i := range inf {
				inf[i] = math.Inf(1)
			}
			checkPSNRs(t, outs[0], outs[1], image.Rectangle{}, inf)
		}
	}
}

func drawInto[T draw.Image](tb testing.TB, src image.Image, ctor func(image.Rectangle) T) T {
	tb.Helper()
	dst := newImage(tb, ctor, src.Bounds())
	draw.Draw(dst, dst.Bounds(), src, src.Bounds().Min, draw.Src)
	return dst
}

var testImageTypes = []struct {
	name string
	new  func(image.Rectangle) image.Image
}{
	{"Gray", func(r image.Rectangle) image.Image { return image.NewGray(r) }},
	{"Alpha", func(r image.Rectangle) image.Image { return image.NewAlpha(r) }},
	{"RGBA", func(r image.Rectangle) image.Image { return image.NewRGBA(r) }},
	{"NRGBA", func(r image.Rectangle) image.Image { return image.NewNRGBA(r) }},
	{"CMYK", func(r image.Rectangle) image.Image { return image.NewCMYK(r) }},
	{"Gray16", func(r image.Rectangle) image.Image { return image.NewGray16(r) }},
	{"Alpha16", func(r image.Rectangle) image.Image { return image.NewAlpha16(r) }},
	{"RGBA64", func(r image.Rectangle) image.Image { return image.NewRGBA64(r) }},
	{"NRGBA64", func(r image.Rectangle) image.Image { return image.NewNRGBA64(r) }},
	{"YCbCr420", func(r image.Rectangle) image.Image { return image.NewYCbCr(r, image.YCbCrSubsampleRatio420) }},
	{"YCbCr422", func(r image.Rectangle) image.Image { return image.NewYCbCr(r, image.YCbCrSubsampleRatio422) }},
	{"YCbCr444", func(r image.Rectangle) image.Image { return image.NewYCbCr(r, image.YCbCrSubsampleRatio444) }},
	{"YCbCr411", func(r image.Rectangle) image.Image { return image.NewYCbCr(r, image.YCbCrSubsampleRatio411) }},
	{"YCbCr410", func(r image.Rectangle) image.Image { return image.NewYCbCr(r, image.YCbCrSubsampleRatio410) }},
	{"YCbCr440", func(r image.Rectangle) image.Image { return image.NewYCbCr(r, image.YCbCrSubsampleRatio440) }},
	{"NYCbCrA420", func(r image.Rectangle) image.Image { return image.NewNYCbCrA(r, image.YCbCrSubsampleRatio420) }},
}

func fillImage(rng *rand.Rand, img image.Image) {
	_, planes, err := inspect(img, false)
	if err != nil {
		panic(err)
	}
	for _, p := range planes {
		rng.Read(p.Data)
	}
}

func imageBytes(img image.Image) []byte {
	_, planes, err := inspect(img, false)
	if err != nil {
		panic(err)
	}
	var b []byte
	for _, p := range planes {
		for y := range p.Height {
			b = append(b, p.Data[y*p.Pitch:y*p.Pitch+p.Width*p.Bpp]...)
		}
	}
	return b
}

func subsample(r ChromaRatio, plane int) (sx, sy int) {
	if plane == 0 || plane == alphaPlane {
		return 1, 1
	}
	switch r {
	case Ratio410:
		return 4, 2
	case Ratio411:
		return 4, 1
	case Ratio420:
		return 2, 2
	case Ratio422:
		return 2, 1
	case Ratio440:
		return 1, 2
	}
	return 1, 1
}

func checkOutside(t *testing.T, img image.Image, sub image.Rectangle, v byte) func() {
	t.Helper()
	d, planes, err := inspect(img, false)
	expect(t, err, nil)
	for _, p := range planes {
		for i := range p.Data {
			p.Data[i] = v
		}
	}
	return func() {
		t.Helper()
		for i, p := range planes {
			sx, sy := subsample(d.Ratio, i)
			x0, x1 := sub.Min.X/sx*p.Bpp, ((sub.Max.X-1)/sx+1)*p.Bpp
			y0, y1 := sub.Min.Y/sy, (sub.Max.Y-1)/sy+1
			for y := range p.Height {
				for x := range p.Width * p.Bpp {
					if b := p.Data[y*p.Pitch+x]; b != v && (y < y0 || y >= y1 || x < x0 || x >= x1) {
						t.Fatalf("%T plane %v: byte %v,%v outside %v written", img, i, x, y, sub)
					}
				}
			}
		}
	}
}
