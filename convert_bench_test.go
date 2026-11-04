// Copyright 2026 Patrick Gaskin, 2013 Benoît Amiaux. All rights reserved. Use
// of this source code is governed by a MIT-style license that can be found in
// the LICENSE file.

package arez

import (
	"image"
	"testing"
)

type BenchType struct {
	win, hin    int
	wout, hout  int
	interlaced  bool
	conv        func(testing.TB, image.Image) image.Image
	filter      Filter
	premultiply bool
}

func toGray16(tb testing.TB, src image.Image) image.Image {
	tb.Helper()
	return drawInto(tb, src, image.NewGray16)
}

func toRGBA64(tb testing.TB, src image.Image) image.Image {
	tb.Helper()
	return drawInto(tb, src, image.NewRGBA64)
}

func toRGBA(tb testing.TB, src image.Image) image.Image {
	tb.Helper()
	return toRGB(tb, src)
}

func toNRGBA(tb testing.TB, src image.Image) image.Image {
	tb.Helper()
	return drawInto(tb, src, image.NewNRGBA)
}

var (
	benchs = []BenchType{
		{640, 480, 1920, 1080, false, nil, Bilinear(), false},
		{640, 480, 1920, 1080, false, nil, Bicubic(), false},
		{640, 480, 1920, 1080, false, nil, Lanczos(3), false},
		{1920, 1080, 640, 480, false, nil, Bilinear(), false},
		{1920, 1080, 640, 480, false, nil, Bicubic(), false},
		{1920, 1080, 640, 480, false, nil, Lanczos(3), false},
		{640, 480, 1920, 1080, true, nil, Bicubic(), false},
		{1920, 1080, 640, 480, true, nil, Bicubic(), false},
		{512, 512, 512, 512, true, nil, Bilinear(), false},
		{720, 576, 640, 480, false, toRGBA, Bicubic(), false},
		{720, 576, 640, 480, false, toGray16, Bicubic(), false},
		{720, 576, 640, 480, false, toRGBA64, Bicubic(), false},
		{640, 480, 1920, 1080, false, toGray16, Lanczos(3), false},
		{720, 576, 640, 480, false, toNRGBA, Bicubic(), false},
		{720, 576, 640, 480, false, toNRGBA, Bicubic(), true},
	}
	thumbs = []BenchType{
		{3840, 2160, 320, 180, false, nil, Lanczos(3), false},
		{3840, 2160, 320, 180, false, nil, Bicubic(), false},
		{3840, 2160, 320, 180, false, nil, Bilinear(), false},
		{3840, 2160, 320, 180, false, toRGBA, Lanczos(3), false},
		{1920, 1080, 160, 90, false, nil, Lanczos(3), false},
		{1920, 1080, 160, 90, false, nil, Bicubic(), false},
		{1920, 1080, 160, 90, false, toRGBA, Lanczos(3), false},
	}
)

func benchSpeed(b *testing.B, bt BenchType, simd bool) {
	raw := readImage(b, "testdata/lenna.jpg")
	var src, dst image.Image
	src = newImage(b, ycbcr(image.YCbCrSubsampleRatio420), image.Rect(0, 0, bt.win, bt.hin))
	convert(b, src, raw, true, bt.interlaced, bt.filter)
	dst = newImage(b, ycbcr(image.YCbCrSubsampleRatio420), image.Rect(0, 0, bt.wout, bt.hout))
	if bt.conv != nil {
		src = bt.conv(b, src)
		dst = bt.conv(b, dst)
	}
	_, planes, err := inspect(dst, false)
	expect(b, err, nil)
	bytes := int64(0)
	for _, p := range planes {
		bytes += int64(p.Width * p.Bpp * p.Height)
	}
	var opts []func(*ConverterConfig)
	if bt.premultiply {
		opts = append(opts, premultiplied)
	}
	converter := prepare(b, dst, src, simd, bt.interlaced, bt.filter, 0, opts...)
	b.SetBytes(bytes)
	for b.Loop() {
		converter.Convert(dst, src)
	}
}

func BenchmarkImageBilinearUpGo(b *testing.B)      { benchSpeed(b, benchs[0], false) }
func BenchmarkImageBilinearUpSimd(b *testing.B)    { benchSpeed(b, benchs[0], true) }
func BenchmarkImageBicubicUpGo(b *testing.B)       { benchSpeed(b, benchs[1], false) }
func BenchmarkImageBicubicUpSimd(b *testing.B)     { benchSpeed(b, benchs[1], true) }
func BenchmarkImageLanczosUpGo(b *testing.B)       { benchSpeed(b, benchs[2], false) }
func BenchmarkImageLanczosUpSimd(b *testing.B)     { benchSpeed(b, benchs[2], true) }
func BenchmarkImageBilinearDownGo(b *testing.B)    { benchSpeed(b, benchs[3], false) }
func BenchmarkImageBilinearDownSimd(b *testing.B)  { benchSpeed(b, benchs[3], true) }
func BenchmarkImageBicubicDownGo(b *testing.B)     { benchSpeed(b, benchs[4], false) }
func BenchmarkImageBicubicDownSimd(b *testing.B)   { benchSpeed(b, benchs[4], true) }
func BenchmarkImageLanczosDownGo(b *testing.B)     { benchSpeed(b, benchs[5], false) }
func BenchmarkImageLanczosDownSimd(b *testing.B)   { benchSpeed(b, benchs[5], true) }
func BenchmarkImageBicubicIUpGo(b *testing.B)      { benchSpeed(b, benchs[6], false) }
func BenchmarkImageBicubicIUpSimd(b *testing.B)    { benchSpeed(b, benchs[6], true) }
func BenchmarkImageBicubicIDownGo(b *testing.B)    { benchSpeed(b, benchs[7], false) }
func BenchmarkImageBicubicIDownSimd(b *testing.B)  { benchSpeed(b, benchs[7], true) }
func BenchmarkImageBicubicRgbGo(b *testing.B)      { benchSpeed(b, benchs[9], false) }
func BenchmarkImageBicubicRgbSimd(b *testing.B)    { benchSpeed(b, benchs[9], true) }
func BenchmarkCopy(b *testing.B)                   { benchSpeed(b, benchs[8], false) }
func BenchmarkImageBicubicGray16Go(b *testing.B)   { benchSpeed(b, benchs[10], false) }
func BenchmarkImageBicubicGray16Simd(b *testing.B) { benchSpeed(b, benchs[10], true) }
func BenchmarkImageBicubicRgba64Go(b *testing.B)   { benchSpeed(b, benchs[11], false) }
func BenchmarkImageBicubicRgba64Simd(b *testing.B) { benchSpeed(b, benchs[11], true) }
func BenchmarkImageLanczosUpGray16Go(b *testing.B) { benchSpeed(b, benchs[12], false) }
func BenchmarkImageLanczosUpGray16Simd(b *testing.B) {
	benchSpeed(b, benchs[12], true)
}

func BenchmarkImageBicubicNrgbaGo(b *testing.B)                { benchSpeed(b, benchs[13], false) }
func BenchmarkImageBicubicNrgbaSimd(b *testing.B)              { benchSpeed(b, benchs[13], true) }
func BenchmarkImageBicubicNrgbaPremultipliedGo(b *testing.B)   { benchSpeed(b, benchs[14], false) }
func BenchmarkImageBicubicNrgbaPremultipliedSimd(b *testing.B) { benchSpeed(b, benchs[14], true) }
func BenchmarkImageThumbLanczos4kYuvGo(b *testing.B)           { benchSpeed(b, thumbs[0], false) }
func BenchmarkImageThumbLanczos4kYuvSimd(b *testing.B)         { benchSpeed(b, thumbs[0], true) }
func BenchmarkImageThumbBicubic4kYuvGo(b *testing.B)           { benchSpeed(b, thumbs[1], false) }
func BenchmarkImageThumbBicubic4kYuvSimd(b *testing.B)         { benchSpeed(b, thumbs[1], true) }
func BenchmarkImageThumbBilinear4kYuvGo(b *testing.B)          { benchSpeed(b, thumbs[2], false) }
func BenchmarkImageThumbBilinear4kYuvSimd(b *testing.B)        { benchSpeed(b, thumbs[2], true) }
func BenchmarkImageThumbLanczos4kRgbaGo(b *testing.B)          { benchSpeed(b, thumbs[3], false) }
func BenchmarkImageThumbLanczos4kRgbaSimd(b *testing.B)        { benchSpeed(b, thumbs[3], true) }
func BenchmarkImageThumbLanczos1080YuvGo(b *testing.B)         { benchSpeed(b, thumbs[4], false) }
func BenchmarkImageThumbLanczos1080YuvSimd(b *testing.B)       { benchSpeed(b, thumbs[4], true) }
func BenchmarkImageThumbBicubic1080YuvGo(b *testing.B)         { benchSpeed(b, thumbs[5], false) }
func BenchmarkImageThumbBicubic1080YuvSimd(b *testing.B)       { benchSpeed(b, thumbs[5], true) }
func BenchmarkImageThumbLanczos1080RgbaGo(b *testing.B)        { benchSpeed(b, thumbs[6], false) }
func BenchmarkImageThumbLanczos1080RgbaSimd(b *testing.B)      { benchSpeed(b, thumbs[6], true) }
