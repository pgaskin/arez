// Copyright 2026 Patrick Gaskin, 2013 Benoît Amiaux. All rights reserved. Use
// of this source code is governed by a MIT-style license that can be found in
// the LICENSE file.

package arez_test

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"log"

	"github.com/pgaskin/arez"
)

func ExampleConvert() {
	src := image.NewRGBA(image.Rect(0, 0, 640, 480))
	draw.Draw(src, src.Bounds(), image.NewUniform(color.RGBA{R: 200, G: 100, B: 50, A: 255}), image.Point{}, draw.Src)

	dst := image.NewRGBA(image.Rect(0, 0, 320, 240))
	if err := arez.Convert(dst, src, arez.Lanczos(3)); err != nil {
		log.Fatal(err)
	}
	fmt.Println(dst.Bounds(), dst.RGBAAt(160, 120))
	// Output: (0,0)-(320,240) {200 100 50 255}
}

func ExampleNewConverter() {
	// frames of a 4:2:0 video, downscaled and converted to 4:4:4
	frames := make([]*image.YCbCr, 3)
	for i := range frames {
		frames[i] = image.NewYCbCr(image.Rect(0, 0, 1280, 720), image.YCbCrSubsampleRatio420)
	}
	dst := image.NewYCbCr(image.Rect(0, 0, 640, 360), image.YCbCrSubsampleRatio444)

	cfg, err := arez.Prepare(dst, frames[0])
	if err != nil {
		log.Fatal(err)
	}
	converter, err := arez.NewConverter(cfg, arez.Bicubic())
	if err != nil {
		log.Fatal(err)
	}
	for _, frame := range frames {
		if err := converter.Convert(dst, frame); err != nil {
			log.Fatal(err)
		}
	}
	fmt.Println(cfg.Input)
	fmt.Println(cfg.Output)
	// Output:
	// 1280x720 3 planes 4:2:0 1-packed 8-bit progressive
	// 640x360 3 planes 4:4:4 1-packed 8-bit progressive
}

func ExampleNewResizer() {
	// a horizontal ramp of 2 rows of 4 gray pixels, doubled in width
	src := []byte{
		0, 80, 160, 240,
		240, 160, 80, 0,
	}
	dst := make([]byte, 2*8)
	r, err := arez.NewResizer(&arez.ResizerConfig{Input: 4, Output: 8}, arez.Bilinear())
	if err != nil {
		log.Fatal(err)
	}
	r.Resize(dst, src, 4, 2, 8, 4)
	fmt.Println(dst[:8])
	fmt.Println(dst[8:])
	// Output:
	// [0 20 60 100 140 180 220 240]
	// [240 220 180 140 100 60 20 0]
}

func ExampleConverterConfig_premultiplyAlpha() {
	// opaque red on the left, transparent green on the right
	src := image.NewNRGBA(image.Rect(0, 0, 8, 8))
	for y := range 8 {
		for x := range 8 {
			c := color.NRGBA{R: 255, A: 255}
			if x >= 4 {
				c = color.NRGBA{G: 255}
			}
			src.SetNRGBA(x, y, c)
		}
	}
	dst := image.NewNRGBA(image.Rect(0, 0, 4, 4))

	for _, premultiply := range []bool{false, true} {
		cfg, err := arez.Prepare(dst, src)
		if err != nil {
			log.Fatal(err)
		}
		cfg.PremultiplyAlpha = premultiply
		converter, err := arez.NewConverter(cfg, arez.Bilinear())
		if err != nil {
			log.Fatal(err)
		}
		if err := converter.Convert(dst, src); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("premultiply=%v: %v %v\n", premultiply, dst.NRGBAAt(1, 0), dst.NRGBAAt(2, 0))
	}
	// Output:
	// premultiply=false: {223 32 0 223} {32 223 0 32}
	// premultiply=true: {255 0 0 223} {255 0 0 32}
}
