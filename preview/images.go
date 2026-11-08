// Copyright 2026 Patrick Gaskin. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"math"
	"path/filepath"
	"strings"

	"golang.org/x/image/font"
	"golang.org/x/image/font/opentype"
	"golang.org/x/image/math/fixed"
)

const size = 256 // of the test images

// The test images are laid out on a 192x192 grid scaled to size, except for
// 1px features such as thin lines, checkerboards, and dots.
const scale = float64(size) / 192

// px scales v from the layout grid to pixels.
func px(v int) int {
	return int(math.Round(float64(v) * scale))
}

const lorem = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."

// textImage returns lorem ipsum at decreasing sizes, black on white.
func textImage() *image.RGBA {
	img := newRGBA(color.White)
	y := px(4)
	for _, t := range []struct {
		font  *opentype.Font
		size  int
		lines int
	}{
		{serif, 22, 1},
		{sans, 16, 2},
		{serif, 13, 2},
		{sans, 11, 3},
		{mono, 9, 3},
	} {
		face := newFace(t.font, float64(t.size)*scale)
		d := font.Drawer{Dst: img, Src: image.Black, Face: face}
		ascent := face.Metrics().Ascent.Ceil()
		words := strings.Fields(lorem)
		for range t.lines {
			var line string
			for len(words) > 0 {
				next := strings.TrimSpace(line + " " + words[0])
				if d.MeasureString(next).Ceil() > size-px(8) {
					break
				}
				line, words = next, words[1:]
			}
			d.Dot = fixed.P(px(4), y+ascent)
			d.DrawString(line)
			y += px(t.size * 6 / 5)
		}
		y += px(3)
	}
	if y > size {
		panic("text does not fit")
	}
	return img
}

// ringsImage returns a zone plate reaching the Nyquist frequency at the edges.
func ringsImage() *image.Gray {
	img := image.NewGray(image.Rect(0, 0, size, size))
	k := math.Pi / size
	for y := range size {
		for x := range size {
			dx, dy := float64(x)-size/2+0.5, float64(y)-size/2+0.5
			img.SetGray(x, y, color.Gray{uint8(math.Round(127.5 + 127.5*math.Cos(k*(dx*dx+dy*dy))))})
		}
	}
	return img
}

// starImage returns a Siemens star.
func starImage() *image.Gray {
	img := image.NewGray(image.Rect(0, 0, size, size))
	draw.Draw(img, img.Bounds(), image.White, image.Point{}, draw.Src)
	const spokes = 48
	fill(img, color.Black, func(x, y float64) bool {
		dx, dy := x-size/2, y-size/2
		if math.Hypot(dx, dy) > size/2-2 {
			return false
		}
		a := math.Atan2(dy, dx)/(2*math.Pi) + 0.5
		return int(a*spokes*2)%2 == 0
	})
	return img
}

// edgesImage returns hard edges, thin lines, fine checkerboards, and
// gradients.
func edgesImage() *image.RGBA {
	img := newRGBA(color.White)
	for x := range size {
		v := uint8(255 * x / (size - 1))
		rect(img, x, 0, x+1, px(24), color.RGBA{v, 0, 255 - v, 255})
		rect(img, x, px(24), x+1, px(48), color.RGBA{v, v, v, 255})
	}
	for i, c := range []color.RGBA{
		{255, 0, 0, 255}, {0, 255, 0, 255}, {0, 0, 255, 255},
		{255, 255, 0, 255}, {0, 255, 255, 255}, {255, 0, 255, 255},
	} {
		x := 4 + i*31
		rect(img, px(x), px(52), px(x+27), px(79), c)
		rect(img, px(x+8), px(60), px(x+19), px(71), color.Black)
	}
	for i := range 12 {
		x := px(4 + i*8)
		rect(img, x, px(84), x+1+i%3, px(133), color.Black)
	}
	for y := px(84); y < px(133); y++ {
		for x := px(100); x < px(144); x++ {
			if (x+y)%2 != 0 {
				img.Set(x, y, color.Black)
			}
		}
		for x := px(144); x < px(188); x++ {
			if (x/2+y/2)%2 != 0 {
				img.Set(x, y, color.Black)
			}
		}
	}
	fill(img, color.Black, func(x, y float64) bool {
		return math.Abs(math.Hypot(x-29*scale, y-163*scale)-25*scale) < 0.5
	})
	fill(img, color.RGBA{200, 0, 0, 255}, func(x, y float64) bool {
		return math.Hypot(x-29*scale, y-163*scale) < 15*scale
	})
	fill(img, color.Black, func(x, y float64) bool {
		x, y = x/scale, y/scale
		if y < 138 || y > 188 {
			return false
		}
		// lines from (60+8i, 188) to (100+8i, 138)
		u := x - 60 - (188-y)*0.8
		i := math.Round(u / 8)
		return i >= 0 && i < 8 && math.Abs(u-8*i)*scale*50/math.Hypot(40, 50) < 0.5
	})
	rect(img, px(140), px(138), px(189), px(189), color.Black)
	rect(img, px(150), px(148), px(179), px(179), color.White)
	for _, p := range []image.Point{{164, 163}, {160, 158}, {168, 168}} {
		img.Set(px(p.X), px(p.Y), color.Black)
	}
	return img
}

// photoImage returns a cropped (and scaled) high-resolution photo.
//
//	magick daniele-levis-pelusi-fUOTndD4Tak-unsplash.jpg -crop 1280x1280+1386+604 +repage -filter box -resize 256x256 -strip PNG24:photo.png
//
// https://unsplash.com/photos/fUOTndD4Tak
func photoImage() (image.Image, error) {
	photo, err := decode(filepath.Join("images", "photo.png"))
	if err != nil {
		return nil, err
	}
	if b := photo.Bounds(); b.Dx() != size || b.Dy() != size {
		return nil, fmt.Errorf("photo: size %v, expected %dx%d", b.Size(), size, size)
	}
	return photo, nil
}

func newRGBA(bg color.Color) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, size, size))
	draw.Draw(img, img.Bounds(), image.NewUniform(bg), image.Point{}, draw.Src)
	return img
}

func rect(img draw.Image, x0, y0, x1, y1 int, c color.Color) {
	draw.Draw(img, image.Rect(x0, y0, x1, y1), image.NewUniform(c), image.Point{}, draw.Src)
}

// fill blends c over the pixels of img where in is true, weighted by the
// fraction of 8x8 samples in each pixel.
func fill(img draw.Image, c color.Color, in func(x, y float64) bool) {
	const n = 8
	b := img.Bounds()
	src := image.NewUniform(c)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			var cov int
			for sy := range n {
				for sx := range n {
					if in(float64(x)+(float64(sx)+0.5)/n, float64(y)+(float64(sy)+0.5)/n) {
						cov++
					}
				}
			}
			if cov != 0 {
				mask := image.NewUniform(color.Alpha16{uint16(cov * 0xffff / (n * n))})
				draw.DrawMask(img, image.Rect(x, y, x+1, y+1), src, image.Point{}, mask, image.Point{}, draw.Over)
			}
		}
	}
}
