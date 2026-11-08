// Copyright 2026 Patrick Gaskin. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"log"
	"os"
	"path/filepath"

	"github.com/go-fonts/dejavu/dejavusans"
	"github.com/go-fonts/liberation/liberationmonoregular"
	"github.com/go-fonts/liberation/liberationsansregular"
	"github.com/go-fonts/liberation/liberationserifregular"
	"github.com/pgaskin/arez"
	"golang.org/x/image/font"
	"golang.org/x/image/font/opentype"
	"golang.org/x/image/math/fixed"
)

//go:generate go run .

var (
	serif = parseFont(liberationserifregular.TTF)
	sans  = parseFont(liberationsansregular.TTF)
	mono  = parseFont(liberationmonoregular.TTF)
	ui    = parseFont(dejavusans.TTF)
)

var filters = []struct {
	name   string
	filter arez.Filter
}{
	{"Bilinear()", arez.Bilinear()},
	{"Bicubic()", arez.Bicubic()},
	{"CustomBicubic(1/3, 1/3)", arez.CustomBicubic(1.0/3, 1.0/3)},
	{"CustomBicubic(1, 0)", arez.CustomBicubic(1, 0)},
	{"Lanczos(2)", arez.Lanczos(2)},
	{"Lanczos(3)", arez.Lanczos(3)},
}

var ops = []struct {
	name       string
	crop, size int
}{
	{fmt.Sprintf("%d\u00d7%d \u2192 %d\u00d7%d", size, size, size*5/12, size*5/12), size, size * 5 / 12},
	{fmt.Sprintf("center %d\u00d7%d \u2192 %d\u00d7%d", size/4, size/4, size, size), size / 4, size},
}

// layout of compare.png
const (
	pad      = 10 // between cells
	groupPad = 32 // between ops
	lineSize = 16 // of the labels
	labelPad = 8  // above and below the label text
)

var (
	background      = color.White
	border          = labelBackground
	labelBackground = color.RGBA{232, 232, 232, 255}
	label           = color.Black
	sublabel        = color.RGBA{96, 96, 96, 255}
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	type testImage struct {
		name string
		img  image.Image
	}
	images := []testImage{
		{"text", textImage()},
		{"rings", ringsImage()},
		{"star", starImage()},
		{"edges", edgesImage()},
	}

	if err := os.MkdirAll("images", 0777); err != nil {
		return err
	}
	for _, t := range images {
		if err := encode(filepath.Join("images", t.name+".png"), t.img); err != nil {
			return err
		}
	}

	photo, err := photoImage()
	if err != nil {
		return err
	}
	images = append(images, testImage{"photo", photo})

	type row struct {
		name, sub string
		gap       int // above the label
		cells     []image.Image
	}
	originals := make([]image.Image, len(images))
	for i, t := range images {
		originals[i] = t.img
	}
	rows := []row{{"original", fmt.Sprintf("%d×%d", size, size), 0, originals}}
	for _, op := range ops {
		gap := groupPad
		for _, f := range filters {
			cells := make([]image.Image, len(images))
			for i, t := range images {
				img, err := resize(t.img, op.crop, op.size, f.filter)
				if err != nil {
					return fmt.Errorf("%s: %s: %s: %w", t.name, op.name, f.name, err)
				}
				cells[i] = img
			}
			rows = append(rows, row{f.name, op.name, gap, cells})
			gap = pad
		}
	}

	labelFace := newFace(mono, lineSize)
	sublabelFace := newFace(ui, lineSize)
	labelHeight := lineSize + 2*labelPad
	cellHeight := func(r row) int {
		var h int
		for _, img := range r.cells {
			h = max(h, img.Bounds().Dy())
		}
		return h
	}

	width := pad + len(images)*(size+pad)
	height := pad
	for _, r := range rows {
		height += r.gap + labelHeight + pad + cellHeight(r)
	}
	height += pad
	canvas := image.NewRGBA(image.Rect(0, 0, width, height))
	draw.Draw(canvas, canvas.Bounds(), image.NewUniform(background), image.Point{}, draw.Src)

	y := pad
	for _, r := range rows {
		y += r.gap
		draw.Draw(canvas, image.Rect(pad-1, y, width-pad+1, y+labelHeight), image.NewUniform(labelBackground), image.Point{}, draw.Src)
		m := labelFace.Metrics()
		baseline := y + (labelHeight+m.Ascent.Ceil()-m.Descent.Ceil())/2
		d := font.Drawer{Dst: canvas, Src: image.NewUniform(label), Face: labelFace, Dot: fixed.P(pad+labelPad, baseline)}
		d.DrawString(r.name)
		d.Src, d.Face = image.NewUniform(sublabel), sublabelFace
		d.DrawString("  " + r.sub)
		y += labelHeight + pad
		h := cellHeight(r)
		x := pad
		for _, img := range r.cells {
			b := img.Bounds()
			r := b.Sub(b.Min).Add(image.Pt(x+(size-b.Dx())/2, y+(h-b.Dy())/2))
			draw.Draw(canvas, r.Inset(-1), image.NewUniform(border), image.Point{}, draw.Src)
			draw.Draw(canvas, r, img, b.Min, draw.Src)
			x += size + pad
		}
		y += h
	}
	return encode("compare.png", canvas)
}

// resize resizes the center crop x crop pixels of img to size x size pixels,
// keeping the image type.
func resize(img image.Image, crop, size int, filter arez.Filter) (image.Image, error) {
	b := img.Bounds()
	c := b.Min.Add(image.Pt((b.Dx()-crop)/2, (b.Dy()-crop)/2))
	r := image.Rectangle{c, c.Add(image.Pt(crop, crop))}
	if !r.In(b) {
		return nil, fmt.Errorf("crop %v outside %v", r, b)
	}
	out := image.Rect(0, 0, size, size)
	var src, dst image.Image
	switch img := img.(type) {
	case *image.Gray:
		src, dst = img.SubImage(r), image.NewGray(out)
	case *image.RGBA:
		src, dst = img.SubImage(r), image.NewRGBA(out)
	default:
		return nil, fmt.Errorf("unsupported image type %T", img)
	}
	if err := arez.Convert(dst, src, filter); err != nil {
		return nil, err
	}
	return dst, nil
}

func decode(name string) (image.Image, error) {
	f, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	return img, nil
}

func encode(name string, img image.Image) error {
	var buf bytes.Buffer
	if err := (&png.Encoder{CompressionLevel: png.BestCompression}).Encode(&buf, img); err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	return os.WriteFile(name, buf.Bytes(), 0666)
}

func parseFont(ttf []byte) *opentype.Font {
	f, err := opentype.Parse(ttf)
	if err != nil {
		panic(err)
	}
	return f
}

func newFace(f *opentype.Font, size float64) font.Face {
	face, err := opentype.NewFace(f, &opentype.FaceOptions{
		Size:    size,
		DPI:     72,
		Hinting: font.HintingNone,
	})
	if err != nil {
		panic(err)
	}
	return face
}
