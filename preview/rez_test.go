// Copyright 2026 Patrick Gaskin. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"image"
	"testing"

	"github.com/bamiaux/rez"
)

var rezFilters = map[string]rez.Filter{
	"Bilinear()":              rez.NewBilinearFilter(),
	"Bicubic()":               rez.NewBicubicFilter(),
	"CustomBicubic(1/3, 1/3)": rez.NewCustomBicubicFilter(1.0/3, 1.0/3),
	"CustomBicubic(1, 0)":     rez.NewCustomBicubicFilter(1, 0),
	"Lanczos(2)":              rez.NewLanczosFilter(2),
	"Lanczos(3)":              rez.NewLanczosFilter(3),
}

// TestRez checks that resized images are bit-identical when resized with the
// original github.com/bamiaux/rez.
func TestRez(t *testing.T) {
	images, err := testImages()
	if err != nil {
		t.Fatal(err)
	}
	for _, op := range ops {
		for _, f := range filters {
			rf, ok := rezFilters[f.name]
			if !ok {
				t.Fatalf("no rez filter for %s", f.name)
			}
			for _, ti := range images {
				src, err := cropImage(ti.img, op.crop)
				if err != nil {
					t.Fatal(err)
				}
				exp := newImage(src, op.size)
				if err := rez.Convert(exp, src, rf); err != nil {
					t.Fatalf("%s: %s: %s: rez: %v", ti.name, op.name, f.name, err)
				}
				act, err := resize(ti.img, op.crop, op.size, f.filter)
				if err != nil {
					t.Fatalf("%s: %s: %s: %v", ti.name, op.name, f.name, err)
				}
				if !bytes.Equal(pix(exp), pix(act)) {
					t.Errorf("%s: %s: %s: output differs from rez", ti.name, op.name, f.name)
				}
			}
		}
	}
}

func pix(img image.Image) []byte {
	switch img := img.(type) {
	case *image.Gray:
		return img.Pix
	case *image.RGBA:
		return img.Pix
	}
	panic("unsupported image type")
}
