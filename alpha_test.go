// Copyright 2026 Patrick Gaskin. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

package arez

import (
	"bytes"
	"encoding/binary"
	"errors"
	"image"
	"image/color"
	"math"
	"math/rand"
	"testing"
)

// premultiplied sets PremultiplyAlpha on a converter configuration.
func premultiplied(cfg *ConverterConfig) { cfg.PremultiplyAlpha = true }

// TestPremultiplyRows checks the alpha passes to ensure: the float reciprocal
// division is exact, 8-bit pixels come back exactly through the 16-bit
// premultiplied form, 16-bit ones have appropriate precision, and a zero alpha
// results in zero.
func TestPremultiplyRows(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for i := range 2000000 {
		a := uint32(1 + rng.Intn(65535))
		n := rng.Uint32()
		switch i % 4 {
		case 0:
			// exact multiples and the values next to them
			n = a*uint32(rng.Intn(65536)) + uint32(rng.Intn(3)) - 1
		case 1:
			n = uint32(rng.Intn(65536))*255 + a/2
		case 2:
			n = uint32(rng.Intn(65536))*65535 + a/2
		}
		if got := divide(n, 1/float64(a)); got != n/a {
			t.Fatalf("divide(%v, %v): got %v, want %v", n, a, got, n/a)
		}
	}
	for _, a := range []uint32{1, 2, 3, 255, 256, 257, 65534, 65535} {
		for _, n := range []uint32{0, 1, a - 1, a, a + 1, math.MaxUint32 - 1, math.MaxUint32} {
			expect(t, divide(n, 1/float64(a)), n/a)
		}
	}
	var s8, d8 [4]byte
	var p16, s16, d16 [8]byte
	for a := range 256 {
		for c := range 256 {
			s8 = [4]byte{byte(c), byte(255 - c), byte(c / 2), byte(a)}
			premultiplyRow8(p16[:], s8[:])
			expect(t, binary.BigEndian.Uint16(p16[6:]), uint16(a*257))
			expect(t, binary.BigEndian.Uint16(p16[0:]), uint16((c*a*257+127)/255))
			unpremultiplyRow8(d8[:], p16[:])
			if a == 0 {
				expect(t, d8, [4]byte{})
			} else {
				expect(t, d8, s8)
			}
		}
	}
	for range 10000 {
		a := uint16(rng.Intn(1 << 16))
		c := uint16(rng.Intn(1 << 16))
		binary.BigEndian.PutUint16(s16[0:], c)
		binary.BigEndian.PutUint16(s16[2:], 65535-c)
		binary.BigEndian.PutUint16(s16[4:], c/2)
		binary.BigEndian.PutUint16(s16[6:], a)
		premultiplyRow16(p16[:], s16[:])
		expect(t, binary.BigEndian.Uint16(p16[6:]), a)
		expect(t, binary.BigEndian.Uint16(p16[0:]), uint16((uint64(c)*uint64(a)+32767)/65535))
		unpremultiplyRow16(d16[:], p16[:])
		if a == 0 {
			expect(t, d16, [8]byte{})
			continue
		}
		// the premultiplied value is rounded to 1/65535, which is 1/a of
		// the straight value
		tol := int(math.Ceil(0.5 * 65535 / float64(a)))
		for i := 0; i < 6; i += 2 {
			got, want := int(binary.BigEndian.Uint16(d16[i:])), int(binary.BigEndian.Uint16(s16[i:]))
			if got < want-tol || got > want+tol {
				t.Fatalf("alpha %v colour %v: got %v, want within %v", a, want, got, tol)
			}
		}
	}
}

// edgeImage returns a w x h image of the type ctor returns where the left half
// is opaque red and right half transparent green, and set stores the 16-bit
// straight alpha channels of a pixel (since the usual Set would go through the
// colour model and lose the colour of transparent pixels).
func edgeImage[T image.Image](tb testing.TB, w, h int, ctor func(image.Rectangle) T, set func(T, int, int, color.NRGBA64)) T {
	tb.Helper()
	img := newImage(tb, ctor, image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			c := color.NRGBA64{R: 0xffff, A: 0xffff}
			if x >= w/2 {
				c = color.NRGBA64{G: 0xffff}
			}
			set(img, x, y, c)
		}
	}
	return img
}

func TestPremultiplyAlpha(t *testing.T) {
	type sample struct{ r, g, a int } // 16-bit
	for _, tt := range []struct {
		name string
		src  image.Image
		new  func(image.Rectangle) image.Image
		at   func(image.Image, int, int) sample
	}{
		{
			"NRGBA",
			edgeImage(t, 32, 32, image.NewNRGBA, func(img *image.NRGBA, x, y int, c color.NRGBA64) {
				img.SetNRGBA(x, y, color.NRGBA{R: uint8(c.R >> 8), G: uint8(c.G >> 8), B: uint8(c.B >> 8), A: uint8(c.A >> 8)})
			}),
			func(r image.Rectangle) image.Image { return image.NewNRGBA(r) },
			func(img image.Image, x, y int) sample {
				c := img.(*image.NRGBA).NRGBAAt(x, y)
				return sample{int(c.R) * 257, int(c.G) * 257, int(c.A) * 257}
			},
		},
		{
			"NRGBA64",
			edgeImage(t, 32, 32, image.NewNRGBA64, func(img *image.NRGBA64, x, y int, c color.NRGBA64) { img.SetNRGBA64(x, y, c) }),
			func(r image.Rectangle) image.Image { return image.NewNRGBA64(r) },
			func(img image.Image, x, y int) sample {
				c := img.(*image.NRGBA64).NRGBA64At(x, y)
				return sample{int(c.R), int(c.G), int(c.A)}
			},
		},
	} {
		for _, size := range []struct{ w, h int }{{16, 16}, {24, 40}, {32, 32}, {48, 20}} {
			var outs [][]byte
			for _, simd := range []bool{true, false} {
				if !simd && !hasSIMD() {
					break
				}
				straight := newImage(t, tt.new, image.Rect(0, 0, size.w, size.h))
				convert(t, straight, tt.src, simd, false, Lanczos(3))
				dst := newImage(t, tt.new, image.Rect(0, 0, size.w, size.h))
				c := prepare(t, dst, tt.src, simd, false, Lanczos(3), 0, premultiplied)
				expect(t, c.Convert(dst, tt.src), nil)
				outs = append(outs, imageBytes(dst))
				// a conversion of the same size copies, transparent colours included
				resized := tt.src.Bounds().Size() != image.Pt(size.w, size.h)
				bleed := 0
				for y := range size.h {
					for x := range size.w {
						s := tt.at(dst, x, y)
						if s.a > 0 && (s.g > 257 || s.r < 254*257) {
							t.Errorf("%v %vx%v simd=%v: pixel %v,%v is %+v, want pure red", tt.name, size.w, size.h, simd, x, y, s)
						}
						if resized && s.a == 0 && (s.g != 0 || s.r != 0) {
							t.Errorf("%v %vx%v simd=%v: pixel %v,%v is %+v, want zero colour", tt.name, size.w, size.h, simd, x, y, s)
						}
						bleed = max(bleed, tt.at(straight, x, y).g)
					}
				}
				if bleed < 16*257 {
					t.Errorf("%v %vx%v simd=%v: straight filtering bleeds only %v green", tt.name, size.w, size.h, simd, bleed)
				}
			}
			if len(outs) == 2 && !bytes.Equal(outs[0], outs[1]) {
				t.Errorf("%v %vx%v: simd differs from go", tt.name, size.w, size.h)
			}
		}
	}
	// what the option rejects
	nrgba := newImage(t, image.NewNRGBA, image.Rect(0, 0, 32, 32))
	small := newImage(t, image.NewNRGBA, image.Rect(0, 0, 16, 16))
	for _, img := range []image.Image{
		newImage(t, nycbcra(image.YCbCrSubsampleRatio420), image.Rect(0, 0, 32, 32)),
		newImage(t, image.NewGray, image.Rect(0, 0, 32, 32)),
		newImage(t, image.NewAlpha16, image.Rect(0, 0, 32, 32)),
	} {
		cfg, err := Prepare(img, img)
		expect(t, err, nil)
		cfg.PremultiplyAlpha = true
		_, err = NewConverter(cfg, Bicubic())
		if !errors.Is(err, ErrUnsupportedImage) {
			t.Errorf("%T: got %v, want ErrUnsupportedImage", img, err)
		}
	}
	c := prepare(t, small, nrgba, true, false, Bicubic(), 0, premultiplied)
	for _, tt := range []struct{ dst, src image.Image }{
		{newImage(t, image.NewRGBA, small.Rect), nrgba},
		{small, newImage(t, image.NewRGBA, nrgba.Rect)},
		{newImage(t, image.NewCMYK, small.Rect), newImage(t, image.NewCMYK, nrgba.Rect)},
	} {
		if err := c.Convert(tt.dst, tt.src); !errors.Is(err, ErrMismatch) {
			t.Errorf("%T from %T: got %v, want ErrMismatch", tt.dst, tt.src, err)
		}
	}
	expect(t, c.Convert(small, nrgba), nil)
}

// TestPremultiplyCopy checks that a premultiplied conversion to the same size
// copies the pixels correctly.
func TestPremultiplyCopy(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	r := image.Rect(0, 0, 64, 64)
	nrgba := newImage(t, image.NewNRGBA, r)
	rng.Read(nrgba.Pix)
	nrgba64 := newImage(t, image.NewNRGBA64, r)
	rng.Read(nrgba64.Pix)
	for i := 0; i < len(nrgba.Pix); i += 4 * 3 {
		nrgba.Pix[i+3] = byte(rng.Intn(4))
		binary.BigEndian.PutUint16(nrgba64.Pix[2*i+6:], uint16(rng.Intn(16)))
	}
	for _, tt := range []struct {
		src image.Image
		new func(image.Rectangle) image.Image
	}{
		{nrgba, func(r image.Rectangle) image.Image { return image.NewNRGBA(r) }},
		{nrgba64, func(r image.Rectangle) image.Image { return image.NewNRGBA64(r) }},
	} {
		for _, simd := range []bool{true, false} {
			for _, interlaced := range []bool{false, true} {
				dst := newImage(t, tt.new, r)
				c := prepare(t, dst, tt.src, simd, interlaced, Bicubic(), 0, premultiplied)
				expect(t, c.Convert(dst, tt.src), nil)
				if !bytes.Equal(imageBytes(dst), imageBytes(tt.src)) {
					t.Errorf("%T simd=%v interlaced=%v: pixels differ", tt.src, simd, interlaced)
				}
			}
		}
	}
}

// refResize resizes the premultiplied float pixels of src, win x hin of 4
// channels, to wout x hout with the float weights of filter.
func refResize(src []float64, win, hin, wout, hout int, filter Filter) []float64 {
	pass := func(src []float64, input, output, rows int, vertical bool) []float64 {
		cfg := ResizerConfig{Input: input, Output: output, Vertical: vertical}
		pos, sums, weights, taps, _ := makeDoubleKernel(&cfg, filter, 0, 0)
		at := func(x, y, c int) float64 { return src[(y*input+x)*4+c] }
		if vertical {
			at = func(x, y, c int) float64 { return src[(x*rows+y)*4+c] }
		}
		dst := make([]float64, output*rows*4)
		for i := range output {
			for r := range rows {
				for c := range 4 {
					v := 0.0
					for j := range taps {
						v += weights[i*taps+j] / sums[i] * at(pos[i]+j, r, c)
					}
					if vertical {
						dst[(i*rows+r)*4+c] = v
					} else {
						dst[(r*output+i)*4+c] = v
					}
				}
			}
		}
		return dst
	}
	mid := pass(src, hin, hout, win, true)
	for i, v := range mid {
		mid[i] = min(max(v, 0), 1)
	}
	return pass(mid, win, wout, hout, false)
}

// TestPremultiplyReference compares premultiplied conversions of random images
// against a float reference implementation, comparing the error from the 16-bit
// filtering against 8-bit premultiplication.
func TestPremultiplyReference(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const win, hin, wout, hout = 64, 48, 40, 30
	filter := Lanczos(3)
	for _, depth := range []int{8, 16} {
		var src, dst image.Image
		pix := make([]float64, win*hin*4)
		straight := make([]float64, win*hin*4)
		if depth == 8 {
			img := newImage(t, image.NewNRGBA, image.Rect(0, 0, win, hin))
			rng.Read(img.Pix)
			for y := range hin / 4 {
				for x := range win {
					img.Pix[y*img.Stride+x*4+3] = byte(1 + rng.Intn(8))
				}
			}
			for i, v := range img.Pix {
				straight[i] = float64(v) / 255
			}
			src, dst = img, newImage(t, image.NewNRGBA, image.Rect(0, 0, wout, hout))
		} else {
			img := newImage(t, image.NewNRGBA64, image.Rect(0, 0, win, hin))
			rng.Read(img.Pix)
			for y := range hin / 4 {
				for x := range win {
					binary.BigEndian.PutUint16(img.Pix[y*img.Stride+x*8+6:], uint16(1+rng.Intn(2048)))
				}
			}
			for i := range straight {
				straight[i] = float64(binary.BigEndian.Uint16(img.Pix[2*i:])) / 65535
			}
			src, dst = img, newImage(t, image.NewNRGBA64, image.Rect(0, 0, wout, hout))
		}
		for i := 0; i < len(pix); i += 4 {
			a := straight[i+3]
			pix[i], pix[i+1], pix[i+2], pix[i+3] = straight[i]*a, straight[i+1]*a, straight[i+2]*a, a
		}
		ref := refResize(pix, win, hin, wout, hout, filter)
		c := prepare(t, dst, src, true, false, filter, 0, premultiplied)
		expect(t, c.Convert(dst, src), nil)
		scale := float64(int(1)<<depth - 1)
		got := func(img image.Image, i int) float64 {
			if depth == 8 {
				return float64(img.(*image.NRGBA).Pix[i])
			}
			return float64(binary.BigEndian.Uint16(img.(*image.NRGBA64).Pix[2*i:]))
		}
		// errors in units of the depth against the reference clamped like the
		// pipeline clamps (the ringing of lanczos gives negative values and
		// colours above the alpha): alpha, premultiplied colour, and straight
		// colour where the alpha is at least 1/255 (low) and 1/8, the division
		// by the alpha amplifying the premultiplied error
		type errs struct{ alpha, pre, colour, colourLow float64 }
		measure := func(pixel func(i int) (a float64, c [3]float64)) errs {
			var e errs
			for i := 0; i < len(ref); i += 4 {
				a := min(max(ref[i+3], 0), 1)
				ga, gc := pixel(i)
				e.alpha = max(e.alpha, math.Abs(ga-a)*scale)
				for k := range 3 {
					c := 0.0
					if a > 0 {
						c = min(max(ref[i+k]/a, 0), 1)
					}
					e.pre = max(e.pre, math.Abs(gc[k]*ga-c*a)*scale)
					if a >= 1.0/255 {
						e.colourLow = max(e.colourLow, math.Abs(gc[k]-c)*scale)
					}
					if a >= 1.0/8 {
						e.colour = max(e.colour, math.Abs(gc[k]-c)*scale)
					}
				}
			}
			return e
		}
		e := measure(func(i int) (float64, [3]float64) {
			return got(dst, i+3) / scale, [3]float64{got(dst, i) / scale, got(dst, i+1) / scale, got(dst, i+2) / scale}
		})
		t.Logf("%T: max error alpha %.2f, premultiplied colour %.2f, colour %.2f (%.2f from alpha 1/255) of %v", src, e.alpha, e.pre, e.colour, e.colourLow, scale)
		// the 14-bit coefficients and the 16-bit rounding of the premultiplied
		// values, amplified by up to 8 in the straight colour
		tol := errs{1, 1, 1, 1}
		if depth == 16 {
			tol = errs{8, 8, 64, 8 * 255}
		}
		if e.alpha > tol.alpha || e.pre > tol.pre || e.colour > tol.colour || e.colourLow > tol.colourLow {
			t.Errorf("%T: errors above %+v", src, tol)
		}
		if depth != 8 {
			continue
		}
		// the same conversion premultiplied at 8 bits, as RGBA
		rgba := newImage(t, image.NewRGBA, image.Rect(0, 0, win, hin))
		for i, v := range pix {
			rgba.Pix[i] = byte(math.Floor(v*255 + 0.5))
		}
		out := newImage(t, image.NewRGBA, image.Rect(0, 0, wout, hout))
		convert(t, out, rgba, true, false, filter)
		e = measure(func(i int) (float64, [3]float64) {
			a := float64(out.Pix[i+3])
			var c [3]float64
			for k := range 3 {
				if a != 0 {
					c[k] = min(math.Floor(float64(out.Pix[i+k])*255/a+0.5), 255) / 255
				}
			}
			return a / 255, c
		})
		t.Logf("%T premultiplied at 8 bits: max error alpha %.2f, premultiplied colour %.2f, colour %.2f (%.2f from alpha 1/255)", src, e.alpha, e.pre, e.colour, e.colourLow)
	}
}

// benchAlphaPass times a pass of ConverterConfig.PremultiplyAlpha over a
// 720x576 plane, single thread.
func benchAlphaPass(b *testing.B, depth int, pass func(threads int, dst, src *plane, depth int), unpremultiply bool) {
	const w, h = 720, 576
	rng := rand.New(rand.NewSource(1))
	bpp := depth / 2
	straight := plane{Data: make([]byte, w*h*bpp), Width: w, Height: h, Pitch: w * bpp, Bpp: bpp}
	pre := plane{Data: make([]byte, w*h*8), Width: w, Height: h, Pitch: w * 8, Bpp: 8}
	rng.Read(straight.Data)
	rng.Read(pre.Data)
	dst, src := &pre, &straight
	if unpremultiply {
		dst, src = &straight, &pre
	}
	b.SetBytes(w * h * int64(bpp))
	for b.Loop() {
		pass(1, dst, src, depth)
	}
}

func BenchmarkPremultiply8(b *testing.B)    { benchAlphaPass(b, 8, premultiply, false) }
func BenchmarkPremultiply16(b *testing.B)   { benchAlphaPass(b, 16, premultiply, false) }
func BenchmarkUnpremultiply8(b *testing.B)  { benchAlphaPass(b, 8, unpremultiply, true) }
func BenchmarkUnpremultiply16(b *testing.B) { benchAlphaPass(b, 16, unpremultiply, true) }
