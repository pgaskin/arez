// Copyright 2026 Patrick Gaskin. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

package arez

import (
	"bytes"
	"encoding/binary"
	"errors"
	"image"
	"math"
	"math/rand"
	"testing"
)

// Slow float reference implementation of conversions to compare against.

type refPlane struct {
	w, h, ch int       // width, height (rows), channels
	v        []float64 // pixels
}

func refLoad(p *plane, depth int) refPlane {
	ch := p.Bpp * 8 / depth
	r := refPlane{p.Width, p.Height, ch, make([]float64, p.Width*p.Height*ch)}
	for y := range p.Height {
		row := p.Data[y*p.Pitch:]
		for i := range p.Width * ch {
			if depth == 16 {
				r.v[y*p.Width*ch+i] = float64(binary.BigEndian.Uint16(row[2*i:]))
			} else {
				r.v[y*p.Width*ch+i] = float64(row[i])
			}
		}
	}
	return r
}

func refScale(r refPlane, output int, vertical, interlaced bool, filter Filter, maxv float64) refPlane {
	input := r.w
	if vertical {
		input = r.h
	}
	cfg := ResizerConfig{Input: input, Output: output, Vertical: vertical, Interlaced: interlaced && vertical}
	field := bin(cfg.Interlaced)
	out := refPlane{output, r.h, r.ch, nil}
	if vertical {
		out = refPlane{r.w, output, r.ch, nil}
	}
	out.v = make([]float64, out.w*out.h*out.ch)
	for idx := uint(0); idx <= field; idx++ {
		pos, sums, weights, taps, size := makeDoubleKernel(&cfg, filter, field, idx)
		for i := range size {
			off := (pos[i] + int(field-idx)) >> field
			o := i<<field + int(idx)
			for j := range taps {
				w := weights[i*taps+j] / sums[i]
				s := (off+j)<<field + int(idx)
				if vertical {
					for x := range r.w * r.ch {
						out.v[o*out.w*out.ch+x] += w * r.v[s*r.w*r.ch+x]
					}
					continue
				}
				for y := range r.h {
					for c := range r.ch {
						out.v[(y*out.w+o)*out.ch+c] += w * r.v[(y*r.w+s)*r.ch+c]
					}
				}
			}
		}
	}
	for i, v := range out.v {
		out.v[i] = min(max(v, 0), maxv)
	}
	return out
}

func refConvert(t testing.TB, dst, src image.Image, interlaced, premultiply bool, filter Filter) []refPlane {
	t.Helper()
	sd, sp, err := inspect(src, interlaced)
	expect(t, err, nil)
	_, dp, err := inspect(dst, interlaced)
	expect(t, err, nil)
	maxv := float64(int(1)<<sd.depth() - 1)
	var out []refPlane
	for i := range sp {
		r := refLoad(&sp[i], sd.depth())
		if premultiply {
			for p := 0; p < len(r.v); p += 4 {
				a := r.v[p+3] / maxv
				r.v[p], r.v[p+1], r.v[p+2] = r.v[p]*a, r.v[p+1]*a, r.v[p+2]*a
			}
		}
		if dp[i].Height != r.h {
			r = refScale(r, dp[i].Height, true, interlaced, filter, maxv)
		}
		if dp[i].Width != r.w {
			r = refScale(r, dp[i].Width, false, false, filter, maxv)
		}
		if premultiply {
			for p := 0; p < len(r.v); p += 4 {
				if a := r.v[p+3] / maxv; a > 0 {
					r.v[p], r.v[p+1], r.v[p+2] = min(r.v[p]/a, maxv), min(r.v[p+1]/a, maxv), min(r.v[p+2]/a, maxv)
				} else {
					r.v[p], r.v[p+1], r.v[p+2] = 0, 0, 0
				}
			}
		}
		out = append(out, r)
	}
	return out
}

// checkReference compares the planes of img with ref within tol, the
// colours of premultiplied pixels only where the alpha is at least a
// quarter (the division by the alpha amplifies the fixed point error).
func checkReference(t testing.TB, img image.Image, ref []refPlane, tol float64, premultiply bool, what string) {
	t.Helper()
	d, planes, err := inspect(img, false)
	expect(t, err, nil)
	maxv := float64(int(1)<<d.depth() - 1)
	for i, p := range planes {
		r := refLoad(&p, d.depth())
		expect(t, r.w, ref[i].w)
		expect(t, r.h, ref[i].h)
		for j, v := range r.v {
			want := ref[i].v[j]
			ctol := tol
			if premultiply && j%4 != 3 {
				if ref[i].v[j/4*4+3] < maxv/4 {
					continue
				}
				ctol = 4 * tol
			}
			if math.Abs(v-want) > ctol {
				t.Fatalf("%v: plane %v sample %v (pixel %v,%v) is %v, reference %.2f", what, i, j, j/r.ch%r.w, j/r.ch/r.w, v, want)
			}
		}
	}
}

type convertCase struct {
	typ         int
	wsrc, hsrc  int
	wdst, hdst  int
	ox, oy      int
	px, py      int
	interlaced  bool
	filter      int
	threads     int
	premultiply bool
	seed        int64
}

var convertFilters = []Filter{Bilinear(), Bicubic(), Lanczos(3), Lanczos(4)}

func subImageEnding(tb testing.TB, ctor func(image.Rectangle) image.Image, w, h, px, py, ox, oy int) image.Image {
	tb.Helper()
	parent := newImage(tb, ctor, image.Rect(px, py, px+ox+w, py+oy+h))
	return parent.(subImager).SubImage(image.Rect(px+ox, py+oy, px+ox+w, py+oy+h))
}

func runConvertCase(t testing.TB, c convertCase) bool {
	t.Helper()
	it := testImageTypes[c.typ%len(testImageTypes)]
	filter := convertFilters[c.filter%len(convertFilters)]
	rng := rand.New(rand.NewSource(c.seed))
	src := subImageEnding(t, it.new, c.wsrc, c.hsrc, c.px, c.py, c.ox, c.oy)
	fillImage(rng, src)
	if c.premultiply && !isStraight(src) {
		c.premultiply = false
	}
	var ref []refPlane
	var outs [][]byte
	for _, simd := range []bool{true, false} {
		if !simd && !hasSIMD() {
			break
		}
		dst := subImageEnding(t, it.new, c.wdst, c.hdst, c.px, c.py, c.ox, c.oy)
		cfg, err := Prepare(dst, src)
		expect(t, err, nil)
		cfg.Input.Interlaced, cfg.Output.Interlaced = c.interlaced, c.interlaced
		cfg.DisableSIMD = !simd
		cfg.Threads = c.threads
		cfg.PremultiplyAlpha = c.premultiply
		conv, err := NewConverter(cfg, filter)
		if errors.Is(err, ErrTooSmall) {
			return false
		}
		expect(t, err, nil)
		expect(t, conv.Convert(dst, src), nil)
		if ref == nil {
			ref = refConvert(t, dst, src, c.interlaced, c.premultiply, filter)
		}
		// measured at most 1.2 and 20 over random cases
		tol := 3.0
		if cfg.Input.depth() == 16 {
			tol = 128
		}
		checkReference(t, dst, ref, tol, c.premultiply, it.name)
		outs = append(outs, imageBytes(dst))
	}
	if len(outs) == 2 && !bytes.Equal(outs[0], outs[1]) {
		t.Fatalf("%v %+v: simd differs from go", it.name, c)
	}
	return true
}

func TestConvertReference(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	n := 400
	if testing.Short() {
		n = 60
	}
	sizes := []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 100}
	ran := 0
	for i := 0; ran < n; i++ {
		c := convertCase{
			typ:         i,
			wsrc:        sizes[rng.Intn(len(sizes))],
			hsrc:        sizes[rng.Intn(len(sizes))],
			wdst:        sizes[rng.Intn(len(sizes))],
			hdst:        sizes[rng.Intn(len(sizes))],
			ox:          rng.Intn(4),
			oy:          rng.Intn(4),
			interlaced:  rng.Intn(3) == 0,
			filter:      rng.Intn(len(convertFilters)),
			threads:     1 + rng.Intn(4),
			premultiply: rng.Intn(2) == 0,
			seed:        rng.Int63(),
		}
		if rng.Intn(4) == 0 {
			c.wdst = c.wsrc
		}
		if rng.Intn(4) == 0 {
			c.hdst = c.hsrc
		}
		if runConvertCase(t, c) {
			ran++
		}
	}
}

func TestOnePixelOutputs(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	srcs := []image.Point{{2, 2}, {3, 5}, {17, 9}, {64, 33}, {300, 200}}
	dsts := []image.Point{{1, 1}, {2, 2}, {1, 7}, {9, 1}, {2, 1}, {1, 2}, {3, 3}}
	if !testing.Short() {
		srcs = append(srcs, image.Point{1920, 1080})
	}
	for typ, it := range testImageTypes {
		for _, s := range srcs {
			for _, d := range dsts {
				c := convertCase{
					typ:         typ,
					wsrc:        s.X,
					hsrc:        s.Y,
					wdst:        d.X,
					hdst:        d.Y,
					ox:          rng.Intn(4),
					oy:          rng.Intn(4),
					interlaced:  rng.Intn(3) == 0,
					filter:      rng.Intn(len(convertFilters)),
					threads:     1 + rng.Intn(4),
					premultiply: rng.Intn(2) == 0,
					seed:        rng.Int63(),
				}
				if !runConvertCase(t, c) && s.X >= 17 {
					t.Errorf("%v %+v: too small", it.name, c)
				}
			}
		}
	}
}

func FuzzConvert(f *testing.F) {
	f.Add(uint8(9), uint8(16), uint8(16), uint8(24), uint8(24), uint8(0), uint8(0), false, uint8(2), uint8(1), false, int64(1))
	f.Add(uint8(3), uint8(33), uint8(7), uint8(5), uint8(31), uint8(3), uint8(1), true, uint8(1), uint8(3), true, int64(2))
	f.Add(uint8(15), uint8(64), uint8(9), uint8(17), uint8(2), uint8(1), uint8(2), true, uint8(3), uint8(4), false, int64(3))
	f.Add(uint8(8), uint8(5), uint8(100), uint8(65), uint8(3), uint8(0), uint8(1), false, uint8(0), uint8(2), true, int64(4))
	f.Add(uint8(14), uint8(16), uint8(16), uint8(4), uint8(8), uint8(2), uint8(3), true, uint8(2), uint8(1), false, int64(5))
	f.Fuzz(func(t *testing.T, typ, wsrc, hsrc, wdst, hdst, ox, oy uint8, interlaced bool, filter, threads uint8, premultiply bool, seed int64) {
		runConvertCase(t, convertCase{
			typ:         int(typ),
			wsrc:        1 + int(wsrc%129),
			hsrc:        1 + int(hsrc%129),
			wdst:        1 + int(wdst%129),
			hdst:        1 + int(hdst%129),
			ox:          int(ox % 4),
			oy:          int(oy % 4),
			interlaced:  interlaced,
			filter:      int(filter),
			threads:     1 + int(threads%4),
			premultiply: premultiply,
			seed:        seed,
		})
	})
}
