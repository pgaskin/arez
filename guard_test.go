// Copyright 2026 Patrick Gaskin. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

//go:build unix

package arez

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"image"
	"math/rand"
	"sync/atomic"
	"syscall"
	"testing"
)

// If supported, -arez.guard puts image planes next to inaccessible pages so
// out-of-bounds reads/writes segfault.

func init() {
	flag.Var(&guard, "arez.guard", "allocate every test image on guarded memory, `mode` back (ending before an inaccessible page), front (starting after one) or both")
	allocPlane = guardAlloc
}

func TestGuardedConvert(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	n := 24
	if testing.Short() {
		n = 6
	}
	sizes := []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 100, 129}
	for _, it := range testImageTypes {
		for i := range n {
			srcRect := image.Rect(0, 0, sizes[rng.Intn(len(sizes))], sizes[rng.Intn(len(sizes))])
			dstRect := image.Rect(0, 0, sizes[rng.Intn(len(sizes))], sizes[rng.Intn(len(sizes))])
			switch i % 4 {
			case 1:
				dstRect.Max.X = srcRect.Max.X
			case 2:
				dstRect.Max.Y = srcRect.Max.Y
			}
			interlaced := rng.Intn(2) == 0
			filter := filters[rng.Intn(len(filters))]
			threads := 1 + rng.Intn(4)
			slack := []int{0, 1, 3, 16, 41}[rng.Intn(5)]
			src := guardedImage(t, it.new, srcRect, []int{0, 1, 7}[rng.Intn(3)], 0)
			fillImage(rng, src)
			var outs [][]byte
			for _, simd := range []bool{true, false} {
				if !simd && !hasSIMD() {
					break
				}
				dst := guardedImage(t, it.new, dstRect, slack, 0xA5)
				cfg, err := Prepare(dst, src)
				expect(t, err, nil)
				cfg.Input.Interlaced, cfg.Output.Interlaced = interlaced, interlaced
				cfg.DisableSIMD = !simd
				cfg.Threads = threads
				c, err := NewConverter(cfg, filter)
				if err != nil {
					break
				}
				expect(t, c.Convert(dst, src), nil)
				checkPadding(t, dst, 0xA5)
				outs = append(outs, imageBytes(dst))
			}
			if len(outs) == 2 && !bytes.Equal(outs[0], outs[1]) {
				t.Errorf("%v %v -> %v interlaced=%v %v threads=%v: simd differs from go", it.name, srcRect.Size(), dstRect.Size(), interlaced, filterString(filter), threads)
			}
		}
	}
}

func TestGuardedResize(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	n := 600
	if testing.Short() {
		n = 100
	}
	for range n {
		c := guardedResizeCase{
			cfg: ResizerConfig{
				Depth:      []int{8, 8, 16}[rng.Intn(3)],
				Input:      2 + rng.Intn(200),
				Output:     1 + rng.Intn(200),
				Vertical:   rng.Intn(2) == 0,
				Interlaced: rng.Intn(3) == 0,
				Pack:       []int{1, 1, 1, 2, 3, 4}[rng.Intn(6)],
				Threads:    1 + rng.Intn(4),
			},
			filter: rng.Intn(len(guardedFilters)),
			other:  1 + rng.Intn(70),
			sslack: []int{0, 0, 1, 5, 16, 33}[rng.Intn(6)],
			dslack: []int{0, 0, 1, 5, 16, 33}[rng.Intn(6)],
		}
		switch rng.Intn(12) {
		case 0:
			c.pitch = pitchOverlap
			c.sslack = rng.Intn(1 << 16)
		case 1:
			c.pitch = pitchZero
		}
		c.seed = rng.Int63()
		runGuardedResize(t, c)
	}
}

// FuzzGuardedResize is TestGuardedResize over fuzzed cases.
func FuzzGuardedResize(f *testing.F) {
	f.Add(false, uint16(200), uint16(100), false, false, uint8(0), uint8(0), uint8(0), uint8(7), uint16(0), uint16(0), uint8(0), int64(1))
	f.Add(true, uint16(17), uint16(64), true, true, uint8(3), uint8(2), uint8(2), uint8(33), uint16(5), uint16(16), uint8(0), int64(2))
	f.Add(false, uint16(64), uint16(32), true, false, uint8(0), uint8(1), uint8(0), uint8(32), uint16(16), uint16(0), uint8(1), int64(3))
	f.Add(false, uint16(9), uint16(3), true, true, uint8(2), uint8(3), uint8(3), uint8(5), uint16(0), uint16(1), uint8(2), int64(4))
	f.Add(true, uint16(511), uint16(37), false, false, uint8(1), uint8(0), uint8(4), uint8(69), uint16(33), uint16(3), uint8(0), int64(5))
	f.Add(false, uint16(3), uint16(500), false, false, uint8(2), uint8(1), uint8(1), uint8(1), uint16(1), uint16(64), uint8(1), int64(6))
	f.Fuzz(func(t *testing.T, wide bool, input, output uint16, vertical, interlaced bool, pack, threads, filter, other uint8, sslack, dslack uint16, pitch uint8, seed int64) {
		c := guardedResizeCase{
			cfg: ResizerConfig{
				Depth:      8,
				Input:      2 + int(input%511),
				Output:     1 + int(output%512),
				Vertical:   vertical,
				Interlaced: interlaced,
				Pack:       1 + int(pack%4),
				Threads:    1 + int(threads%4),
			},
			filter: int(filter) % len(guardedFilters),
			other:  1 + int(other%70),
			sslack: int(sslack % 65),
			dslack: int(dslack % 65),
			pitch:  int(pitch % 3),
			seed:   seed,
		}
		if wide {
			c.cfg.Depth = 16
		}
		if c.pitch == pitchOverlap {
			c.sslack = int(sslack)
		}
		runGuardedResize(t, c)
	})
}

var guardedFilters = append(filters, Lanczos(4), CustomBicubic(1.0/3, 1.0/3), Nearest(), Box())

const (
	pitchSlack   = iota // rows sslack bytes longer than their pixels
	pitchOverlap        // rows overlapping by sslack modulo the row bytes
	pitchZero           // one row repeated
)

type guardedResizeCase struct {
	cfg            ResizerConfig
	filter         int
	other          int
	sslack, dslack int
	pitch          int
	seed           int64
}

func runGuardedResize(t testing.TB, c guardedResizeCase) {
	t.Helper()
	cfg := c.cfg
	if cfg.Interlaced && cfg.Input < 4 {
		cfg.Input = 4
	}
	filter := guardedFilters[c.filter]
	bpp := cfg.Pack * (cfg.Depth / 8)
	width, height, columns, rows := cfg.Input, c.other, cfg.Output, c.other
	if cfg.Vertical {
		width, height, columns, rows = c.other, cfg.Input, c.other, cfg.Output
	}
	sp, dp := width*bpp+c.sslack, columns*bpp+c.dslack
	switch c.pitch {
	case pitchOverlap:
		sp = width*bpp - c.sslack%(width*bpp)
	case pitchZero:
		sp = 0
	}
	what := fmt.Sprintf("%+v %v %vx%v sp=%v dp=%v", cfg, filterString(filter), width, height, sp, dp)
	src, freeSrc := guardAlloc(sp*(height-1) + width*bpp)
	defer freeSrc()
	rand.New(rand.NewSource(c.seed)).Read(src)
	var outs [][]byte
	for _, simd := range []bool{true, false} {
		if !simd && !hasSIMD() {
			break
		}
		cfg.DisableSIMD = !simd
		r, err := NewResizer(&cfg, filter)
		expect(t, err, nil)
		dst, freeDst := guardAlloc(dp*(rows-1) + columns*bpp)
		for i := range dst {
			dst[i] = 0xA5
		}
		r.Resize(dst, src, width, height, dp, sp)
		for y := range rows - 1 {
			for x := columns * bpp; x < dp; x++ {
				if dst[y*dp+x] != 0xA5 {
					t.Fatalf("%v simd=%v: padding byte %v of row %v written", what, simd, x, y)
				}
			}
		}
		outs = append(outs, bytes.Clone(dst))
		freeDst()
	}
	if len(outs) == 2 && !bytes.Equal(outs[0], outs[1]) {
		t.Errorf("%v: simd differs from go", what)
	}
}

func guardImage(tb testing.TB, img image.Image) {
	tb.Helper()
	if guard == guardOff {
		return
	}
	switch img := img.(type) {
	case *image.Gray:
		img.Pix, img.Stride = guardPlane(tb, img.Pix, img.Stride, img.Rect.Dx(), img.Rect.Dy())
	case *image.Alpha:
		img.Pix, img.Stride = guardPlane(tb, img.Pix, img.Stride, img.Rect.Dx(), img.Rect.Dy())
	case *image.Paletted:
		img.Pix, img.Stride = guardPlane(tb, img.Pix, img.Stride, img.Rect.Dx(), img.Rect.Dy())
	case *image.Gray16:
		img.Pix, img.Stride = guardPlane(tb, img.Pix, img.Stride, img.Rect.Dx()*2, img.Rect.Dy())
	case *image.Alpha16:
		img.Pix, img.Stride = guardPlane(tb, img.Pix, img.Stride, img.Rect.Dx()*2, img.Rect.Dy())
	case *image.RGBA:
		img.Pix, img.Stride = guardPlane(tb, img.Pix, img.Stride, img.Rect.Dx()*4, img.Rect.Dy())
	case *image.NRGBA:
		img.Pix, img.Stride = guardPlane(tb, img.Pix, img.Stride, img.Rect.Dx()*4, img.Rect.Dy())
	case *image.CMYK:
		img.Pix, img.Stride = guardPlane(tb, img.Pix, img.Stride, img.Rect.Dx()*4, img.Rect.Dy())
	case *image.RGBA64:
		img.Pix, img.Stride = guardPlane(tb, img.Pix, img.Stride, img.Rect.Dx()*8, img.Rect.Dy())
	case *image.NRGBA64:
		img.Pix, img.Stride = guardPlane(tb, img.Pix, img.Stride, img.Rect.Dx()*8, img.Rect.Dy())
	case *image.YCbCr:
		guardYCbCr(tb, img)
	case *image.NYCbCrA:
		guardYCbCr(tb, &img.YCbCr)
		img.A, img.AStride = guardPlane(tb, img.A, img.AStride, img.Rect.Dx(), img.Rect.Dy())
	default:
		panic(fmt.Sprintf("guardImage: unsupported image type %T", img))
	}
}

func guardYCbCr(tb testing.TB, img *image.YCbCr) {
	tb.Helper()
	r := img.Rect
	sx, sy := 1, 1
	if ratio, ok := ChromaRatioOf(img.SubsampleRatio); ok {
		sx, sy = subsample(ratio, 1)
	}
	cw, ch := (r.Max.X+sx-1)/sx-r.Min.X/sx, (r.Max.Y+sy-1)/sy-r.Min.Y/sy
	img.Y, img.YStride = guardPlane(tb, img.Y, img.YStride, r.Dx(), r.Dy())
	stride := img.CStride
	img.Cb, img.CStride = guardPlane(tb, img.Cb, stride, cw, ch)
	img.Cr, _ = guardPlane(tb, img.Cr, stride, cw, ch)
}

func guardPlane(tb testing.TB, pix []byte, stride, width, height int) ([]byte, int) {
	tb.Helper()
	buf, free := guardAlloc(width * height)
	tb.Cleanup(free)
	for y := range height {
		copy(buf[y*width:(y+1)*width], pix[y*stride:])
	}
	return buf, width
}

// guardAlloc returns size bytes ending right before a page without access, or
// starting right after one (based on -arez.guard).
func guardAlloc(size int) ([]byte, func()) {
	front := guard == guardFront || guard == guardBoth && guardCount.Add(1)%2 == 0
	return guardAllocAt(size, front)
}

var guardCount atomic.Uint64

func guardAllocAt(size int, front bool) ([]byte, func()) {
	page := syscall.Getpagesize()
	n := (size + page - 1) / page * page
	mem, err := syscall.Mmap(-1, 0, n+2*page, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		panic(err)
	}
	if err := syscall.Mprotect(mem[:page], syscall.PROT_NONE); err != nil {
		panic(err)
	}
	if err := syscall.Mprotect(mem[page+n:], syscall.PROT_NONE); err != nil {
		panic(err)
	}
	start := page + n - size
	if front {
		start = page
	}
	return mem[start : start+size : start+size], func() { syscall.Munmap(mem) }
}

type guardMode int

const (
	guardOff guardMode = iota // only enable specifically for guard tests
	guardBack
	guardFront
	guardBoth
)

var guard guardMode

var guardModes = []string{guardOff: "", guardBack: "back", guardFront: "front", guardBoth: "both"}

func (m guardMode) String() string { return guardModes[m] }

func (m *guardMode) Set(s string) error {
	for i, name := range guardModes {
		if s == name && s != "" {
			*m = guardMode(i)
			return nil
		}
	}
	return errors.New("want back, front or both")
}

func guardedImage(t testing.TB, ctor func(image.Rectangle) image.Image, r image.Rectangle, slack int, fill byte) image.Image {
	t.Helper()
	img := ctor(r)
	_, planes, err := inspect(img, false)
	expect(t, err, nil)
	pix := make([][]byte, len(planes))
	pitches := make([]int, len(planes))
	for i, p := range planes {
		pitches[i] = p.Width*p.Bpp + slack
		var free func()
		pix[i], free = guardAlloc(pitches[i]*(p.Height-1) + p.Width*p.Bpp)
		t.Cleanup(free)
		for j := range pix[i] {
			pix[i][j] = fill
		}
	}
	switch img := img.(type) {
	case *image.YCbCr:
		img.Y, img.Cb, img.Cr, img.YStride, img.CStride = pix[0], pix[1], pix[2], pitches[0], pitches[1]
	case *image.NYCbCrA:
		img.Y, img.Cb, img.Cr, img.YStride, img.CStride = pix[0], pix[1], pix[2], pitches[0], pitches[1]
		img.A, img.AStride = pix[3], pitches[3]
	case *image.Gray:
		img.Pix, img.Stride = pix[0], pitches[0]
	case *image.Alpha:
		img.Pix, img.Stride = pix[0], pitches[0]
	case *image.RGBA:
		img.Pix, img.Stride = pix[0], pitches[0]
	case *image.NRGBA:
		img.Pix, img.Stride = pix[0], pitches[0]
	case *image.CMYK:
		img.Pix, img.Stride = pix[0], pitches[0]
	case *image.Gray16:
		img.Pix, img.Stride = pix[0], pitches[0]
	case *image.Alpha16:
		img.Pix, img.Stride = pix[0], pitches[0]
	case *image.RGBA64:
		img.Pix, img.Stride = pix[0], pitches[0]
	case *image.NRGBA64:
		img.Pix, img.Stride = pix[0], pitches[0]
	default:
		t.Fatalf("unexpected image type %T", img)
	}
	return img
}

// checkPadding checks that the bytes between the rows of the planes of img
// still hold v.
func checkPadding(t testing.TB, img image.Image, v byte) {
	t.Helper()
	_, planes, err := inspect(img, false)
	expect(t, err, nil)
	for i, p := range planes {
		for y := range p.Height - 1 {
			for x := p.Width * p.Bpp; x < p.Pitch; x++ {
				if p.Data[y*p.Pitch+x] != v {
					t.Fatalf("%T plane %v: padding byte %v of row %v written", img, i, x, y)
				}
			}
		}
	}
}
