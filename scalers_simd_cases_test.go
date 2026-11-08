// Copyright 2026 Patrick Gaskin. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

//go:build goexperiment.simd && (amd64 || arm64 || wasm)

package arez

import (
	"bytes"
	"fmt"
	"math/rand"
	"testing"
)

// scalerCase describes a random scaler test. taps, width and input are in
// pixels of pack samples, of 2 bytes when wide, dp, sp and slack in bytes.
// ratio is the largest horizontal offset step between two output pixels, unless
// scale is set, in which case the offsets are those of a regular downscale by
// scale. sparse > 1 keeps only the coefficients of taps t with t%sparse ==
// x%sparse for output pixel x.
type scalerCase struct {
	vertical bool
	wide     bool
	taps     int
	width    int
	height   int
	input    int
	pack     int
	ratio    int
	scale    float64
	sparse   int
	dp       int
	sp       int
	slack    int
	seed     int64
}

func (c scalerCase) String() string {
	dir := "h"
	if c.vertical {
		dir = "v"
	}
	if c.wide {
		dir += "16"
	}
	return fmt.Sprintf("%v taps=%v width=%v height=%v input=%v pack=%v ratio=%v scale=%v sparse=%v dp=%v sp=%v slack=%v seed=%v",
		dir, c.taps, c.width, c.height, c.input, c.pack, c.ratio, c.scale, c.sparse, c.dp, c.sp, c.slack, c.seed)
}

// bytes returns the bytes per pixel of c.
func (c scalerCase) bytes() int {
	if c.wide {
		return 2 * max(c.pack, 1)
	}
	return max(c.pack, 1)
}

// depth returns the bits per sample of c.
func (c scalerCase) depth() int {
	if c.wide {
		return 16
	}
	return 8
}

// unpacked returns c with width and input in bytes and taps in coefficients per
// output sample, as the scalers see them (vertical kernels have one coefficient
// per tap whatever the pack).
func (c scalerCase) unpacked() scalerCase {
	if !c.vertical {
		c.taps *= max(c.pack, 1)
	}
	c.width *= c.bytes()
	c.input *= c.bytes()
	return c
}

type scalerVariant struct {
	name   string
	scaler scaler
	k      kernel
}

func checkScalerVariants(t *testing.T, c scalerCase, variants func(cfg *ResizerConfig, k kernel) []scalerVariant) []scalerVariant {
	t.Helper()
	src, cof, off := makeScalerCase(c)
	cfg := ResizerConfig{Vertical: c.vertical, Input: c.input, Pack: max(c.pack, 1), Depth: c.depth()}
	c = c.unpacked()
	k := kernel{coeffs: cof, offsets: off, size: c.taps, cofscale: 1, wide: c.wide}
	// the source and the outputs are guarded where the platform allows it
	gsrc, free := allocPlane(len(src))
	defer free()
	copy(gsrc, src)
	ref := runScaler(c, goVariant(&cfg, k), gsrc)
	vs := variants(&cfg, k)
	for _, v := range vs {
		out := runScaler(c, v, gsrc)
		if !bytes.Equal(ref, out) {
			t.Fatalf("%v: %v output differs from go\n%v\n%v", c, v.name, ref, out)
		}
	}
	return vs
}

// benchVariant times the scaler returned by variant on 1920 x 16 output bytes,
// 960 samples per row when wide.
func benchVariant(b *testing.B, vertical, wide bool, taps int, variant func(cfg *ResizerConfig, k kernel) scalerVariant) {
	c := scalerCase{
		vertical: vertical,
		wide:     wide,
		taps:     taps,
		width:    1920,
		height:   16,
		dp:       1920,
		seed:     1,
	}
	c.width /= c.bytes()
	if vertical {
		c.input = taps + 16*2
		c.sp = 1920
	} else {
		c.input = c.width * 2
		c.sp = 1920 * 2
	}
	src, cof, off := makeScalerCase(c)
	cfg := ResizerConfig{Vertical: vertical, Input: c.input, Pack: 1, Depth: c.depth()}
	c = c.unpacked()
	v := variant(&cfg, kernel{coeffs: cof, offsets: off, size: c.taps, cofscale: 1, wide: wide})
	dst := make([]byte, c.dp*(c.height-1)+c.width)
	b.SetBytes(int64(c.width * c.height))
	for b.Loop() {
		v.scaler(dst, src, v.k.coeffs, v.k.offsets, v.k.size, c.width, c.height, c.dp, c.sp)
	}
}

// randomScalerCase returns a case with random height, pitches, input size and
// slack after the source plane. Horizontal cases with pack > 1 are expected to
// set dp and sp in bytes after this.
func randomScalerCase(rng *rand.Rand, vertical bool, taps, width int) scalerCase {
	c := scalerCase{
		vertical: vertical,
		taps:     taps,
		width:    width,
		height:   1 + rng.Intn(5),
		dp:       width + rng.Intn(3)*7,
		slack:    rng.Intn(3) * 20,
		seed:     rng.Int63(),
	}
	if vertical {
		c.input = taps + c.height*4
		c.sp = width + rng.Intn(3)*5
	} else {
		c.input = taps + width*2 + rng.Intn(4)
		c.sp = c.input + rng.Intn(3)*5
	}
	return c
}

// makeScalerCase builds random source pixels, offsets and coefficients in the
// natural layout used by the Go scalers, unpacked if c.pack > 1, with byte
// offsets when wide. The coefficients of wide cases are bounded so that no sum
// of 16-bit samples overflows int32.
func makeScalerCase(c scalerCase) (src []byte, cof []int16, off []int32) {
	rng := rand.New(rand.NewSource(c.seed))
	size := c.width
	if c.vertical {
		size = c.height
	}
	cof = make([]int16, size*c.taps)
	lim := 1 << 15 / c.taps
	for i := range cof {
		switch rng.Intn(4) {
		case 0:
			cof[i] = int16(rng.Intn(1<<bits+1) - 1<<(bits-1))
		case 1:
			cof[i] = int16(rng.Intn(1 << 16))
		default:
			cof[i] = int16(rng.Intn(3<<bits) - 1<<bits)
		}
		if c.wide {
			cof[i] = int16(rng.Intn(2*lim+1) - lim)
		}
		if c.sparse > 1 && (i%c.taps)%c.sparse != (i/c.taps)%c.sparse {
			cof[i] = 0
		}
	}
	off = make([]int32, size)
	if c.vertical {
		// relative row offsets, cumulative sum must fit in the input
		total := c.input - c.taps
		pos := 0
		for i := range off {
			step := 0
			if total-pos > 0 {
				step = rng.Intn(min(total-pos, 4) + 1)
			}
			if i == 0 {
				step = rng.Intn(min(total, 2) + 1)
			}
			if c.scale != 0 {
				step = clip(int((float64(i)+0.5)*c.scale)-c.taps/2, 0, total) - pos
			}
			off[i] = int32(step)
			pos += step
		}
		rows := pos + c.taps
		src = make([]byte, c.sp*(rows-1)+c.width*c.bytes()+c.slack)
	} else {
		ratio := c.ratio
		if ratio == 0 {
			ratio = 2
		}
		pos := 0
		for i := range off {
			pos = min(c.input-c.taps, pos+rng.Intn(ratio+1))
			if c.scale != 0 {
				pos = clip(int((float64(i)+0.5)*c.scale)-c.taps/2, 0, c.input-c.taps)
			}
			off[i] = int32(pos)
		}
		pack := max(c.pack, 1)
		if pack > 1 {
			cof, off, _ = unpack(cof, off, c.taps, pack)
		}
		if c.wide {
			for i := range off {
				off[i] *= 2
			}
		}
		src = make([]byte, c.sp*(c.height-1)+c.input*c.bytes()+c.slack)
	}
	rng.Read(src)
	return src, cof, off
}

// runScaler runs the variant v on the unpacked case c. The taps argument is the
// size of the variant's kernel, as Resize passes it: twice the taps of c for
// the widened kernels of 16-bit variants.
func runScaler(c scalerCase, v scalerVariant, src []byte) []byte {
	dst, free := allocPlane(c.dp*(c.height-1) + c.width)
	defer free()
	v.scaler(dst, src, v.k.coeffs, v.k.offsets, v.k.size, c.width, c.height, c.dp, c.sp)
	return bytes.Clone(dst)
}

// checkResizeDispatch resizes a random plane of cfg.Input pixels, 7 rows or
// columns, to cfg.Output pixels with NewResizer and compares the dispatched
// scaler with the Go one.
func checkResizeDispatch(t *testing.T, cfg ResizerConfig, filter Filter) {
	t.Helper()
	const rows = 7
	rng := rand.New(rand.NewSource(int64(cfg.Input*cfg.Output + cfg.Pack)))
	width, height, dw, dh := cfg.Input, rows, cfg.Output, rows
	if cfg.Vertical {
		width, height, dw, dh = rows, cfg.Input, rows, cfg.Output
	}
	bpp := cfg.Pack * cfg.Depth / 8
	sp := width*bpp + 5
	dp := dw*bpp + 3
	src := make([]byte, sp*height)
	rng.Read(src)
	want := make([]byte, dp*dh)
	got := make([]byte, dp*dh)
	cfg.DisableSIMD = true
	r, err := NewResizer(&cfg, filter)
	expect(t, err, nil)
	r.Resize(want, src, width, height, dp, sp)
	cfg.DisableSIMD = false
	r, err = NewResizer(&cfg, filter)
	expect(t, err, nil)
	r.Resize(got, src, width, height, dp, sp)
	if !bytes.Equal(want, got) {
		t.Errorf("%v %+v: output differs from go", filterString(filter), cfg)
	}
}

// TestSIMDDispatchOne resizes planes of every pack and depth to one pixel, from
// inputs that fit a single kernel row or need full width taps, and compares the
// dispatched scalers with the Go ones.
func TestSIMDDispatchOne(t *testing.T) {
	for _, input := range []int{2, 3, 5, 16, 17, 64, 100, 1000, 3840} {
		for _, filter := range []Filter{Bilinear(), Bicubic(), Lanczos(3)} {
			for pack := 1; pack <= 4; pack++ {
				for _, depth := range []int{8, 16} {
					for _, vertical := range []bool{false, true} {
						checkResizeDispatch(t, ResizerConfig{Input: input, Output: 1, Pack: pack, Depth: depth, Vertical: vertical, Threads: 1}, filter)
					}
				}
			}
		}
	}
}

// TestSIMDDispatchPacks resizes planes of 5 to 16 samples per pixel, which only
// NewResizer accepts, in both directions, interlaced too, with 1 to 3 threads,
// and compares the dispatched scalers with the Go ones.
func TestSIMDDispatchPacks(t *testing.T) {
	sizes := []struct{ input, output int }{{17, 5}, {5, 17}, {33, 32}, {64, 9}, {9, 64}}
	i := 0
	for pack := 5; pack <= 16; pack++ {
		for _, depth := range []int{8, 16} {
			for _, filter := range []Filter{Bilinear(), Bicubic(), Lanczos(3)} {
				for _, s := range sizes {
					for _, mode := range []struct{ vertical, interlaced bool }{{false, false}, {true, false}, {true, true}} {
						i++
						checkResizeDispatch(t, ResizerConfig{
							Input:      s.input,
							Output:     s.output,
							Pack:       pack,
							Depth:      depth,
							Vertical:   mode.vertical,
							Interlaced: mode.interlaced,
							Threads:    1 + i%3,
						}, filter)
					}
				}
			}
		}
	}
}

func goVariant(cfg *ResizerConfig, k kernel) scalerVariant {
	if cfg.Vertical {
		return scalerVariant{"go", getVerticalScalerGo(k.size, k.wide), k}
	}
	return scalerVariant{"go", getHorizontalScalerGo(k.size, k.wide), k}
}

// fuzzScalers runs every variant of a backend against the Go scalers on the
// case the fuzz arguments describe, see TestSIMDScalers for the ranges.
func fuzzScalers(f *testing.F, variants func(cfg *ResizerConfig, k kernel) []scalerVariant) {
	f.Add(false, false, uint8(2), uint8(63), uint8(2), uint8(0), uint8(1), uint8(0), uint8(5), int64(1))
	f.Add(false, false, uint8(17), uint8(99), uint8(1), uint8(0), uint8(11), uint8(0), uint8(0), int64(2))
	f.Add(false, false, uint8(5), uint8(16), uint8(3), uint8(3), uint8(1), uint8(2), uint8(9), int64(3))
	f.Add(false, true, uint8(3), uint8(20), uint8(0), uint8(3), uint8(3), uint8(0), uint8(2), int64(4))
	f.Add(true, false, uint8(2), uint8(32), uint8(3), uint8(0), uint8(0), uint8(0), uint8(1), int64(5))
	f.Add(true, true, uint8(6), uint8(7), uint8(2), uint8(0), uint8(0), uint8(0), uint8(4), int64(6))
	f.Fuzz(func(t *testing.T, vertical, wide bool, taps, width, height, pack, ratio, sparse, slack uint8, seed int64) {
		c := scalerCase{
			vertical: vertical,
			wide:     wide,
			taps:     2 + 2*int(taps%33),
			width:    1 + int(width%129),
			height:   1 + int(height%5),
			pack:     1 + int(pack%4),
			ratio:    1 + int(ratio%20),
			sparse:   int(sparse % 5),
			slack:    int(slack%3) * 20,
			seed:     seed,
		}
		if vertical {
			c.input = c.taps + c.height*4
			c.sp = c.width*c.bytes() + int(slack>>2)%3*5
		} else {
			c.input = c.taps + c.width*c.ratio/2 + int(slack>>4)%4
			c.sp = c.input*c.bytes() + int(slack>>2)%3*5
		}
		c.dp = c.width*c.bytes() + int(slack>>6)*7
		checkScalerVariants(t, c, variants)
	})
}
