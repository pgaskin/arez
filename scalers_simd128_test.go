// Copyright 2026 Patrick Gaskin. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

//go:build goexperiment.simd && (arm64 || wasm)

package arez

import (
	"math/rand"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

// simd128Variants returns every 128-bit scaler able to run k, the first one
// being the one which would be picked by dispatch. The variants of 16-bit
// kernels are named with a w suffix.
func simd128Variants(cfg *ResizerConfig, k kernel) []scalerVariant {
	ks := prepareKernelSIMD128(cfg, k)
	pack := cfg.Pack
	suffix := ""
	hN, hTbl16, hTbl32, hP8 := h8scaleNSIMD128, h8scaleTbl16SIMD128, h8scaleTbl32SIMD128, h8scaleP8SIMD128
	if k.wide {
		suffix = "w"
		hN, hTbl16, hTbl32, hP8 = h16scaleNSIMD128, h16scaleTbl16SIMD128, h16scaleTbl32SIMD128, h16scaleP8SIMD128
		if !cfg.Vertical {
			k = widenKernel(k)
			pack *= 2
		}
	}
	if cfg.Vertical {
		return []scalerVariant{{"dispatch" + suffix, getVerticalScalerSIMD128(&ks, true), ks}}
	}
	v := []scalerVariant{{"dispatch" + suffix, getHorizontalScalerSIMD128(&ks, true), ks}}
	v = append(v, scalerVariant{"hN" + suffix, hN, padKernel(k, 8, simd128Pixels)})
	input := cfg.Input * pack
	if cof, off := prepareHorizontalCoeffsSIMD128Packed(k.coeffs, k.offsets, k.size, 4); pack == 4 && !k.wide && cof != nil {
		kp := k
		kp.coeffs, kp.offsets = cof, off
		v = append(v, scalerVariant{"hP4", h8scaleP4SIMD128, kp})
	}
	if cof, off := prepareHorizontalCoeffsSIMD128Packed(k.coeffs, k.offsets, k.size, 8); pack == 8 && cof != nil {
		kp := k
		kp.coeffs, kp.offsets = cof, off
		v = append(v, scalerVariant{"hP8" + suffix, hP8, kp})
	}
	if cof, off := prepareHorizontalCoeffsSIMD128Tbl(k.coeffs, k.offsets, k.size, input, 16); cof != nil {
		kt := k
		kt.coeffs, kt.offsets = cof, off
		v = append(v, scalerVariant{"hTbl16" + suffix, hTbl16, kt})
	}
	if cof, off := prepareHorizontalCoeffsSIMD128Tbl(k.coeffs, k.offsets, k.size, input, 32); cof != nil {
		kt := k
		kt.coeffs, kt.offsets = cof, off
		v = append(v, scalerVariant{"hTbl32" + suffix, hTbl32, kt})
	}
	return v
}

// simd128Dispatch returns the 128-bit scaler dispatch would pick for k.
func simd128Dispatch(cfg *ResizerConfig, k kernel) scalerVariant {
	return simd128Variants(cfg, k)[0]
}

// checkSIMD128Case checks every 128-bit variant of c against Go and returns the
// name of the scaler dispatch picked.
func checkSIMD128Case(t *testing.T, c scalerCase) string {
	t.Helper()
	v := checkScalerVariants(t, c, simd128Variants)[0]
	name := runtime.FuncForPC(reflect.ValueOf(v.scaler).Pointer()).Name()
	return name[strings.LastIndex(name, ".")+1:]
}

func TestSIMD128Enabled(t *testing.T) {
	cases := []struct {
		cfg    ResizerConfig
		taps   int
		layout int
		h, v   scaler
	}{
		{ResizerConfig{Input: 100, Output: 200, Pack: 1}, 6, layoutSIMD128Tbl16, h8scaleTbl16SIMD128, nil},
		{ResizerConfig{Input: 1920, Output: 960, Pack: 1}, 12, layoutSIMD128Tbl32, h8scaleTbl32SIMD128, nil},
		{ResizerConfig{Input: 1920, Output: 640, Pack: 1}, 18, layoutSIMD128N, h8scaleNSIMD128, nil},
		{ResizerConfig{Input: 3840, Output: 320, Pack: 1}, 72, layoutSIMD128N, h8scaleNSIMD128, nil},
		{ResizerConfig{Input: 3840, Output: 320, Pack: 4}, 288, layoutSIMD128P4, h8scaleP4SIMD128, nil},
		{ResizerConfig{Input: 3840, Output: 320, Vertical: true}, 72, 0, nil, v8scaleNSIMD128},
		{ResizerConfig{Input: 100, Output: 200, Pack: 4}, 24, layoutSIMD128P4, h8scaleP4SIMD128, nil},
		{ResizerConfig{Input: 1920, Output: 640, Pack: 4}, 72, layoutSIMD128P4, h8scaleP4SIMD128, nil},
		{ResizerConfig{Input: 100, Output: 200, Vertical: true}, 6, 0, nil, v8scaleNSIMD128},
		{ResizerConfig{Input: 1920, Output: 960, Vertical: true}, 12, 0, nil, v8scaleNSIMD128},
		// 16-bit kernels are widened to bytes, twice the taps
		{ResizerConfig{Input: 100, Output: 200, Pack: 1, Depth: 16}, 12, layoutSIMD128Tbl16, h16scaleTbl16SIMD128, nil},
		{ResizerConfig{Input: 1920, Output: 1280, Pack: 1, Depth: 16}, 20, layoutSIMD128Tbl32, h16scaleTbl32SIMD128, nil},
		{ResizerConfig{Input: 1920, Output: 960, Pack: 1, Depth: 16}, 24, layoutSIMD128N, h16scaleNSIMD128, nil},
		{ResizerConfig{Input: 100, Output: 200, Pack: 4, Depth: 16}, 48, layoutSIMD128P8, h16scaleP8SIMD128, nil},
		{ResizerConfig{Input: 1920, Output: 640, Pack: 4, Depth: 16}, 144, layoutSIMD128P8, h16scaleP8SIMD128, nil},
		{ResizerConfig{Input: 3840, Output: 320, Pack: 4, Depth: 16}, 576, layoutSIMD128P8, h16scaleP8SIMD128, nil},
		{ResizerConfig{Input: 100, Output: 200, Vertical: true, Depth: 16}, 6, 0, nil, v16scaleNSIMD128},
	}
	for _, c := range cases {
		k, err := makeKernel(&c.cfg, Lanczos(3), 0)
		expect(t, err, nil)
		expect(t, k.size, c.taps)
		expect(t, k.layout, c.layout)
		if c.cfg.Vertical {
			s := getVerticalScaler(&k, true)
			expect(t, reflect.ValueOf(s).Pointer(), reflect.ValueOf(c.v).Pointer())
		} else {
			s := getHorizontalScaler(&k, true)
			expect(t, reflect.ValueOf(s).Pointer(), reflect.ValueOf(c.h).Pointer())
		}
	}
}

func TestSIMD128Scalers(t *testing.T) {
	taps := []int{2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 30, 32, 34, 48, 64, 66}
	widths := []int{1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 100, 127, 128, 129}
	rng := rand.New(rand.NewSource(1))
	dispatched := map[string]int{}
	for _, vertical := range []bool{false, true} {
		for _, taps := range taps {
			for _, width := range widths {
				for i := range 6 {
					c := randomScalerCase(rng, vertical, taps, width)
					// 16-bit cases as Gray16 and RGBA64 rows
					c.wide = i >= 4
					if c.wide {
						c.pack = 1 + 3*(i-4)
						c.dp = width*c.bytes() + rng.Intn(3)*7
					}
					if !vertical {
						// mostly upscales, sparse kernels like the
						// unpacked Pack > 1 ones, and large downscales
						c.ratio = 1 + rng.Intn(2)
						if i == 2 || i == 5 {
							c.ratio = []int{6, 12, 20}[rng.Intn(3)]
							c.input = taps + width*c.ratio/2 + rng.Intn(4)
						}
						c.sp = c.input*c.bytes() + rng.Intn(3)*5
						if i == 3 {
							c.sparse = []int{2, 3, 4}[rng.Intn(3)]
						}
					} else {
						c.sp = width*c.bytes() + rng.Intn(3)*5
					}
					dispatched[checkSIMD128Case(t, c)]++
				}
			}
		}
	}
	for _, taps := range []int{2, 4, 6, 8, 10, 12, 18, 30} {
		for _, pixels := range []int{1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 33} {
			for i := range 4 {
				for _, pack := range []int{4, 8} {
					c := randomScalerCase(rng, false, taps, pixels)
					c.pack = pack
					c.ratio = []int{1, 2, 8, 16}[i]
					c.input = taps + pixels*c.ratio/2 + rng.Intn(4)
					c.dp = pixels*pack + rng.Intn(3)*7
					c.sp = c.input*pack + rng.Intn(3)*5
					dispatched[checkSIMD128Case(t, c)]++
				}
			}
		}
	}
	t.Logf("cases per scaler: %v", dispatched)
	for _, s := range []string{
		"h8scaleNSIMD128", "h8scaleTbl16SIMD128", "h8scaleTbl32SIMD128", "h8scaleP4SIMD128", "h8scaleP8SIMD128", "v8scaleNSIMD128",
		"h16scaleNSIMD128", "h16scaleTbl16SIMD128", "h16scaleTbl32SIMD128", "h16scaleP8SIMD128", "v16scaleNSIMD128",
	} {
		if dispatched[s] == 0 {
			t.Fatalf("scaler %v not covered", s)
		}
	}
}

// TestSIMD128DispatchLarge runs thumbnail sized reductions through NewResizer,
// comparing the dispatched kernels with the Go scalers.
func TestSIMD128DispatchLarge(t *testing.T) {
	tests := []struct {
		input, output, pack, depth int
		filter                     Filter
	}{
		{3840, 320, 1, 8, Lanczos(3)},
		{3840, 320, 1, 8, Bicubic()},
		{1920, 160, 1, 8, Lanczos(3)},
		{1920, 100, 1, 8, Lanczos(3)},
		{3840, 320, 4, 8, Lanczos(3)},
		{1920, 160, 2, 8, Bicubic()},
		{1920, 160, 3, 8, Bicubic()},
		{3840, 320, 1, 16, Lanczos(3)},
		{1920, 160, 1, 16, Bicubic()},
		{3840, 320, 4, 16, Lanczos(3)},
		{1920, 160, 4, 16, Bicubic()},
	}
	for _, tt := range tests {
		for _, vertical := range []bool{false, true} {
			checkResizeDispatch(t, ResizerConfig{Input: tt.input, Output: tt.output, Pack: tt.pack, Depth: tt.depth, Vertical: vertical, Threads: 1}, tt.filter)
		}
	}
}

func TestSIMD128ScalersLarge(t *testing.T) {
	if testing.Short() {
		t.Skip()
	}
	for _, vertical := range []bool{false, true} {
		for _, wide := range []bool{false, true} {
			for _, taps := range []int{6, 8, 12, 14, 18} {
				c := scalerCase{
					vertical: vertical,
					wide:     wide,
					taps:     taps,
					width:    1920,
					height:   4,
					dp:       1920,
					seed:     int64(taps),
				}
				c.width /= c.bytes()
				if vertical {
					c.input = taps + 16
					c.sp = 1920
				} else {
					c.input = 4000 / c.bytes()
					c.sp = 4000
				}
				checkSIMD128Case(t, c)
			}
		}
	}
}

func BenchmarkSimd128H8Go(b *testing.B)      { benchVariant(b, false, false, 8, goVariant) }
func BenchmarkSimd128H8Simd(b *testing.B)    { benchVariant(b, false, false, 8, simd128Dispatch) }
func BenchmarkSimd128H12Go(b *testing.B)     { benchVariant(b, false, false, 12, goVariant) }
func BenchmarkSimd128H12Simd(b *testing.B)   { benchVariant(b, false, false, 12, simd128Dispatch) }
func BenchmarkSimd128H32Go(b *testing.B)     { benchVariant(b, false, false, 32, goVariant) }
func BenchmarkSimd128H32Simd(b *testing.B)   { benchVariant(b, false, false, 32, simd128Dispatch) }
func BenchmarkSimd128V6Go(b *testing.B)      { benchVariant(b, true, false, 6, goVariant) }
func BenchmarkSimd128V6Simd(b *testing.B)    { benchVariant(b, true, false, 6, simd128Dispatch) }
func BenchmarkSimd128V14Go(b *testing.B)     { benchVariant(b, true, false, 14, goVariant) }
func BenchmarkSimd128V14Simd(b *testing.B)   { benchVariant(b, true, false, 14, simd128Dispatch) }
func BenchmarkSimd128H8W16Go(b *testing.B)   { benchVariant(b, false, true, 8, goVariant) }
func BenchmarkSimd128H8W16Simd(b *testing.B) { benchVariant(b, false, true, 8, simd128Dispatch) }
func BenchmarkSimd128V6W16Go(b *testing.B)   { benchVariant(b, true, true, 6, goVariant) }
func BenchmarkSimd128V6W16Simd(b *testing.B) { benchVariant(b, true, true, 6, simd128Dispatch) }

func FuzzSIMD128Scalers(f *testing.F) {
	fuzzScalers(f, simd128Variants)
}
