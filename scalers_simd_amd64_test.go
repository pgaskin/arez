// Copyright 2026 Patrick Gaskin. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

//go:build amd64 && goexperiment.simd

package arez

import (
	"fmt"
	"math/rand"
	"os"
	"reflect"
	"strconv"
	"testing"
)

// REZ_SIMD=avx512 disables the AVX-512 VBMI kernels, REZ_SIMD=avx2 restricts
// the archsimd dispatch to AVX2 kernels and REZ_SIMD=none falls back to the Go
// scalers, as on CPUs without AVX2. Use -count=1, the test cache does not see
// these.
func init() {
	if n, err := strconv.Atoi(os.Getenv("REZ_STRIP")); err == nil {
		simdStripBytes = n
	}
	switch os.Getenv("REZ_SIMD") {
	case "avx512":
		useVBMI = false
	case "avx2":
		useAVX512, useVBMI = false, false
	case "none":
		useSIMD, useAVX512, useVBMI = false, false, false
	}
}

// simdVariants returns every archsimd scaler the host can run for k, the first
// one being the one dispatch would pick. The variants of 16-bit kernels are
// named with a w suffix.
func simdVariants(cfg *ResizerConfig, k kernel) []scalerVariant {
	ks := prepareKernelSIMD(cfg, k)
	pack := cfg.Pack
	suffix := ""
	hN, hShuf, hPermB := h8scaleNSIMD, h8scaleShufSIMD, h8scalePermBSIMD
	v256, v512 := v8scaleNSIMD, v8scaleNSIMD512
	if k.wide {
		suffix = "w"
		hN, hShuf, hPermB = h16scaleNSIMD, h16scaleShufSIMD, h16scalePermBSIMD
		v256, v512 = v16scaleNSIMD, v16scaleNSIMD512
		if !cfg.Vertical {
			k = widenKernel(k)
			pack *= 2
		}
	}
	var v []scalerVariant
	if cfg.Vertical {
		v = append(v, scalerVariant{"dispatch" + suffix, getVerticalScalerSIMD(&ks, true), ks})
		k256 := k
		k256.coeffs, k256.cofscale = prepareVerticalCoeffsSIMD(k.coeffs, len(k.offsets), k.size, 16)
		v = append(v, scalerVariant{"v256" + suffix, v256, k256})
		if useAVX512 {
			k512 := k
			k512.coeffs, k512.cofscale = prepareVerticalCoeffsSIMD(k.coeffs, len(k.offsets), k.size, 32)
			v = append(v, scalerVariant{"v512" + suffix, v512, k512})
		}
		return v
	}
	v = append(v, scalerVariant{"dispatch" + suffix, getHorizontalScalerSIMD(&ks, true), ks})
	v = append(v, scalerVariant{"hN" + suffix, hN, padKernel(k, 16, simdPixels)})
	for _, share := range []int{1, 2, 4} {
		if cof, off := prepareHorizontalCoeffsSIMDShuf(k.coeffs, k.offsets, k.size, pack, share, 4); cof != nil {
			ks := k
			ks.coeffs, ks.offsets = cof, off
			v = append(v, scalerVariant{fmt.Sprintf("hShuf%v%v", share, suffix), hShuf, ks})
		}
	}
	if cof, off := prepareHorizontalCoeffsSIMDShuf(k.coeffs, k.offsets, k.size, pack, 0, 2); cof != nil {
		ks := k
		ks.coeffs, ks.offsets = cof, off
		v = append(v, scalerVariant{"hShuf16" + suffix, hShuf, ks})
	}
	if useVBMI {
		for _, outs := range []int{16, 8, 4} {
			if cof, off := prepareHorizontalCoeffsSIMDPermB(k.coeffs, k.offsets, k.size, pack, outs); cof != nil {
				kp := k
				kp.coeffs, kp.offsets = cof, off
				v = append(v, scalerVariant{fmt.Sprintf("hPermB%v%v", outs, suffix), hPermB, kp})
			}
		}
	}
	return v
}

// simdDispatch returns the archsimd scaler dispatch would pick for k.
func simdDispatch(cfg *ResizerConfig, k kernel) scalerVariant {
	return simdVariants(cfg, k)[0]
}

// checkScalerCase compares every archsimd variant with the Go scaler and counts
// the variants that ran in ran.
func checkScalerCase(t *testing.T, c scalerCase, ran map[string]int) {
	t.Helper()
	for _, v := range checkScalerVariants(t, c, simdVariants) {
		if ran != nil {
			ran[v.name]++
		}
	}
}

func TestSIMDEnabled(t *testing.T) {
	if !useSIMD {
		t.Skip("no avx2")
	}
	vbmi := func(a, b int) int {
		if useVBMI {
			return a
		}
		return b
	}
	avx512 := func(a, b int) int {
		if useAVX512 {
			return a
		}
		return b
	}
	tests := []struct {
		cfg    ResizerConfig
		filter Filter
		taps   int
		layout int
	}{
		{ResizerConfig{Input: 100, Output: 200}, Lanczos(3), 6, vbmi(layoutSIMDPermB, layoutSIMDShuf)},
		{ResizerConfig{Input: 1920, Output: 640}, Lanczos(3), 18, vbmi(layoutSIMDPermB, layoutSIMDShuf)},
		{ResizerConfig{Input: 1920, Output: 400}, Lanczos(3), 30, vbmi(layoutSIMDPermB, layoutSIMDShuf)},
		{ResizerConfig{Input: 1920, Output: 320}, Lanczos(3), 36, vbmi(layoutSIMDPermB, layoutSIMDShuf)},
		{ResizerConfig{Input: 3840, Output: 320}, Lanczos(3), 72, vbmi(layoutSIMDPermB, layoutSIMDN)},
		{ResizerConfig{Input: 3840, Output: 320}, Bicubic(), 48, vbmi(layoutSIMDPermB, layoutSIMDN)},
		{ResizerConfig{Input: 3840, Output: 320}, Bilinear(), 24, vbmi(layoutSIMDPermB, layoutSIMDN)},
		{ResizerConfig{Input: 1920, Output: 100}, Lanczos(3), 116, layoutSIMDN},
		{ResizerConfig{Input: 640, Output: 1920}, Bilinear(), 2, vbmi(layoutSIMDPermB, layoutSIMDShuf)},
		{ResizerConfig{Input: 720, Output: 640, Pack: 4}, Bicubic(), 24, vbmi(layoutSIMDPermB, layoutSIMDShuf)},
		{ResizerConfig{Input: 1920, Output: 640, Pack: 4}, Bicubic(), 48, vbmi(layoutSIMDPermB, layoutSIMDShuf)},
		{ResizerConfig{Input: 3840, Output: 320, Pack: 4}, Lanczos(3), 288, vbmi(layoutSIMDPermB, layoutSIMDShuf)},
		{ResizerConfig{Input: 1920, Output: 160, Pack: 4}, Lanczos(3), 288, vbmi(layoutSIMDPermB, layoutSIMDShuf)},
		{ResizerConfig{Input: 100, Output: 200, Vertical: true}, Lanczos(3), 6, avx512(layoutSIMDV512, 0)},
		// 16-bit kernels are widened to bytes, twice the taps
		{ResizerConfig{Input: 100, Output: 200, Depth: 16}, Lanczos(3), 12, vbmi(layoutSIMDPermB, layoutSIMDShuf)},
		{ResizerConfig{Input: 1920, Output: 640, Depth: 16}, Lanczos(3), 36, vbmi(layoutSIMDPermB, layoutSIMDShuf)},
		{ResizerConfig{Input: 3840, Output: 320, Depth: 16}, Lanczos(3), 144, vbmi(layoutSIMDPermB, layoutSIMDN)},
		{ResizerConfig{Input: 720, Output: 640, Pack: 4, Depth: 16}, Bicubic(), 48, vbmi(layoutSIMDPermB, layoutSIMDShuf)},
		{ResizerConfig{Input: 1920, Output: 160, Pack: 4, Depth: 16}, Lanczos(3), 576, vbmi(layoutSIMDPermB, layoutSIMDShuf)},
		{ResizerConfig{Input: 100, Output: 200, Vertical: true, Depth: 16}, Lanczos(3), 6, avx512(layoutSIMDV512, 0)},
	}
	for _, tt := range tests {
		cfg := tt.cfg
		cfg.Pack = max(cfg.Pack, 1)
		k, err := makeKernel(&cfg, tt.filter, 0)
		expect(t, err, nil)
		if k.size != tt.taps || k.layout != tt.layout {
			t.Errorf("%+v %v: got %v taps layout %v, want %v taps layout %v", cfg, tt.filter.Name(), k.size, k.layout, tt.taps, tt.layout)
		}
		var got, want uintptr
		if cfg.Vertical {
			got = reflect.ValueOf(getVerticalScaler(&k, true)).Pointer()
			want = reflect.ValueOf(getVerticalScalerSIMD(&k, true)).Pointer()
		} else {
			got = reflect.ValueOf(getHorizontalScaler(&k, true)).Pointer()
			want = reflect.ValueOf(getHorizontalScalerSIMD(&k, true)).Pointer()
		}
		if got != want {
			t.Errorf("%+v: dispatch does not use the archsimd scaler", cfg)
		}
	}
}

func TestSIMDScalers(t *testing.T) {
	if !useSIMD {
		t.Skip("no avx2")
	}
	taps := []int{2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 30, 32, 34, 48, 64, 66}
	widths := []int{1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 100, 127, 128, 129}
	rng := rand.New(rand.NewSource(1))
	ran := map[string]int{}
	strip := simdStripBytes
	defer func() { simdStripBytes = strip }()
	for _, vertical := range []bool{false, true} {
		for _, taps := range taps {
			for _, width := range widths {
				for i := range 4 {
					c := scalerCase{
						vertical: vertical,
						wide:     i == 3,
						taps:     taps,
						width:    width,
						height:   1 + rng.Intn(5),
						pack:     1,
						slack:    rng.Intn(3) * 20,
						seed:     rng.Int63(),
					}
					if vertical {
						c.input = taps + c.height*4
						c.sp = width*c.bytes() + rng.Intn(3)*5
					} else {
						if taps <= 48 && rng.Intn(2) == 0 {
							c.pack = 2 + rng.Intn(3)
						}
						c.ratio = []int{1, 2, 3, 4, 6, 8, 12, 16, 20}[rng.Intn(9)]
						if i == 2 {
							c.sparse = 2 + rng.Intn(3)
						}
						c.input = taps + width*c.ratio/2 + rng.Intn(4)
						c.sp = c.input*c.bytes() + rng.Intn(3)*5
					}
					c.dp = width*c.bytes() + rng.Intn(3)*7
					// small strips split rows in many strips, see hscale
					simdStripBytes = []int{1, 3000, strip}[rng.Intn(3)]
					checkScalerCase(t, c, ran)
				}
			}
		}
	}
	t.Logf("cases per variant: %v", ran)
	want := []string{"hN", "hShuf1", "hShuf2", "hShuf4", "hShuf16", "v256"}
	if useAVX512 {
		want = append(want, "v512")
	}
	if useVBMI {
		want = append(want, "hPermB16", "hPermB8", "hPermB4")
	}
	for _, name := range want {
		for _, suffix := range []string{"", "w"} {
			if ran[name+suffix] == 0 {
				t.Errorf("variant %v not covered", name+suffix)
			}
		}
	}
}

// TestSIMDDispatchLarge runs thumbnail sized reductions through NewResizer,
// comparing the dispatched kernels with the Go scalers.
func TestSIMDDispatchLarge(t *testing.T) {
	if !useSIMD {
		t.Skip("no avx2")
	}
	tests := []struct {
		input, output, pack, depth int
		filter                     Filter
	}{
		{3840, 320, 1, 8, Lanczos(3)},
		{3840, 320, 1, 8, Bicubic()},
		{3840, 320, 1, 8, Bilinear()},
		{1920, 160, 1, 8, Lanczos(3)},
		{1920, 100, 1, 8, Lanczos(3)},
		{3840, 320, 4, 8, Lanczos(3)},
		{1920, 160, 4, 8, Lanczos(3)},
		{1920, 160, 2, 8, Bicubic()},
		{1920, 160, 3, 8, Bicubic()},
		{3840, 320, 1, 16, Lanczos(3)},
		{1920, 160, 1, 16, Bicubic()},
		{1920, 100, 1, 16, Lanczos(3)},
		{3840, 320, 4, 16, Lanczos(3)},
		{1920, 160, 4, 16, Bicubic()},
	}
	for _, tt := range tests {
		for _, vertical := range []bool{false, true} {
			checkResizeDispatch(t, ResizerConfig{Input: tt.input, Output: tt.output, Pack: tt.pack, Depth: tt.depth, Vertical: vertical, Threads: 1}, tt.filter)
		}
	}
}

func TestSIMDScalersLarge(t *testing.T) {
	if !useSIMD || testing.Short() {
		t.Skip()
	}
	for _, vertical := range []bool{false, true} {
		for _, wide := range []bool{false, true} {
			for _, taps := range []int{6, 8, 14, 18} {
				c := scalerCase{
					vertical: vertical,
					wide:     wide,
					taps:     taps,
					width:    1920,
					height:   8,
					dp:       1920,
					seed:     int64(taps),
				}
				c.width /= c.bytes()
				if vertical {
					c.input = taps + 32
					c.sp = 1920
				} else {
					c.input = 4000 / c.bytes()
					c.sp = 4000
				}
				checkScalerCase(t, c, nil)
			}
		}
	}
}

func BenchmarkSimdKernels(b *testing.B) {
	type benchCase struct {
		taps  int
		scale float64
		pack  int
		wide  bool
	}
	width := 1920
	if w, err := strconv.Atoi(os.Getenv("REZ_BENCH_WIDTH")); err == nil {
		width = w
	}
	hcases := []benchCase{
		{2, 0.5, 1, false}, {4, 0.5, 1, false}, {8, 0.5, 1, false}, {10, 0.5, 1, false}, {12, 0.5, 1, false}, {14, 0.5, 1, false},
		{8, 3, 1, false}, {12, 3, 1, false}, {18, 3, 1, false}, {32, 5, 1, false},
		{36, 6, 1, false}, {48, 8, 1, false}, {72, 12, 1, false}, {96, 16, 1, false}, {24, 12, 1, false}, {16, 8, 1, false},
		{12, 6, 1, false}, {24, 6, 1, false}, {30, 5, 1, false}, {36, 9, 1, false}, {48, 12, 1, false}, {60, 10, 1, false},
		{2, 0.5, 4, false}, {4, 0.5, 4, false}, {8, 0.5, 4, false}, {12, 3, 4, false}, {4, 0.5, 3, false},
		{36, 6, 4, false}, {72, 12, 4, false}, {24, 12, 4, false},
		{2, 0.5, 1, true}, {6, 0.5, 1, true}, {12, 2, 1, true}, {18, 3, 1, true}, {36, 6, 1, true},
		{6, 0.5, 4, true}, {24, 4, 4, true},
	}
	for _, hc := range hcases {
		c := scalerCase{
			wide:   hc.wide,
			taps:   hc.taps,
			height: 16 * 1920 / width,
			pack:   hc.pack,
			scale:  hc.scale,
			seed:   1,
		}
		c.width = width / c.bytes()
		c.input = int(float64(c.width)*hc.scale) + hc.taps
		c.dp = width
		c.sp = c.input * c.bytes()
		name := fmt.Sprintf("h%vs%vp%v", hc.taps, hc.scale, hc.pack)
		if hc.wide {
			name += "w"
		}
		b.Run(name, func(b *testing.B) {
			benchScalerVariants(b, c)
		})
	}
	for _, taps := range []int{2, 4, 6, 8, 10, 12, 14} {
		for _, wide := range []bool{false, true} {
			c := scalerCase{
				vertical: true,
				wide:     wide,
				taps:     taps,
				width:    1920,
				height:   16,
				input:    taps + 16*2,
				dp:       1920,
				sp:       1920,
				seed:     1,
			}
			c.width /= c.bytes()
			name := fmt.Sprintf("v%v", taps)
			if wide {
				name += "w"
			}
			b.Run(name, func(b *testing.B) {
				benchScalerVariants(b, c)
			})
		}
	}
	// a whole 1080p plane, downscaled to 480, 180 and 90 rows
	for _, vc := range []struct{ taps, height int }{{6, 480}, {10, 480}, {14, 480}, {36, 180}, {72, 90}} {
		c := scalerCase{
			vertical: true,
			taps:     vc.taps,
			width:    width,
			height:   vc.height,
			input:    1080,
			scale:    1080.0 / float64(vc.height),
			dp:       width,
			sp:       width,
			seed:     1,
		}
		b.Run(fmt.Sprintf("v%vs%v", vc.taps, c.scale), func(b *testing.B) {
			benchScalerVariants(b, c)
		})
	}
}

func benchScalerVariants(b *testing.B, c scalerCase) {
	src, cof, off := makeScalerCase(c)
	cfg := ResizerConfig{Vertical: c.vertical, Input: c.input, Pack: max(c.pack, 1), Depth: c.depth()}
	c = c.unpacked()
	k := kernel{coeffs: cof, offsets: off, size: c.taps, cofscale: 1, wide: c.wide}
	variants := append([]scalerVariant{goVariant(&cfg, k)}, simdVariants(&cfg, k)...)
	for _, v := range variants {
		b.Run(v.name, func(b *testing.B) {
			dst := make([]byte, c.dp*(c.height-1)+c.width)
			b.SetBytes(int64(c.width * c.height))
			for b.Loop() {
				v.scaler(dst, src, v.k.coeffs, v.k.offsets, v.k.size, c.width, c.height, c.dp, c.sp)
			}
		})
	}
}

func BenchmarkSimdH8Go(b *testing.B)      { benchVariant(b, false, false, 8, goVariant) }
func BenchmarkSimdH8Simd(b *testing.B)    { benchVariant(b, false, false, 8, simdDispatch) }
func BenchmarkSimdH18Go(b *testing.B)     { benchVariant(b, false, false, 18, goVariant) }
func BenchmarkSimdH18Simd(b *testing.B)   { benchVariant(b, false, false, 18, simdDispatch) }
func BenchmarkSimdV6Go(b *testing.B)      { benchVariant(b, true, false, 6, goVariant) }
func BenchmarkSimdV6Simd(b *testing.B)    { benchVariant(b, true, false, 6, simdDispatch) }
func BenchmarkSimdV14Go(b *testing.B)     { benchVariant(b, true, false, 14, goVariant) }
func BenchmarkSimdV14Simd(b *testing.B)   { benchVariant(b, true, false, 14, simdDispatch) }
func BenchmarkSimdH32Go(b *testing.B)     { benchVariant(b, false, false, 32, goVariant) }
func BenchmarkSimdH32Simd(b *testing.B)   { benchVariant(b, false, false, 32, simdDispatch) }
func BenchmarkSimdH8W16Go(b *testing.B)   { benchVariant(b, false, true, 8, goVariant) }
func BenchmarkSimdH8W16Simd(b *testing.B) { benchVariant(b, false, true, 8, simdDispatch) }
func BenchmarkSimdV6W16Go(b *testing.B)   { benchVariant(b, true, true, 6, goVariant) }
func BenchmarkSimdV6W16Simd(b *testing.B) { benchVariant(b, true, true, 6, simdDispatch) }

func FuzzSIMDScalers(f *testing.F) {
	if !useSIMD {
		f.Skip("no avx2")
	}
	fuzzScalers(f, simdVariants)
}
