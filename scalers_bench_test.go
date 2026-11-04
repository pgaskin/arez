// Copyright 2026 Patrick Gaskin, 2013 Benoît Amiaux. All rights reserved. Use
// of this source code is governed by a MIT-style license that can be found in
// the LICENSE file.

package arez

import (
	"testing"
)

func benchScaler(b *testing.B, simd, vertical bool, taps int) {
	n := 96
	src := make([]byte, n*n)
	dst := make([]byte, n*n*2)
	cfg := ResizerConfig{
		Input:       n,
		Output:      n * 2,
		Vertical:    vertical,
		Interlaced:  false,
		Threads:     1,
		DisableSIMD: !simd,
	}
	dp := n
	if !vertical {
		dp *= 2
	}
	resizer, err := NewResizer(&cfg, Lanczos(taps>>1))
	expect(b, err, nil)
	for b.Loop() {
		resizer.Resize(dst, src, n, n, dp, n)
	}
}

// synthetic benchmarks
func BenchmarkVerticalScaler2Go(b *testing.B)      { benchScaler(b, false, true, 2) }
func BenchmarkVerticalScaler2Simd(b *testing.B)    { benchScaler(b, true, true, 2) }
func BenchmarkVerticalScaler4Go(b *testing.B)      { benchScaler(b, false, true, 4) }
func BenchmarkVerticalScaler4Simd(b *testing.B)    { benchScaler(b, true, true, 4) }
func BenchmarkVerticalScaler6Go(b *testing.B)      { benchScaler(b, false, true, 6) }
func BenchmarkVerticalScaler6Simd(b *testing.B)    { benchScaler(b, true, true, 6) }
func BenchmarkVerticalScaler8Go(b *testing.B)      { benchScaler(b, false, true, 8) }
func BenchmarkVerticalScaler8Simd(b *testing.B)    { benchScaler(b, true, true, 8) }
func BenchmarkVerticalScaler10Go(b *testing.B)     { benchScaler(b, false, true, 10) }
func BenchmarkVerticalScaler10Simd(b *testing.B)   { benchScaler(b, true, true, 10) }
func BenchmarkVerticalScaler12Go(b *testing.B)     { benchScaler(b, false, true, 12) }
func BenchmarkVerticalScaler12Simd(b *testing.B)   { benchScaler(b, true, true, 12) }
func BenchmarkVerticalScalerNGo(b *testing.B)      { benchScaler(b, false, true, 14) }
func BenchmarkVerticalScalerNSimd(b *testing.B)    { benchScaler(b, true, true, 14) }
func BenchmarkHorizontalScaler2Go(b *testing.B)    { benchScaler(b, false, false, 2) }
func BenchmarkHorizontalScaler2Simd(b *testing.B)  { benchScaler(b, true, false, 2) }
func BenchmarkHorizontalScaler4Go(b *testing.B)    { benchScaler(b, false, false, 4) }
func BenchmarkHorizontalScaler4Simd(b *testing.B)  { benchScaler(b, true, false, 4) }
func BenchmarkHorizontalScaler6Go(b *testing.B)    { benchScaler(b, false, false, 6) }
func BenchmarkHorizontalScaler6Simd(b *testing.B)  { benchScaler(b, true, false, 6) }
func BenchmarkHorizontalScaler8Go(b *testing.B)    { benchScaler(b, false, false, 8) }
func BenchmarkHorizontalScaler8Simd(b *testing.B)  { benchScaler(b, true, false, 8) }
func BenchmarkHorizontalScaler10Go(b *testing.B)   { benchScaler(b, false, false, 10) }
func BenchmarkHorizontalScaler10Simd(b *testing.B) { benchScaler(b, true, false, 10) }
func BenchmarkHorizontalScaler12Go(b *testing.B)   { benchScaler(b, false, false, 12) }
func BenchmarkHorizontalScaler12Simd(b *testing.B) { benchScaler(b, true, false, 12) }
func BenchmarkHorizontalScalerNGo(b *testing.B)    { benchScaler(b, false, false, 14) }
func BenchmarkHorizontalScalerNSimd(b *testing.B)  { benchScaler(b, true, false, 14) }
