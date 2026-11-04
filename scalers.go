// Copyright 2026 Patrick Gaskin, 2013 Benoît Amiaux. All rights reserved. Use
// of this source code is governed by a MIT-style license that can be found in
// the LICENSE file.

//go:generate go run scalers_fallback_generate.go

package arez

import "encoding/binary"

// bits is the fractional bits of the fixed point kernel coefficients.
const bits = 14

var (
	// hasSIMD reports whether accelerated scalers are available.
	hasSIMD func() bool
	// prepareKernel converts a kernel from the natural layout used by the Go
	// scalers to the layout of the accelerated scalers.
	prepareKernel func(cfg *ResizerConfig, k kernel) kernel
	// getHorizontalScaler returns the horizontal scaler for a kernel.
	getHorizontalScaler func(k *kernel, simd bool) scaler
	// getVerticalScaler returns the vertical scaler for a kernel.
	getVerticalScaler func(k *kernel, simd bool) scaler
)

func init() {
	if hasSIMD == nil || prepareKernel == nil || getHorizontalScaler == nil || getVerticalScaler == nil {
		panic("arez: scalers not set")
	}
}

// setScalers must be called in a package-level variable initializer to run
// before init.
func setScalers(simd func() bool, prepare func(cfg *ResizerConfig, k kernel) kernel, horizontal, vertical func(k *kernel, simd bool) scaler) bool {
	if hasSIMD != nil {
		panic("arez: scalers already set")
	}
	hasSIMD, prepareKernel, getHorizontalScaler, getVerticalScaler = simd, prepare, horizontal, vertical
	return true
}

func getHorizontalScalerGo(taps int, wide bool) scaler {
	if wide {
		return getHorizontalScaler16Go(taps) // 16-bit samples
	}
	switch taps {
	case 2:
		return h8scale2Go
	case 4:
		return h8scale4Go
	case 6:
		return h8scale6Go
	case 8:
		return h8scale8Go
	case 10:
		return h8scale10Go
	case 12:
		return h8scale12Go
	}
	return h8scaleNGo
}

func getVerticalScalerGo(taps int, wide bool) scaler {
	if wide {
		return getVerticalScaler16Go(taps) // 16-bit samples
	}
	switch taps {
	case 2:
		return v8scale2Go
	case 4:
		return v8scale4Go
	case 6:
		return v8scale6Go
	case 8:
		return v8scale8Go
	case 10:
		return v8scale10Go
	case 12:
		return v8scale12Go
	}
	return v8scaleNGo
}

func getHorizontalScaler16Go(taps int) scaler {
	switch taps {
	case 2:
		return h16scale2Go
	case 4:
		return h16scale4Go
	case 6:
		return h16scale6Go
	case 8:
		return h16scale8Go
	case 10:
		return h16scale10Go
	case 12:
		return h16scale12Go
	}
	return h16scaleNGo
}

func getVerticalScaler16Go(taps int) scaler {
	switch taps {
	case 2:
		return v16scale2Go
	case 4:
		return v16scale4Go
	case 6:
		return v16scale6Go
	case 8:
		return v16scale8Go
	case 10:
		return v16scale10Go
	case 12:
		return v16scale12Go
	}
	return v16scaleNGo
}

func u8(x int) byte {
	return byte(min(max(x, 0), 0xFF))
}

func u16(x int) uint16 {
	return uint16(min(max(x, 0), 0xFFFF))
}

func copyPlane(dst, src []byte, width, height, dp, sp int) {
	di := 0
	si := 0
	for range height {
		copy(dst[di:di+width], src[si:si+width])
		di += dp
		si += sp
	}
}

func h8scaleNGo(dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	di := 0
	si := 0
	for range height {
		c := cof
		s := src[si:]
		d := dst[di:]
		for x, xoff := range off[:width] {
			pix := 0
			for i, v := range s[int(xoff) : int(xoff)+taps] {
				pix += int(v) * int(c[i])
			}
			d[x] = u8((pix + 1<<(bits-1)) >> bits)
			c = c[taps:]
		}
		di += dp
		si += sp
	}
}

func v8scaleNGo(dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	di := 0
	for _, yoff := range off[:height] {
		src = src[sp*int(yoff):]
		for x := range dst[di : di+width] {
			pix := 0
			for i, c := range cof[:taps] {
				pix += int(c) * int(src[sp*i+x])
			}
			dst[di+x] = u8((pix + 1<<(bits-1)) >> bits)
		}
		cof = cof[taps:]
		di += dp
	}
}

// The 16-bit Go scalers read and write big endian samples, width is in
// bytes and horizontal offsets are in bytes, with taps coefficients per
// output sample.
func h16scaleNGo(dst, src []byte, cof []int16, off []int32, taps, width, height, dp, sp int) {
	di := 0
	si := 0
	for range height {
		c := cof
		s := src[si:]
		d := dst[di:]
		for x, xoff := range off[:width>>1] {
			pix := 0
			for i, v := range c[:taps] {
				pix += int(v) * int(binary.BigEndian.Uint16(s[int(xoff)+2*i:]))
			}
			binary.BigEndian.PutUint16(d[2*x:], u16((pix+1<<(bits-1))>>bits))
			c = c[taps:]
		}
		di += dp
		si += sp
	}
}

func v16scaleNGo(dst, src []byte, cof []int16, off []int32,
	taps, width, height, dp, sp int) {
	di := 0
	for _, yoff := range off[:height] {
		src = src[sp*int(yoff):]
		d := dst[di : di+width]
		for x := 0; x < width; x += 2 {
			pix := 0
			for i, c := range cof[:taps] {
				pix += int(c) * int(binary.BigEndian.Uint16(src[sp*i+x:]))
			}
			binary.BigEndian.PutUint16(d[x:], u16((pix+1<<(bits-1))>>bits))
		}
		cof = cof[taps:]
		di += dp
	}
}
