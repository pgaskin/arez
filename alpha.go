// Copyright 2026 Patrick Gaskin. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

package arez

import (
	"encoding/binary"
	"image"
)

// note: claude wrote most of these ones lol

// The passes of ConverterConfig.PremultiplyAlpha. Straight alpha pixels are
// premultiplied into a 16-bit plane (regardless of the depth) so precision is
// preserved for nearly transparent 8-bit pixels keep their precision, then
// resized with the 16-bit kernels and divided by their alpha into the
// destination, rounded to 8 bits for 8-bit images.

func isStraight(img image.Image) bool {
	switch img.(type) {
	case *image.NRGBA, *image.NRGBA64:
		return true
	}
	return false
}

// premultiply writes the straight alpha pixels of src, 8 or 16-bit of depth, as
// 16-bit premultiplied pixels in dst, with threads goroutines.
func premultiply(threads int, dst, src *plane, depth int) {
	forRows(threads, src.Height, func(y0, y1 int) {
		for y := y0; y < y1; y++ {
			d := dst.Data[y*dst.Pitch:][:dst.Width*8]
			if depth == 16 {
				premultiplyRow16(d, src.Data[y*src.Pitch:][:src.Width*8])
			} else {
				premultiplyRow8(d, src.Data[y*src.Pitch:][:src.Width*4])
			}
		}
	})
}

// unpremultiply writes the 16-bit premultiplied pixels of src as straight alpha
// pixels of depth bits in dst, with threads goroutines. dst and src may be the
// same plane when depth is 16.
func unpremultiply(threads int, dst, src *plane, depth int) {
	forRows(threads, src.Height, func(y0, y1 int) {
		for y := y0; y < y1; y++ {
			s := src.Data[y*src.Pitch:][:src.Width*8]
			if depth == 16 {
				unpremultiplyRow16(dst.Data[y*dst.Pitch:][:dst.Width*8], s)
			} else {
				unpremultiplyRow8(dst.Data[y*dst.Pitch:][:dst.Width*4], s)
			}
		}
	})
}

// premultiplyRow8 converts 8-bit straight alpha pixels to 16-bit premultiplied
// ones: c*a*257/255 rounded, the 16-bit value of c scaled by the alpha, and
// a*257.
func premultiplyRow8(d, s []byte) {
	for x := 0; x+4 <= len(s); x += 4 {
		p := binary.LittleEndian.Uint32((*[4]byte)(s[x:])[:])
		a := p >> 24
		v := scale8(p&0xff, a)<<48 | scale8(p>>8&0xff, a)<<32 | scale8(p>>16&0xff, a)<<16 | uint64(a*257)
		binary.BigEndian.PutUint64((*[8]byte)(d[2*x:])[:], v)
	}
}

// premultiplyRow16 converts 16-bit straight alpha pixels to premultiplied ones:
// c*a/65535 rounded.
func premultiplyRow16(d, s []byte) {
	for x := 0; x+8 <= len(s); x += 8 {
		p := binary.BigEndian.Uint64((*[8]byte)(s[x:])[:])
		a := p & 0xffff
		v := scale16(p>>48, a)<<48 | scale16(p>>32&0xffff, a)<<32 | scale16(p>>16&0xffff, a)<<16 | a
		binary.BigEndian.PutUint64((*[8]byte)(d[x:])[:], v)
	}
}

// unpremultiplyRow8 converts 16-bit premultiplied pixels to 8-bit straight
// alpha ones: c = p*255/a rounded and clamped, a*255/65535 rounded, the colour
// being zero when the rounded alpha is.
func unpremultiplyRow8(d, s []byte) {
	for x := 0; x+4 <= len(d); x += 4 {
		p := binary.BigEndian.Uint64((*[8]byte)(s[2*x:])[:])
		a := uint32(p & 0xffff)
		a8 := (a + 128) / 257
		v := a8 << 24
		switch {
		case a8 == 0:
		case a == 65535:
			v |= narrow16(uint32(p>>48)) | narrow16(uint32(p>>32&0xffff))<<8 | narrow16(uint32(p>>16&0xffff))<<16
		default:
			inv := 1 / float64(a)
			v |= unscale8(uint32(p>>48), a, inv) | unscale8(uint32(p>>32&0xffff), a, inv)<<8 | unscale8(uint32(p>>16&0xffff), a, inv)<<16
		}
		binary.LittleEndian.PutUint32((*[4]byte)(d[x:])[:], v)
	}
}

// unpremultiplyRow16 converts 16-bit premultiplied pixels to straight alpha
// ones: c = p*65535/a rounded and clamped.
func unpremultiplyRow16(d, s []byte) {
	for x := 0; x+8 <= len(d); x += 8 {
		p := binary.BigEndian.Uint64((*[8]byte)(s[x:])[:])
		a := uint32(p & 0xffff)
		v := uint64(a)
		switch {
		case a == 0:
		case a == 65535:
			v = p
		default:
			inv := 1 / float64(a)
			v |= unscale16(uint32(p>>48), a, inv)<<48 | unscale16(uint32(p>>32&0xffff), a, inv)<<32 | unscale16(uint32(p>>16&0xffff), a, inv)<<16
		}
		binary.BigEndian.PutUint64((*[8]byte)(d[x:])[:], v)
	}
}

// scale8 premultiplies the 8-bit colour c by the 8-bit alpha a into a 16-bit
// sample, c*a*257/255 rounded.
func scale8(c, a uint32) uint64 {
	return uint64((c*a*257 + 127) / 255)
}

// scale16 premultiplies the 16-bit colour c by the 16-bit alpha a, c*a/65535
// rounded.
func scale16(c, a uint64) uint64 {
	return (c*a + 32767) / 65535
}

// narrow16 rounds a 16-bit sample to 8 bits.
func narrow16(c uint32) uint32 {
	return (c*255 + 32767) / 65535
}

// unscale8 divides the 16-bit premultiplied colour c by the 16-bit alpha a,
// given with its reciprocal, into an 8-bit colour, c*255/a rounded and clamped.
func unscale8(c, a uint32, inv float64) uint32 {
	return min(divide(c*255+a/2, inv), 255)
}

// unscale16 divides the 16-bit premultiplied colour c by the 16-bit alpha a,
// given with its reciprocal, c*65535/a rounded and clamped.
func unscale16(c, a uint32, inv float64) uint64 {
	return uint64(min(divide(c*65535+a/2, inv), 65535))
}

// divide returns n/a for a 16-bit a given as inv = 1/float64(a), exactly: n/a
// is an integer or at least 1/a away from one, while the float product is
// within 2^32/a * 2^-52 < 1e-6/a of it, so a bias between the two makes the
// floor exact for every 32-bit n.
func divide(n uint32, inv float64) uint32 {
	return uint32(float64(n)*inv + 1.0/(1<<17))
}

// forRows calls f for consecutive ranges of rows [y0, y1) covering height rows,
// in up to threads goroutines, and waits for them.
func forRows(threads, height int, f func(y0, y1 int)) {
	threads = clip(threads, 1, height)
	n := (height + threads - 1) / threads
	var group jobs
	for y := 0; y < height; y += n {
		y0, y1 := y, min(y+n, height)
		dispatch(&group, threads, func() { f(y0, y1) })
	}
	group.Wait()
}
