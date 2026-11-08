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
	case *image.NRGBA, *image.NRGBA64, *image.NYCbCrA:
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

// The NYCbCrA passes premultiply the luma by the alpha, and the chroma by the
// mean alpha of the luma pixels of each chroma sample, resized as another plane
// at the chroma resolution (chromaAlpha) to divide the resized chroma by. The
// chroma is weighted without its 128 offset, which cancels out in the division.

// chromaFootprint maps the chroma samples of a YCbCr image to its luma pixels
// like image.YCbCr, by the fields of interlaced images.
type chromaFootprint struct {
	rows  [][]int // luma rows of each chroma row
	cols  []int   // chroma column of each luma column
	ncols []int   // number of luma columns of each chroma column
}

func makeChromaFootprint(d *Descriptor) chromaFootprint {
	sx, sy := subsampling(d.Ratio)
	cw, ch := d.Width(1), d.Height(1)
	f := chromaFootprint{
		rows:  make([][]int, ch),
		cols:  make([]int, d.Width(0)),
		ncols: make([]int, cw),
	}
	for x := range f.cols {
		c := clip((d.Rect.Min.X+x)/sx-d.Rect.Min.X/sx, 0, cw-1)
		f.cols[x] = c
		f.ncols[c]++
	}
	for y := range d.Height(0) {
		c := (d.Rect.Min.Y+y)/sy - d.Rect.Min.Y/sy
		if d.Interlaced && sy == 2 {
			c = y>>2<<1 | y&1 // in the same field
		}
		c = clip(c, 0, ch-1)
		f.rows[c] = append(f.rows[c], y)
	}
	return f
}

// premultiplyYCbCrA writes the 8-bit straight alpha planes of src as 16-bit
// premultiplied planes, and the chromaAlpha plane, in pre, with threads
// goroutines.
func premultiplyYCbCrA(threads int, pre, src []plane, f *chromaFootprint) {
	luma, alpha := &src[0], &src[alphaPlane]
	forRows(threads, luma.Height, func(y0, y1 int) {
		for y := y0; y < y1; y++ {
			a := alpha.Data[y*alpha.Pitch:][:alpha.Width]
			dy := pre[0].Data[y*pre[0].Pitch:][:2*luma.Width]
			da := pre[alphaPlane].Data[y*pre[alphaPlane].Pitch:][:2*luma.Width]
			for x, v := range luma.Data[y*luma.Pitch:][:luma.Width] {
				binary.BigEndian.PutUint16(dy[2*x:], uint16(scale8(uint32(v), uint32(a[x]))))
				binary.BigEndian.PutUint16(da[2*x:], uint16(a[x])*257)
			}
		}
	})
	cb, cr := &src[1], &src[2]
	forRows(threads, cb.Height, func(y0, y1 int) {
		sums := make([]uint32, cb.Width)
		for y := y0; y < y1; y++ {
			clear(sums)
			for _, ly := range f.rows[y] {
				for x, v := range alpha.Data[ly*alpha.Pitch:][:alpha.Width] {
					sums[f.cols[x]] += uint32(v)
				}
			}
			b := cb.Data[y*cb.Pitch:][:cb.Width]
			r := cr.Data[y*cr.Pitch:][:cr.Width]
			db := pre[1].Data[y*pre[1].Pitch:][:2*cb.Width]
			dr := pre[2].Data[y*pre[2].Pitch:][:2*cb.Width]
			da := pre[chromaAlpha].Data[y*pre[chromaAlpha].Pitch:][:2*cb.Width]
			for x, sum := range sums {
				var a uint32 // mean 16-bit alpha
				if n := uint32(len(f.rows[y]) * f.ncols[x]); n != 0 {
					a = (sum*257 + n/2) / n
				}
				binary.BigEndian.PutUint16(db[2*x:], uint16((uint32(b[x])*a+127)/255))
				binary.BigEndian.PutUint16(dr[2*x:], uint16((uint32(r[x])*a+127)/255))
				binary.BigEndian.PutUint16(da[2*x:], uint16(a))
			}
		}
	})
}

// unpremultiplyYCbCrA writes the 16-bit premultiplied planes of out as 8-bit
// straight alpha planes in dst, with threads goroutines. The colour of pixels
// with an alpha of zero is black.
func unpremultiplyYCbCrA(threads int, dst, out []plane) {
	luma, alpha := &dst[0], &dst[alphaPlane]
	forRows(threads, luma.Height, func(y0, y1 int) {
		for y := y0; y < y1; y++ {
			s := out[0].Data[y*out[0].Pitch:][:2*luma.Width]
			sa := out[alphaPlane].Data[y*out[alphaPlane].Pitch:][:2*luma.Width]
			d := luma.Data[y*luma.Pitch:][:luma.Width]
			da := alpha.Data[y*alpha.Pitch:][:alpha.Width]
			for x := range d {
				a := uint32(binary.BigEndian.Uint16(sa[2*x:]))
				da[x] = byte((a + 128) / 257)
				d[x] = byte(unscaleSample8(uint32(binary.BigEndian.Uint16(s[2*x:])), a, 0))
			}
		}
	})
	cb, cr := &dst[1], &dst[2]
	forRows(threads, cb.Height, func(y0, y1 int) {
		for y := y0; y < y1; y++ {
			sb := out[1].Data[y*out[1].Pitch:][:2*cb.Width]
			sr := out[2].Data[y*out[2].Pitch:][:2*cb.Width]
			sa := out[chromaAlpha].Data[y*out[chromaAlpha].Pitch:][:2*cb.Width]
			b := cb.Data[y*cb.Pitch:][:cb.Width]
			r := cr.Data[y*cr.Pitch:][:cr.Width]
			for x := range b {
				a := uint32(binary.BigEndian.Uint16(sa[2*x:]))
				b[x] = byte(unscaleSample8(uint32(binary.BigEndian.Uint16(sb[2*x:])), a, 128))
				r[x] = byte(unscaleSample8(uint32(binary.BigEndian.Uint16(sr[2*x:])), a, 128))
			}
		}
	})
}

// unscaleSample8 divides the 16-bit premultiplied sample c by the 16-bit alpha
// a into an 8-bit sample, or zero if the alpha rounds to zero at 8 bits.
func unscaleSample8(c, a, zero uint32) uint32 {
	switch {
	case (a+128)/257 == 0:
		return zero
	case a == 65535:
		return narrow16(c)
	}
	return unscale8(c, a, 1/float64(a))
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
