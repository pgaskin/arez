// Copyright 2026 Patrick Gaskin, 2013 Benoît Amiaux. All rights reserved. Use
// of this source code is governed by a MIT-style license that can be found in
// the LICENSE file.

package arez

import (
	"errors"
	"fmt"
	"image"
	"reflect"
	"runtime"
	"sync"
)

// Base errors (compare with [errors.Is]).
var (
	// ErrUnsupportedImage is returned for unsupported image types (Paletted,
	// Uniform, other [image.Image] implementations), nil images, YCbCr images
	// of an unknown subsample ratio, images with a negative stride, or ones
	// which otherwise overflow/overlap.
	ErrUnsupportedImage = errors.New("arez: unsupported image")

	// ErrInvalidDescriptor is returned by [Descriptor.Check], by [NewConverter]
	// for an invalid input or output descriptor, and by [NewResizer] for an
	// invalid depth or pack.
	ErrInvalidDescriptor = errors.New("arez: invalid descriptor")

	// ErrIncompatible is returned when the input and output of a conversion
	// differ in interlacing, pack, planes or depth, or are images of different
	// types.
	ErrIncompatible = errors.New("arez: incompatible formats")

	// ErrTooSmall is returned when an input or output image is too small to
	// resize.
	ErrTooSmall = errors.New("arez: size too small")

	// ErrTooLarge is returned when the width/height of an image is too large
	// and would overflow.
	ErrTooLarge = errors.New("arez: size too large")

	// ErrMismatch is returned by [Converter.Convert] for images that do not
	// match the configuration the converter was created for.
	ErrMismatch = errors.New("arez: image does not match the converter")

	// ErrInvalidFilter is returned for a nil or invalid [Filter].
	ErrInvalidFilter = errors.New("arez: invalid filter")
)

// ChromaRatio is the chroma subsampling ratio of a YCbCr image, see
// [ChromaRatioOf].
type ChromaRatio int

// String returns the ratio in J:a:b notation, such as "4:2:0".
func (r ChromaRatio) String() string {
	switch r {
	case Ratio410:
		return "4:1:0"
	case Ratio411:
		return "4:1:1"
	case Ratio420:
		return "4:2:0"
	case Ratio422:
		return "4:2:2"
	case Ratio440:
		return "4:4:0"
	case Ratio444:
		return "4:4:4"
	}
	return fmt.Sprintf("ChromaRatio(%d)", int(r))
}

const (
	Ratio410 ChromaRatio = iota // 4:1:0
	Ratio411                    // 4:1:1
	Ratio420                    // 4:2:0
	Ratio422                    // 4:2:2
	Ratio440                    // 4:4:0
	Ratio444                    // 4:4:4, no subsampling
)

// Descriptor describes the format and size of an image. [Prepare] returns
// the descriptors of two images.
type Descriptor struct {
	Rect       image.Rectangle // bounds of the image, in pixels
	Ratio      ChromaRatio     // chroma subsampling, not used for single plane images
	Pack       int             // samples per pixel, 1 to 4
	Interlaced bool            // whether the rows are two interlaced fields
	Planes     int             // number of planes, 1 to 4
	Depth      int             // bits per sample, 8 (or 0) or 16
}

// String describes the image in a human-readable form.
func (d Descriptor) String() string {
	at := ""
	if d.Rect.Min != (image.Point{}) {
		at = " at " + d.Rect.Min.String()
	}
	planes := "planes"
	if d.Planes == 1 {
		planes = "plane"
	}
	return fmt.Sprintf("%dx%d%s %d %s %v %v %d-bit %v", d.Rect.Dx(), d.Rect.Dy(), at, d.Planes, planes, d.Ratio,
		toPackedString(d.Pack), d.depth(), toInterlacedString(d.Interlaced))
}

func (d Descriptor) Check() error {
	if d.Pack < 1 || d.Pack > 4 {
		return fmt.Errorf("%w: pack %v", ErrInvalidDescriptor, d.Pack)
	}
	if d.Depth != 0 && d.Depth != 8 && d.Depth != 16 {
		return fmt.Errorf("%w: depth %v", ErrInvalidDescriptor, d.Depth)
	}
	if d.Planes < 1 || d.Planes > maxPlanes {
		return fmt.Errorf("%w: planes %v", ErrInvalidDescriptor, d.Planes)
	}
	if d.Ratio < Ratio410 || d.Ratio > Ratio444 {
		return fmt.Errorf("%w: ratio %v", ErrInvalidDescriptor, d.Ratio)
	}
	return nil
}

func (d Descriptor) matches(cfg *Descriptor) bool {
	if d.Pack != cfg.Pack || d.Planes != cfg.Planes || d.depth() != cfg.depth() {
		return false
	}
	if d.Planes > 1 && d.Ratio != cfg.Ratio {
		return false
	}
	for i := range d.Planes {
		if d.Width(i) != cfg.Width(i) || d.Height(i) != cfg.Height(i) {
			return false
		}
	}
	return true
}

func (d Descriptor) checkSize() error {
	if d.Rect.Dx() < 1 || d.Rect.Dy() < 1 {
		return fmt.Errorf("%w: %vx%v image", ErrTooSmall, d.Rect.Dx(), d.Rect.Dy())
	}
	for i := range d.Planes {
		if w, h := d.Width(i), d.Height(i); w < 1 || h < 1 {
			return fmt.Errorf("%w: plane %v of %v is %vx%v", ErrTooSmall, i, d, w, h)
		}
	}
	return nil
}

func (d Descriptor) depth() int {
	if d.Depth == 16 {
		return 16
	}
	return 8
}

func (d Descriptor) Width(plane int) int {
	if plane < 0 || plane+1 > maxPlanes {
		panic(fmt.Sprintf("arez: invalid plane %v", plane))
	}
	r := d.Rect
	if plane == 0 || plane == alphaPlane {
		return r.Dx()
	}
	switch d.Ratio {
	case Ratio410, Ratio411:
		return chromaSize(r.Min.X, r.Max.X, 4)
	case Ratio420, Ratio422:
		return chromaSize(r.Min.X, r.Max.X, 2)
	case Ratio440, Ratio444:
		return r.Dx()
	}
	panic(fmt.Sprintf("arez: invalid ratio %v", d.Ratio))
}

func (d Descriptor) Height(plane int) int {
	if plane < 0 || plane+1 > maxPlanes {
		panic(fmt.Sprintf("arez: invalid plane %v", plane))
	}
	r := d.Rect
	if plane == 0 || plane == alphaPlane {
		return r.Dy()
	}
	switch d.Ratio {
	case Ratio411, Ratio422, Ratio444:
		return r.Dy()
	case Ratio410, Ratio420, Ratio440:
		return chromaSize(r.Min.Y, r.Max.Y, 2)
	}
	panic(fmt.Sprintf("arez: invalid ratio %v", d.Ratio))
}

func chromaSize(lo, hi, n int) int {
	return (hi+n-1)/n - lo/n
}

// ConverterConfig contains configuration for a [Converter].
//
// Additional options may be added in future versions, but the zero value will
// always have the same behaviour within a major version.
type ConverterConfig struct {
	Input            Descriptor // input format and size
	Output           Descriptor // output format and size
	Threads          int        // goroutines per plane, <= 0 for GOMAXPROCS
	DisableSIMD      bool       // only use the non-accelerated Go implementation
	PremultiplyAlpha bool       // premultiply straight images before filtering
}

const (
	maxPlanes  = 4
	alphaPlane = 3             // the alpha plane of NYCbCrA, at the luma resolution
	preScratch = maxPlanes     // the premultiplied input
	outScratch = maxPlanes + 1 // the premultiplied 16-bit output of an 8-bit conversion
)

type plane struct {
	Data   []byte // plane buffer
	Width  int    // width in pixels
	Height int    // height in pixels
	Pitch  int    // pitch in bytes
	Bpp    int    // bytes per pixel
}

// Converter resizes images of one format and size into another, see
// [NewConverter]. It converts the chroma subsampling of YCbCr images. A
// Converter is safe for concurrent use, and may use up to
// [ConverterConfig.Threads] goroutines per plane.
type Converter struct {
	cfg     ConverterConfig
	wrez    [maxPlanes]*Resizer
	hrez    [maxPlanes]*Resizer
	scratch [maxPlanes + 2]plane // Pitch is 0 for unused planes
	size    int                  // bytes of all scratch planes
	pool    sync.Pool            // []byte of size bytes
	premul  bool                 // PremultiplyAlpha
}

func toInterlacedString(interlaced bool) string {
	if interlaced {
		return "interlaced"
	}
	return "progressive"
}

func toPackedString(pack int) string {
	return fmt.Sprintf("%v-packed", pack)
}

func align(value, align int) int {
	return (value + align - 1) & -align
}

func checkConversion(dst, src *Descriptor) error {
	if err := src.Check(); err != nil {
		return fmt.Errorf("%w in the input", err)
	}
	if err := dst.Check(); err != nil {
		return fmt.Errorf("%w in the output", err)
	}
	if src.Interlaced != dst.Interlaced {
		return fmt.Errorf("%w: %v input, %v output", ErrIncompatible,
			toInterlacedString(src.Interlaced),
			toInterlacedString(dst.Interlaced))
	}
	if src.Pack != dst.Pack {
		return fmt.Errorf("%w: %v input, %v output", ErrIncompatible,
			toPackedString(src.Pack),
			toPackedString(dst.Pack))
	}
	if src.Planes != dst.Planes {
		return fmt.Errorf("%w: %v planes input, %v planes output", ErrIncompatible,
			src.Planes, dst.Planes)
	}
	if src.depth() != dst.depth() {
		return fmt.Errorf("%w: %v-bit input, %v-bit output", ErrIncompatible,
			src.depth(), dst.depth())
	}
	return nil
}

func checkTypes(output, input image.Image) error {
	if reflect.TypeOf(output) != reflect.TypeOf(input) {
		return fmt.Errorf("%w: %T input, %T output", ErrIncompatible, input, output)
	}
	return nil
}

// NewConverter returns a [Converter] for cfg (which is copied), resizing with
// filter (which must not be nil).
func NewConverter(cfg *ConverterConfig, filter Filter) (*Converter, error) {
	if filter == nil {
		return nil, fmt.Errorf("%w: nil filter", ErrInvalidFilter)
	}
	err := checkConversion(&cfg.Output, &cfg.Input)
	if err != nil {
		return nil, err
	}
	ctx := &Converter{
		cfg: *cfg,
	}
	cfg = &ctx.cfg
	if cfg.Threads <= 0 {
		cfg.Threads = runtime.GOMAXPROCS(0)
	}
	depth := cfg.Input.depth()
	if cfg.PremultiplyAlpha {
		if cfg.Input.Pack != 4 || cfg.Input.Planes != 1 {
			return nil, fmt.Errorf("%w: premultiplied filtering of %v, it needs one plane of 4 samples", ErrUnsupportedImage, cfg.Input)
		}
	}
	ctx.premul = cfg.PremultiplyAlpha && cfg.Input.Rect.Size() != cfg.Output.Rect.Size()
	if ctx.premul {
		// the premultiplied planes are 16-bit regardless of the images
		ctx.addScratch(preScratch, cfg.Input.Rect.Dx(), cfg.Input.Rect.Dy(), 8)
		if depth == 8 {
			ctx.addScratch(outScratch, cfg.Output.Rect.Dx(), cfg.Output.Rect.Dy(), 8)
		}
		depth = 16
	}
	bpp := cfg.Input.Pack * depth / 8
	field := bin(cfg.Output.Interlaced)
	var group jobs
	var errs [maxPlanes][2]error
	for i := range cfg.Output.Planes {
		win := cfg.Input.Width(i)
		hin := cfg.Input.Height(i)
		wout := cfg.Output.Width(i)
		hout := cfg.Output.Height(i)
		if win < 2 || hin>>field < 2 {
			return nil, fmt.Errorf("%w: input plane %v is %vx%v", ErrTooSmall, i, win, hin)
		}
		if wout < 1 || hout < 1 {
			return nil, fmt.Errorf("%w: output plane %v is %vx%v", ErrTooSmall, i, wout, hout)
		}
		idx := i
		if win != wout {
			dispatch(&group, cfg.Threads, func() {
				threads := min(cfg.Threads, hout)
				r, err := NewResizer(&ResizerConfig{
					Depth:       depth,
					Input:       win,
					Output:      wout,
					Vertical:    false,
					Interlaced:  false,
					Pack:        cfg.Input.Pack,
					Threads:     threads,
					DisableSIMD: cfg.DisableSIMD,
				}, filter)
				if err != nil {
					errs[idx][0] = err
					return
				}
				ctx.wrez[idx] = r
			})
		}
		if hin != hout {
			dispatch(&group, cfg.Threads, func() {
				threads := max(min(cfg.Threads, hout>>field), 1)
				r, err := NewResizer(&ResizerConfig{
					Depth:       depth,
					Input:       hin,
					Output:      hout,
					Vertical:    true,
					Interlaced:  cfg.Output.Interlaced,
					Pack:        cfg.Output.Pack,
					Threads:     threads,
					DisableSIMD: cfg.DisableSIMD,
				}, filter)
				if err != nil {
					errs[idx][1] = err
					return
				}
				ctx.hrez[idx] = r
			})
		}
		if win != wout && hin != hout {
			ctx.addScratch(i, win, hout, bpp)
		}
	}
	group.Wait()
	for _, e := range errs {
		for _, err := range e {
			if err != nil {
				return nil, err
			}
		}
	}
	return ctx, nil
}

// ChromaRatioOf returns the ChromaRatio of an [image.YCbCrSubsampleRatio],
// and whether it is one of the supported ratios.
func ChromaRatioOf(value image.YCbCrSubsampleRatio) (ChromaRatio, bool) {
	switch value {
	case image.YCbCrSubsampleRatio410:
		return Ratio410, true
	case image.YCbCrSubsampleRatio411:
		return Ratio411, true
	case image.YCbCrSubsampleRatio420:
		return Ratio420, true
	case image.YCbCrSubsampleRatio422:
		return Ratio422, true
	case image.YCbCrSubsampleRatio440:
		return Ratio440, true
	case image.YCbCrSubsampleRatio444:
		return Ratio444, true
	}
	return 0, false
}

func inspect(data image.Image, interlaced bool) (*Descriptor, []plane, error) {
	if v := reflect.ValueOf(data); v.Kind() == reflect.Pointer && v.IsNil() {
		return nil, nil, fmt.Errorf("%w: nil %T", ErrUnsupportedImage, data)
	}
	switch t := data.(type) {
	case *image.YCbCr:
		return inspectYCbCr(t, 3, nil, 0, interlaced)
	case *image.NYCbCrA:
		return inspectYCbCr(&t.YCbCr, maxPlanes, t.A, t.AStride, interlaced)
	case *image.RGBA:
		return inspectPacked(t.Rect, t.Stride, t.Pix, 4, 8, interlaced)
	case *image.NRGBA:
		return inspectPacked(t.Rect, t.Stride, t.Pix, 4, 8, interlaced)
	case *image.Gray:
		return inspectPacked(t.Rect, t.Stride, t.Pix, 1, 8, interlaced)
	case *image.Alpha:
		return inspectPacked(t.Rect, t.Stride, t.Pix, 1, 8, interlaced)
	case *image.CMYK:
		return inspectPacked(t.Rect, t.Stride, t.Pix, 4, 8, interlaced)
	case *image.RGBA64:
		return inspectPacked(t.Rect, t.Stride, t.Pix, 4, 16, interlaced)
	case *image.NRGBA64:
		return inspectPacked(t.Rect, t.Stride, t.Pix, 4, 16, interlaced)
	case *image.Gray16:
		return inspectPacked(t.Rect, t.Stride, t.Pix, 1, 16, interlaced)
	case *image.Alpha16:
		return inspectPacked(t.Rect, t.Stride, t.Pix, 1, 16, interlaced)
	}
	return nil, nil, fmt.Errorf("%w: %T", ErrUnsupportedImage, data)
}

func inspectOutput(data image.Image, interlaced bool) (*Descriptor, []plane, error) {
	d, planes, err := inspect(data, interlaced)
	if err != nil {
		return nil, nil, err
	}
	for i, p := range planes {
		if p.Height > 1 && p.Pitch < p.Width*p.Bpp {
			return nil, nil, fmt.Errorf("%w: output %v with overlapping rows, a stride of %v for plane %v", ErrUnsupportedImage, d, p.Pitch, i)
		}
	}
	return d, planes, nil
}

func inspectYCbCr(img *image.YCbCr, n int, alpha []byte, apitch int, interlaced bool) (*Descriptor, []plane, error) {
	ratio, ok := ChromaRatioOf(img.SubsampleRatio)
	if !ok {
		return nil, nil, fmt.Errorf("%w: %T with subsample ratio %v", ErrUnsupportedImage, img, img.SubsampleRatio)
	}
	d := &Descriptor{
		Rect:       img.Rect,
		Ratio:      ratio,
		Interlaced: interlaced,
		Pack:       1,
		Planes:     n,
	}
	if err := d.checkSize(); err != nil {
		return nil, nil, err
	}
	pix := [maxPlanes][]byte{img.Y, img.Cb, img.Cr, alpha}
	pitches := [maxPlanes]int{img.YStride, img.CStride, img.CStride, apitch}
	planes := make([]plane, d.Planes)
	for i := range planes {
		p, ok := makePlane(d.Width(i), d.Height(i), pitches[i], d.Pack, pix[i])
		if !ok {
			return nil, nil, fmt.Errorf("%w: %v with a stride of %v and %v bytes for plane %v", ErrUnsupportedImage, d, pitches[i], len(pix[i]), i)
		}
		planes[i] = p
	}
	return d, planes, nil
}

func inspectPacked(rect image.Rectangle, pitch int, pix []byte, pack, depth int, interlaced bool) (*Descriptor, []plane, error) {
	d := &Descriptor{
		Rect:       rect,
		Ratio:      Ratio444,
		Interlaced: interlaced,
		Pack:       pack,
		Planes:     1,
		Depth:      depth,
	}
	if err := d.checkSize(); err != nil {
		return nil, nil, err
	}
	p, ok := makePlane(rect.Dx(), rect.Dy(), pitch, pack*depth/8, pix)
	if !ok {
		return nil, nil, fmt.Errorf("%w: %v with a stride of %v and %v bytes", ErrUnsupportedImage, d, pitch, len(pix))
	}
	return d, []plane{p}, nil
}

func makePlane(width, height, pitch, bpp int, pix []byte) (plane, bool) {
	n, ok := planeBytes(width, height, bpp, pitch)
	if pitch < 0 || !ok || len(pix) < n {
		return plane{}, false
	}
	return plane{
		Data:   pix[:n],
		Width:  width,
		Height: height,
		Pitch:  pitch,
		Bpp:    bpp,
	}, true
}

func resizePlane(group *jobs, threads int, dst, src, buf *plane, hrez, wrez *Resizer) {
	dispatch(group, threads, func() {
		hdst := dst
		wsrc := src
		if hrez != nil && wrez != nil {
			hdst = buf
			wsrc = buf
		}
		if hrez != nil {
			hrez.Resize(hdst.Data, src.Data, src.Width, src.Height, hdst.Pitch, src.Pitch)
		}
		if wrez != nil {
			wrez.Resize(dst.Data, wsrc.Data, wsrc.Width, wsrc.Height, dst.Pitch, wsrc.Pitch)
		}
		if hrez == nil && wrez == nil {
			copyPlane(dst.Data, src.Data, src.Width*src.Bpp, src.Height, dst.Pitch, src.Pitch)
		}
	})
}

// Convert resizes src into dst, which must match the descriptors the converter
// was created for, returning an error otherwise. The result is undefined if
// output and input overlap.
func (ctx *Converter) Convert(dst, src image.Image) error {
	id, srcp, err := inspect(src, ctx.cfg.Input.Interlaced)
	if err != nil {
		return err
	}
	od, dstp, err := inspectOutput(dst, ctx.cfg.Output.Interlaced)
	if err != nil {
		return err
	}
	if !id.matches(&ctx.cfg.Input) {
		return fmt.Errorf("%w: input image %v, configured %v", ErrMismatch, id, ctx.cfg.Input)
	}
	if !od.matches(&ctx.cfg.Output) {
		return fmt.Errorf("%w: output image %v, configured %v", ErrMismatch, od, ctx.cfg.Output)
	}
	if ctx.cfg.PremultiplyAlpha && (!isStraight(src) || !isStraight(dst)) {
		return fmt.Errorf("%w: premultiplied filtering of %T to %T, it needs NRGBA or NRGBA64 images", ErrMismatch, src, dst)
	}
	err = checkConversion(od, id)
	if err != nil {
		return err
	}
	err = checkTypes(dst, src)
	if err != nil {
		return err
	}
	scratch, buf := ctx.buffers()
	var group jobs
	if ctx.premul {
		ctx.convertPremultiplied(&group, &dstp[0], &srcp[0], &scratch)
	} else {
		for i := range ctx.cfg.Input.Planes {
			resizePlane(&group, ctx.cfg.Threads, &dstp[i], &srcp[i], &scratch[i], ctx.hrez[i], ctx.wrez[i])
		}
	}
	group.Wait()
	if buf != nil {
		ctx.pool.Put(buf)
	}
	return nil
}

func (ctx *Converter) convertPremultiplied(group *jobs, dst, src *plane, scratch *[maxPlanes + 2]plane) {
	depth := ctx.cfg.Input.depth()
	pre := &scratch[preScratch]
	premultiply(ctx.cfg.Threads, pre, src, depth)
	out := dst
	if depth == 8 {
		out = &scratch[outScratch]
	}
	resizePlane(group, ctx.cfg.Threads, out, pre, &scratch[0], ctx.hrez[0], ctx.wrez[0])
	group.Wait()
	unpremultiply(ctx.cfg.Threads, dst, out, depth)
}

func (ctx *Converter) addScratch(i, width, height, bpp int) {
	p := plane{
		Width:  width,
		Height: height,
		Pitch:  align(width*bpp, 16),
		Bpp:    bpp,
	}
	ctx.size += p.Pitch * p.Height
	ctx.scratch[i] = p
}

func (ctx *Converter) buffers() ([maxPlanes + 2]plane, []byte) {
	scratch := ctx.scratch
	if ctx.size == 0 {
		return scratch, nil
	}
	buf, _ := ctx.pool.Get().([]byte)
	if buf == nil {
		buf = make([]byte, ctx.size)
	}
	idx := 0
	for i, p := range scratch {
		if p.Pitch != 0 {
			scratch[i].Data = buf[idx : idx+p.Pitch*(p.Height-1)+p.Width*p.Bpp]
			idx += p.Pitch * p.Height
		}
	}
	return scratch, buf
}

// Prepare returns the configuration of a conversion from images like input
// to images like output, for [NewConverter].
func Prepare(output, input image.Image) (*ConverterConfig, error) {
	src, _, err := inspect(input, false)
	if err != nil {
		return nil, err
	}
	dst, _, err := inspectOutput(output, false)
	if err != nil {
		return nil, err
	}
	err = checkConversion(dst, src)
	if err != nil {
		return nil, err
	}
	err = checkTypes(output, input)
	if err != nil {
		return nil, err
	}
	return &ConverterConfig{
		Input:  *src,
		Output: *dst,
	}, nil
}

// Convert resizes input into output with filter. The images must be the same
// type. A [Converter] is faster for repeated conversions of identical images.
func Convert(output, input image.Image, filter Filter) error {
	cfg, err := Prepare(output, input)
	if err != nil {
		return err
	}
	converter, err := NewConverter(cfg, filter)
	if err != nil {
		return err
	}
	return converter.Convert(output, input)
}
