// Copyright 2013 Benoît Amiaux. All rights reserved. Use of this source code is
// governed by a MIT-style license that can be found in the LICENSE file.

package arez

import (
	"fmt"
	"math"
	"reflect"
	"strings"
)

// Filter is an interpolation filter, which gives the weights of the input
// pixels around an output pixel for a resize by scale (output/input). Filters
// must be safe for concurrent use.
//
// Distances are in input pixels, or in rows of a field for interlaced resizes.
type Filter interface {
	// Support returns the radius of the filter, which must be positive and
	// finite.
	Support(scale float64) float64
	// Weight returns the weight of the input pixel at the signed distance dx
	// from the center of an output pixel (the output center minus the input
	// center). The weights of an output pixel are normalized by their sum. The
	// result must be zero unless -Support(scale) < dx <= Support(scale).
	Weight(dx, scale float64) float64
}

type kernelFilter struct {
	support float64
	k       func(dx float64) float64
}

func (f *kernelFilter) Support(scale float64) float64 {
	return kernelSupport(f.support, scale)
}

func (f *kernelFilter) Weight(dx, scale float64) float64 {
	return kernelWeight(f.k, f.support, dx, scale)
}

func (f *kernelFilter) String() string {
	return fmt.Sprintf("kernel(support=%g)", f.support)
}

// Kernel returns a filter for a symmetric convolution kernel with the specified
// support radius, which is called with distances from 0 to support. The kernel
// is stretched over the output pixels when downscaling.
func Kernel(support float64, k func(dx float64) float64) Filter {
	return &kernelFilter{support, k}
}

func kernelSupport(support, scale float64) float64 {
	return support / min(1, scale)
}

func kernelWeight(k func(float64) float64, support, dx, scale float64) float64 {
	if x := math.Abs(dx) * min(1, scale); x < support {
		return k(x)
	}
	return 0
}

var pkg, _, _ = strings.Cut(reflect.TypeFor[bilinear]().String(), ".") // arez

type bilinear struct{}

func (bilinear) String() string   { return "bilinear" }
func (bilinear) GoString() string { return pkg + ".Bilinear()" }

func (bilinear) Support(scale float64) float64 {
	return kernelSupport(1, scale)
}

func (f bilinear) Weight(dx, scale float64) float64 {
	return kernelWeight(f.get, 1, dx, scale)
}

func (bilinear) get(x float64) float64 {
	return 1 - x
}

// Bilinear returns a bilinear filter.
func Bilinear() Filter {
	return bilinear{}
}

type bicubic struct {
	a, b, c, d, e, f, g float64 // polynomial coefficients
	pb, pc              float64 // parameters
}

func (f *bicubic) String() string {
	return fmt.Sprintf("bicubic(b=%g, c=%g)", f.pb, f.pc)
}

func (f *bicubic) GoString() string {
	if f.pb == 0 && f.pc == 0.5 {
		return pkg + ".Bicubic()"
	}
	return fmt.Sprintf("%s.CustomBicubic(%g, %g)", pkg, f.pb, f.pc)
}

func (*bicubic) Support(scale float64) float64 {
	return kernelSupport(2, scale)
}

func (f *bicubic) Weight(dx, scale float64) float64 {
	return kernelWeight(f.get, 2, dx, scale)
}

func (f *bicubic) get(x float64) float64 {
	if x < 1 {
		return f.a + x*x*(f.b+x*f.c)
	}
	return f.d + x*(f.e+x*(f.f+x*f.g))
}

// CustomBicubic returns the Mitchell-Netravali bicubic filter of the parameters
// b and c. For example, b = c = 1/3 is the filter Mitchell and Netravali
// recommend, and b = 0, c = 0.5 the Catmull-Rom spline of [Bicubic].
func CustomBicubic(b, c float64) Filter {
	f := &bicubic{pb: b, pc: c}
	f.a = 1 - b/3
	f.b = -3 + 2*b + c
	f.c = 2 - 3*b/2 - c
	f.d = 4*b/3 + 4*c
	f.e = -2*b - 8*c
	f.f = b + 5*c
	f.g = -b/6 - c
	return f
}

// Bicubic returns the Catmull-Rom bicubic filter, CustomBicubic(0, 0.5).
func Bicubic() Filter {
	return CustomBicubic(0, 0.5)
}

type lanczos struct {
	alpha float64
}

func (f lanczos) String() string {
	return fmt.Sprintf("lanczos(a=%g)", f.alpha)
}

func (f lanczos) GoString() string {
	return fmt.Sprintf("%s.Lanczos(%g)", pkg, f.alpha)
}

func (f lanczos) Support(scale float64) float64 {
	return kernelSupport(f.alpha, scale)
}

func (f lanczos) Weight(dx, scale float64) float64 {
	return kernelWeight(f.get, f.alpha, dx, scale)
}

func (f lanczos) get(x float64) float64 {
	if x == 0 {
		return 1
	}
	b := x * math.Pi
	c := b / f.alpha
	return math.Sin(b) * math.Sin(c) / (b * c)
}

// Lanczos returns a Lanczos filter with a support of alpha pixels, commonly 2
// or 3.
func Lanczos(alpha float64) Filter {
	return lanczos{alpha: alpha}
}

type nearest struct{}

func (nearest) String() string   { return "nearest" }
func (nearest) GoString() string { return pkg + ".Nearest()" }

func (nearest) Support(float64) float64 {
	return 0.5
}

func (nearest) Weight(dx, _ float64) float64 {
	if dx > -0.5 && dx <= 0.5 {
		return 1
	}
	return 0
}

// Nearest returns a nearest-neighbor filter.
func Nearest() Filter {
	return nearest{}
}

type box struct{}

func (box) String() string   { return "box" }
func (box) GoString() string { return pkg + ".Box()" }

func (box) Support(scale float64) float64 {
	return 0.5/scale + 0.5
}

func (box) Weight(dx, scale float64) float64 {
	// overlap of the output pixel, 1/scale input pixels wide, and the input
	// pixel
	w, d := 0.5/scale, math.Abs(dx)
	return max(0, min(w, 0.5-d)+min(w, 0.5+d))
}

// Box returns a box filter.
func Box() Filter {
	return box{}
}

// filterString returns the name of filter for errors, its String if it
// implements [fmt.Stringer], or its type otherwise.
func filterString(filter Filter) string {
	if s, ok := filter.(fmt.Stringer); ok {
		return s.String()
	}
	t := reflect.TypeOf(filter)
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	return t.String()
}
