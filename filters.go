// Copyright 2013 Benoît Amiaux. All rights reserved. Use of this source code is
// governed by a MIT-style license that can be found in the LICENSE file.

package arez

import (
	"math"
)

// Filter is an interpolation filter, which gives the weights of the input
// pixels around an output pixel. Filters must be safe for concurrent use.
type Filter interface {
	// Support returns the radius of the filter, which must be positive and
	// finite.
	Support() float64
	// Name returns the name of the filter.
	Name() string
	// Get returns the weight of an input pixel at the distance dx from the
	// center of an output pixel, in input pixels when upscaling and in output
	// pixels when downscaling. The weights of an output pixel are normalized by
	// their sum. When dx >= Support(), the result is zero.
	Get(dx float64) float64
}

type bilinear struct{}

func (bilinear) Support() float64 { return 1 }
func (bilinear) Name() string     { return "bilinear" }

func (bilinear) Get(x float64) float64 {
	if x < 1 {
		return 1 - x
	}
	return 0
}

// Bilinear returns a bilinear filter.
func Bilinear() Filter {
	return bilinear{}
}

type bicubic struct {
	a, b, c, d, e, f, g float64
}

func (*bicubic) Support() float64 {
	return 2
}

func (*bicubic) Name() string {
	return "bicubic"
}

func (f *bicubic) Get(x float64) float64 {
	if x < 1 {
		return f.a + x*x*(f.b+x*f.c)
	} else if x < 2 {
		return f.d + x*(f.e+x*(f.f+x*f.g))
	}
	return 0
}

// CustomBicubic returns the Mitchell-Netravali bicubic filter of the parameters
// b and c. For example, b = c = 1/3 is the filter Mitchell and Netravali
// recommend, and b = 0, c = 0.5 the Catmull-Rom spline of [Bicubic].
func CustomBicubic(b, c float64) Filter {
	f := &bicubic{}
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

func (f lanczos) Support() float64 {
	return f.alpha
}

func (lanczos) Name() string {
	return "lanczos"
}

func (f lanczos) Get(x float64) float64 {
	if x >= f.alpha {
		return 0
	} else if x == 0 {
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
