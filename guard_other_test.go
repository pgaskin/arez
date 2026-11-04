// Copyright 2026 Patrick Gaskin. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

//go:build !unix

package arez

import (
	"flag"
	"fmt"
	"image"
	"os"
	"runtime"
	"testing"
)

func init() {
	flag.Func("arez.guard", "allocate every test image on guarded memory (unsupported on "+runtime.GOOS+")", func(string) error {
		fmt.Fprintf(os.Stderr, "arez.guard: not supported on %v\n", runtime.GOOS)
		return nil
	})
}

func guardImage(testing.TB, image.Image) {}
