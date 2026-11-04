// Copyright 2026 Patrick Gaskin. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

package arez

import (
	"testing"
)

func TestU8(t *testing.T) {
	expect(t, u8(-1), byte(0))
	expect(t, u8(0), byte(0))
	expect(t, u8(255), byte(255))
	expect(t, u8(256), byte(255))
}
