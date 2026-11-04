// Copyright 2026 Patrick Gaskin. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

//go:build !(goexperiment.simd && (amd64 || arm64 || wasm))

package arez

var _ = setScalers(
	func() bool { return false },
	func(cfg *ResizerConfig, k kernel) kernel { return k },
	func(k *kernel, simd bool) scaler { return getHorizontalScalerGo(k.size, k.wide) },
	func(k *kernel, simd bool) scaler { return getVerticalScalerGo(k.size, k.wide) },
)
