//===- vulkan-runtime-wrappers.cpp - MLIR Vulkan runner wrapper library ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <numeric>

#include "llvm/Support/raw_ostream.h"

// A struct that corresponds to how MLIR represents memrefs.
template <typename T, int N> struct MemRefType {
  T *basePtr;
  T *data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

template <typename T, int N>
void memHostRegisterMemRef(const MemRefType<T, N> *arg, T value) {
  auto count = std::accumulate(arg->sizes, arg->sizes + N, 1,
                               std::multiplies<int64_t>());
  std::fill_n(arg->data, count, value);
  mcuMemHostRegister(arg->data, count * sizeof(T));
}

extern "C" void memHostRegisterMemRef1dFloat(const MemRefType<float, 1> *arg) {
}

extern "C" void printMemRegister() { llvm::errs() << "call foo "; }
