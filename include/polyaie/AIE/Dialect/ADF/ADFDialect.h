//===- ADFDialect.h ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef XILINX_ADF_DIALECT_H
#define XILINX_ADF_DIALECT_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#include "polyaie/AIE/Dialect/ADF/ADFDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "polyaie/AIE/Dialect/ADF/ADFTypes.h.inc"

namespace xilinx {
namespace ADF {

#define GEN_PASS_CLASSES
#include "polyaie/AIE/Dialect/ADF/ADFPasses.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "polyaie/AIE/Dialect/ADF/ADFPasses.h.inc"

} // namespace ADF
} // namespace xilinx

#endif // XILINX_ADF_DIALECT_H
