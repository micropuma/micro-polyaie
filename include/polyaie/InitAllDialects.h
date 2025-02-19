//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The PolyAIE Authors.
//
//===----------------------------------------------------------------------===//

#ifndef POLYAIE_INITALLDIALECTS_H
#define POLYAIE_INITALLDIALECTS_H

// 先注释aie部分，后续再添加
#include "polyaie/AIE/AIEDialect.h"
// #include "polyaie/AIE/Dialect/AIEVec/IR/AIEVecOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "polyaie/Dataflow/Dataflow.h"

namespace mlir {
namespace polyaie {

// Add all the related dialects to the provided registry.
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  // clang-format off
  // 注册所有的dialect
  registry.insert<
    mlir::func::FuncDialect,
    mlir::cf::ControlFlowDialect,
    mlir::AffineDialect,
    mlir::scf::SCFDialect,
    mlir::memref::MemRefDialect,
    mlir::arith::ArithmeticDialect,
    mlir::math::MathDialect,
    mlir::vector::VectorDialect,
    mlir::bufferization::BufferizationDialect,
    mlir::LLVM::LLVMDialect,

    // 自定义DataflowDialect，对c++代码到aie的systolic阵列提供抽象
    polyaie::dataflow::DataflowDialect,
    // 先注释AIE部分，后续再添加
    xilinx::AIE::AIEDialect
    // xilinx::aievec::AIEVecDialect
  >();
  // clang-format on
}

} // namespace polyaie
} // namespace mlir

#endif // POLYAIE_INITALLDIALECTS_H
