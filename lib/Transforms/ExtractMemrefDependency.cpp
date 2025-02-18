//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The PolyAIE Authors.
//
//===----------------------------------------------------------------------===//

#include "polyaie/Transforms/Passes.h"
#include "polyaie/Utils.h"

using namespace mlir;
using namespace polyaie;
using namespace func;

namespace {
struct ExtractMemrefDependency
    : public polyaie::ExtractMemrefDependencyBase<ExtractMemrefDependency> {
  void runOnOperation() override;
};
} // namespace

// 这个 Pass 主要目的是优化内存操作，减少冗余拷贝，同时确保子视图的复用。
void ExtractMemrefDependency::runOnOperation() {
  auto mod = getOperation();
  auto topFunc = getTopFunc<FuncOp>(mod);

  // Hold the subviews list (or itself) of each global memory.
  // 存储一个全局内存和它的子视图列表（或者它本身）。
  // 这个列表是一个键值对，键是全局内存，值是一个 Value 的列表。
  llvm::SmallDenseMap<Value, SmallVector<Value, 16>, 32> subviewsMap;

  // Construct explicit dependencies between memrefs.
  for (auto call : llvm::make_early_inc_range(topFunc.getOps<CallOp>())) {
    auto func = mod.lookupSymbol<FuncOp>(call.getCallee());
    auto returnOp = func.back().getTerminator();

    for (auto &operand : call->getOpOperands()) {
      // Get the existing subviews list of the current memory.
      auto memory = operand.get();

      // operand的define point如果是subview，则取subview的source作为依赖
      if (auto subview = memory.getDefiningOp<memref::SubViewOp>())
        memory = subview.source();
      // subviews去memory中的映射
      auto &subviews = subviewsMap[memory];

      // Figure out if there exists an subview that has the same type. If so,
      // replace the current operand with the existing subview.
      auto existSubview = llvm::find_if(subviews, [&](Value v) {
        return v.getType() == operand.get().getType();
      });
      if (existSubview != subviews.end())
        call.setOperand(operand.getOperandNumber(), *existSubview);

      // Update the subview list with the current subview if it is returned.
      auto result = llvm::find_if(call.getResults(), [&](OpResult result) {
        return func.getArgument(operand.getOperandNumber()) ==
               returnOp->getOperand(result.getResultNumber());
      });
      if (result != call.getResults().end()) {
        if (existSubview != subviews.end())
          subviews[existSubview - subviews.begin()] = *result;
        else
          subviews.push_back(*result);
      }
    }
  }

  // As we have extracted memref dependencies, we can safely remove redundant
  // copy operations now.
  SmallVector<memref::CopyOp, 16> latestCopies;
  for (auto copy :
       llvm::make_early_inc_range(topFunc.getOps<memref::CopyOp>())) {
    auto latestCopy = llvm::find_if(latestCopies, [&](memref::CopyOp op) {
      return op.target() == copy.target() &&
             op.source().getType() == copy.source().getType();
    });

    if (latestCopy != latestCopies.end()) {
      latestCopy->erase();
      latestCopies[latestCopy - latestCopies.begin()] = copy;
    } else
      latestCopies.push_back(copy);
  }
}

std::unique_ptr<Pass> polyaie::createExtractMemrefDependencyPass() {
  return std::make_unique<ExtractMemrefDependency>();
}
