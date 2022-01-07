//===----------------------------------------------------------------------===//
//
// Copyright 2021-2021 The PolyAIE Authors.
//
//===----------------------------------------------------------------------===//

#include "polyaie/Transforms/Passes.h"

using namespace mlir;
using namespace polyaie;

namespace {
struct HoistMemrefSubview
    : public polyaie::HoistMemrefSubviewBase<HoistMemrefSubview> {
  // Hold the subviews list of each global memory.
  DenseMap<Value, SmallVector<Value, 16>> subviewsMap;

  void runOnOperation() override;
};
} // namespace

void HoistMemrefSubview::runOnOperation() {
  auto mod = getOperation();
  auto b = OpBuilder(mod);

  for (auto call : mod.getOps<CallOp>()) {
    auto func = mod.lookupSymbol<FuncOp>(call.callee());
    auto inputTypes = SmallVector<Type, 8>(func.getArgumentTypes());

    for (auto subview :
         llvm::make_early_inc_range(func.getOps<memref::SubViewOp>())) {
      // Only if the source of the current subview is an argument, there's a
      // chance to hoist the subview out of the current sub-function.
      auto arg = subview.source().dyn_cast<BlockArgument>();
      if (!arg)
        continue;

      // Replace uses of the current subview with the argument in order to
      // prepare for the hoisting.
      inputTypes[arg.getArgNumber()] = subview.getType();
      arg.setType(subview.getType());
      subview.replaceAllUsesWith(arg);

      // Get the existing subviews list of the current memory.
      auto memory = call.getOperand(arg.getArgNumber());
      auto &subviews = subviewsMap[memory];

      // Figure out if there exists an subview that has the same type with the
      // current subview.
      auto existSubview = llvm::find_if(
          subviews, [&](Value v) { return v.getType() == subview.getType(); });

      // If the subview already exists, we can safely erase the curren subview.
      // Otherwise, we need to move it to outside.
      if (existSubview != subviews.end())
        subview.erase();
      else {
        subview.sourceMutable().assign(memory);
        subview->moveAfter(memory.getDefiningOp());
        subviews.push_back(subview.result());
        existSubview = &subviews.back();
      }

      // Set the corresponding call operand to the subview.
      call.setOperand(arg.getArgNumber(), *existSubview);
    }

    // Update the function type.
    func.setType(b.getFunctionType(inputTypes, func.getType().getResults()));
  }
}

std::unique_ptr<Pass> polyaie::createHoistMemrefSubviewPass() {
  return std::make_unique<HoistMemrefSubview>();
}
