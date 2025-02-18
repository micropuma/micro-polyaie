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
struct HoistMemrefSubview
    : public polyaie::HoistMemrefSubviewBase<HoistMemrefSubview> {
  void runOnOperation() override;
};
} // namespace

void HoistMemrefSubview::runOnOperation() {
  auto mod = getOperation();
  auto b = OpBuilder(mod);
  auto topFunc = getTopFunc<FuncOp>(mod);

  // Hold the subviews list of each global memory.
  llvm::SmallDenseMap<Value, SmallVector<Value, 16>, 32> subviewsMap;

  for (auto call : topFunc.getOps<CallOp>()) {
    // Only if the callee is a function, there's a chance to hoist the subviews
    auto func = mod.lookupSymbol<FuncOp>(call.getCallee());
    auto inputTypes = SmallVector<Type, 8>(func.getArgumentTypes().begin(),
                                           func.getArgumentTypes().end());

    for (auto subview :
         llvm::make_early_inc_range(func.getOps<memref::SubViewOp>())) {
      // Only if the source of the current subview is an argument, there's a
      // chance to hoist the subview out of the current sub-function.

      // convert subview source to BlockArgument
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
        // erase 导致整个subview内存没了
        // remove仍旧保留subview
        subview->remove();
        if (auto alloc = memory.getDefiningOp())
          b.setInsertionPointAfter(alloc);
        else
          b.setInsertionPointToStart(memory.getParentBlock());
        b.insert(subview);

        subview.sourceMutable().assign(memory);
        subviews.push_back(subview.result());
        existSubview = &subviews.back();
      }

      // Set the corresponding call operand to the subview.
      call.setOperand(arg.getArgNumber(), *existSubview);
    }

    // Update the function type.
    func.setType(
        b.getFunctionType(inputTypes, func.getFunctionType().getResults()));
  }
}

std::unique_ptr<Pass> polyaie::createHoistMemrefSubviewPass() {
  return std::make_unique<HoistMemrefSubview>();
}
