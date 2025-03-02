//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The PolyAIE Authors.
//
//===----------------------------------------------------------------------===//

#include "polyaie/Transforms/Passes.h"

using namespace mlir;
using namespace polyaie;
using namespace dataflow;

// 这个pass显示的reorder operations
// 1. aie.tile
// 2. aie.lock
// 3. aie.buffer
// 4. aie.core
namespace {
struct ReorderOperation : public polyaie::ReorderOperationBase<ReorderOperation> {
  void runOnOperation() override;
};
} // namespace

void ReorderOperation::runOnOperation() {
  // 获取构造器
  ModuleOp module = getOperation();
  OpBuilder builder(module.getContext());

  SmallVector<Operation*, 4> tile_ops, lock_ops, buffer_ops, core_ops;

  // 遍历 module 内的 operations 并分类
  for (Operation &op : llvm::make_early_inc_range(module.getBody()->getOperations())) {
    if (op.getName().getStringRef() == "aie.tile") {
      tile_ops.push_back(&op);
    } else if (op.getName().getStringRef() == "aie.lock") {
      lock_ops.push_back(&op);
    } else if (op.getName().getStringRef() == "aie.buffer") {
      buffer_ops.push_back(&op);
    } else if (op.getName().getStringRef() == "aie.core") {
      core_ops.push_back(&op);
    }
  }

  if (module.getBody()->empty()) return;

  // 记录 module 体的最后一个 Operation
  Operation *lastOp = &module.getBody()->back();

  // 依次移动到 module 体的末尾，确保严格顺序
  for (Operation *op : tile_ops) {
    op->moveBefore(lastOp);
  }
  for (Operation *op : lock_ops) {
    op->moveBefore(lastOp);
  }
  for (Operation *op : buffer_ops) {
    op->moveBefore(lastOp);
  }
  for (Operation *op : core_ops) {
    op->moveBefore(lastOp);
  }
}

std::unique_ptr<Pass> polyaie::createReorderOperationPass() {
  return std::make_unique<ReorderOperation>();
}