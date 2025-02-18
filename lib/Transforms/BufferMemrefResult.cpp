//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The PolyAIE Authors.
//
//===----------------------------------------------------------------------===//

#include "polyaie/Transforms/Passes.h"
#include "polyaie/Utils.h"

using namespace mlir;
using namespace polyaie;

namespace {
struct BufferMemrefResult
    : public polyaie::BufferMemrefResultBase<BufferMemrefResult> {
  void runOnOperation() override;
};
} // namespace

// 在每一个subfunction的头部，开辟buffer，存储result
// 因为result可能需要多次使用，所以需要存储
void BufferMemrefResult::runOnOperation() {
  auto func = getOperation();
  if (func->hasAttr("polyaie.top_func"))
    return;
  auto b = OpBuilder(func);
  auto loc = b.getUnknownLoc();

  for (auto operand : func.back().getTerminator()->getOperands()) {
    // Get the result type and bypass non-memref results.
    // return type 是memref才可能需要bufferization
    auto resultType = operand.getType().dyn_cast<MemRefType>();
    if (!resultType)
      continue;

    // Create a local buffer.
    b.setInsertionPointToStart(&func.front());
    auto buf = b.create<memref::AllocOp>(loc, resultType);

    // 所有针对该operation value的使用，都替换为alloc开辟的buffer
    operand.replaceAllUsesWith(buf);

    // 找到上一级的插入点
    auto firstUser = *buf->user_begin();
    b.setInsertionPoint(func.getBody().findAncestorOpInRegion(*firstUser));

    // Create explicit memory copy using an affine loop nest.
    // 学习如何用affine loop构造拷贝操作。

    // 索引变量
    SmallVector<Value, 4> ivs;
    for (auto dimSize : resultType.getShape()) {
      // loc是func的头部
      auto loop = b.create<mlir::AffineForOp>(loc, 0, dimSize);
      // 下一个循环的AffineForOp的插入点是当前循环的body的头部
      // 这个插入是动态调整的
      b.setInsertionPointToStart(loop.getBody());
      // ivs存储索引变量
      ivs.push_back(loop.getInductionVar());
    }

    // Create affine load/store operations.
    // 这是在最内层的循环中进行的操作
    // 注意：affine_map 通过 MemRefType 传递
    // 因此这里的operand里就是按照affine_map拷贝
    auto value = b.create<mlir::AffineLoadOp>(loc, operand, ivs);
    b.create<mlir::AffineStoreOp>(loc, value, buf, ivs);
  }
}

std::unique_ptr<Pass> polyaie::createBufferMemrefResultPass() {
  return std::make_unique<BufferMemrefResult>();
}
