//===----------------------------------------------------------------------===//
//
// Copyright 2021-2021 The PolyAIE Authors.
//
//===----------------------------------------------------------------------===//

#include "polyaie/MemRefExt/MemRefExt.h"
#include "polyaie/Transforms/Passes.h"

using namespace mlir;
using namespace polyaie;
using namespace memrefext;

namespace {
struct CreateDataflow : public polyaie::CreateDataflowBase<CreateDataflow> {
  void runOnOperation() override;
};
} // namespace

namespace {
/// This is a class used to maintain a list of buffers with different types.
class BufferList {
public:
  Value getBuffer(Type type) const {
    auto existBufPtr =
        llvm::find_if(impl, [&](Value v) { return v.getType() == type; });
    return existBufPtr != impl.end() ? *existBufPtr : nullptr;
  }

  void updateBuffer(Value newBuf) {
    auto oldBufPtr = llvm::find_if(
        impl, [&](Value v) { return v.getType() == newBuf.getType(); });
    if (oldBufPtr != impl.end())
      impl[oldBufPtr - impl.begin()] = newBuf;
    else
      impl.push_back(newBuf);
  }

private:
  SmallVector<Value, 32> impl;
};
} // namespace

void CreateDataflow::runOnOperation() {
  auto mod = getOperation();

  // Construct a map from memory to its buffer list.
  DenseMap<Value, BufferList> bufListMap;
  for (auto alloc : mod.getOps<memref::AllocOp>())
    bufListMap[alloc.getResult()] = BufferList();

  // Create dataflow between function calls.
  for (auto call : llvm::make_early_inc_range(mod.getOps<CallOp>())) {
    for (auto buf : call.getOperands()) {
      auto loadOp = buf.getDefiningOp<LoadBufferOp>();
      auto &bufList = bufListMap[loadOp.memory()];

      // Find the result buffer if it exists.
      auto resultBufPtr = llvm::find_if(call.getResults(), [&](OpResult r) {
        auto storeOp = cast<StoreBufferOp>(*r.getUsers().begin());
        return r.getType() == buf.getType() &&
               storeOp.memory() == loadOp.memory();
      });

      // Query the buffer list to check whether there is a buffer with the same
      // type existing. If so, replace the buffer generated by LoadBufferOp with
      // the exist buffer and erase the LoadBufferOp.
      if (auto existBuf = bufList.getBuffer(buf.getType())) {
        loadOp.replaceAllUsesWith(existBuf);
        loadOp.erase();
      }

      // Update the buffer list with the current buffer if applicable.
      if (resultBufPtr != call.getResults().end())
        bufList.updateBuffer(*resultBufPtr);
    }
  }

  auto b = OpBuilder(mod);
  for (auto &op : llvm::make_early_inc_range(mod.getBody()->getOperations())) {
    // As long as the buffer is used by operations other than a StoreBufferOp,
    // the StoreBufferOp can be identified as redundant.
    // FIXME: If a buffer is accessed by a function but not updated in the end,
    // the buffer will be identified as redundant. But actually this is not the
    // case, we should store all buffers that are updated back.
    if (auto storeOp = dyn_cast<StoreBufferOp>(op))
      if (!llvm::hasSingleElement(storeOp.buffer().getUsers()))
        storeOp.erase();

    // Remove layout map from all memories.
    if (auto loadOp = dyn_cast<LoadBufferOp>(op)) {
      // Update the type of load buffer operations.
      auto type = loadOp.getType();
      loadOp.getResult().setType(
          MemRefType::get(type.getShape(), type.getElementType()));

    } else if (auto call = dyn_cast<CallOp>(op)) {
      auto func = mod.lookupSymbol<FuncOp>(call.callee());

      // Update the type of functions.
      for (auto arg : func.getArguments()) {
        auto type = arg.getType().cast<MemRefType>();
        arg.setType(MemRefType::get(type.getShape(), type.getElementType()));
      }
      auto resultTypes = func.front().getTerminator()->getOperandTypes();
      func.setType(b.getFunctionType(func.getArgumentTypes(), resultTypes));

      // Update the type of calls.
      for (auto resultAndType : llvm::zip(call.getResults(), resultTypes))
        std::get<0>(resultAndType).setType(std::get<1>(resultAndType));
    }
  }
}

std::unique_ptr<Pass> polyaie::createCreateDataflowPass() {
  return std::make_unique<CreateDataflow>();
}
