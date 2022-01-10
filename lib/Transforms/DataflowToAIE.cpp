//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The PolyAIE Authors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/PassManager.h"
#include "polyaie/Transforms/Passes.h"
#include "polyaie/Utils.h"

using namespace mlir;
using namespace polyaie;
using namespace dataflow;
using namespace xilinx::AIE;

namespace {
struct DataflowToAIE : public polyaie::DataflowToAIEBase<DataflowToAIE> {
  DataflowToAIE() = default;
  DataflowToAIE(const DataflowToAIE &) {}
  DataflowToAIE(const PolyAIEOptions &opts) { vecSize = opts.vectorizeSize; }

  void runOnOperation() override;

  /// Map from a buffer to its destination buffers. All values here must be
  /// defined by BufferOps.
  DenseMap<Value, SmallVector<Value, 4>> destBufsMap;
};
} // namespace

static Block *getCoreBlock(BufferOp buf) {
  return &buf.getTileOp().getCoreOp().body().front();
}

static void createLoopNest(OpBuilder &b, Value srcBuf, Value tgtBuf,
                           int64_t vecSize) {
  auto loc = b.getUnknownLoc();
  auto type = srcBuf.getType().cast<MemRefType>();
  auto rank = std::max(type.getRank(), (int64_t)1);

  SmallVector<Value, 4> ivs;
  unsigned loopIdx = 0;
  for (auto dimSize : type.getShape()) {
    // The last loop level should has a step of "vecSize".
    auto loop = b.create<mlir::AffineForOp>(loc, 0, dimSize,
                                            ++loopIdx == rank ? vecSize : 1);
    b.setInsertionPointToStart(loop.getBody());
    ivs.push_back(loop.getInductionVar());
  }

  // Create affine load/store operations or vector transfer read/write
  // operations according to the value of vector size.
  if (vecSize == 1) {
    auto value = b.create<mlir::AffineLoadOp>(loc, srcBuf, ivs);
    b.create<mlir::AffineStoreOp>(loc, value, tgtBuf, ivs);
  } else {
    auto vecType = VectorType::get({vecSize}, type.getElementType());
    auto value = b.create<vector::TransferReadOp>(loc, vecType, srcBuf, ivs);
    b.create<vector::TransferWriteOp>(loc, value, tgtBuf, ivs);
  }
}

static void createMemCpy(OpBuilder &b, BufferOp srcBuf, BufferOp tgtBuf,
                         StringRef tokName, unsigned tokValA, unsigned tokValR,
                         bool shareSrcTile, int64_t vecSize) {
  auto loc = b.getUnknownLoc();

  // If the shareable tile is source tile, then create memory copy in the end of
  // CoreOp of the source tile. Otherwise, create in the start of CoreOp of the
  // target tile.
  if (shareSrcTile)
    b.setInsertionPoint(getCoreBlock(srcBuf)->getTerminator());
  else
    b.setInsertionPointToStart(getCoreBlock(tgtBuf));

  // Generate token acquire and release.
  b.create<UseTokenOp>(loc, tokName, tokValA, LockAction::Acquire);
  auto useOp = b.create<UseTokenOp>(loc, tokName, tokValR, LockAction::Release);
  b.setInsertionPoint(useOp);

  // Create loop nests to copy data from source to target.
  createLoopNest(b, srcBuf, tgtBuf, vecSize);
}

static void fillMemOpBlocks(OpBuilder &b, Block *dmaBlock, Block *bdBlock,
                            Block *endBlock, BufferOp buf, StringRef tokName,
                            unsigned tokValA, unsigned tokValR, DMAChan chan) {
  auto loc = b.getUnknownLoc();

  // Create DMA start block.
  b.setInsertionPointToStart(dmaBlock);
  b.create<DMAStartOp>(loc, chan, bdBlock, endBlock);

  // Create DMA descripter block.
  b.setInsertionPointToStart(bdBlock);
  b.create<UseTokenOp>(loc, tokName, tokValA, LockAction::Acquire);
  b.create<DMABDOp>(loc, buf, /*offset=*/0, buf.getType().getNumElements(), 0);
  b.create<UseTokenOp>(loc, tokName, tokValR, LockAction::Release);
  b.create<BranchOp>(loc, bdBlock);

  // Create DMA end block.
  if (endBlock->empty()) {
    b.setInsertionPointToStart(endBlock);
    b.create<xilinx::AIE::EndOp>(loc);
  }
}

static void updateMemOp(OpBuilder &b, BufferOp buf, StringRef tokName,
                        unsigned tokValA, unsigned tokValR, DMAChan chan) {
  auto tile = buf.getTileOp();

  if (auto mem = tile.getMemOp()) {
    auto &endBlock = mem.body().back();
    auto &predDmaBlock = *std::prev(mem.body().end(), 3);

    auto dmaBlock = b.createBlock(&endBlock);
    auto bdBlock = b.createBlock(&endBlock);

    auto predDma = cast<DMAStartOp>(&predDmaBlock.front());
    predDma.setSuccessor(dmaBlock, /*index=*/1);

    fillMemOpBlocks(b, dmaBlock, bdBlock, &endBlock, buf, tokName, tokValA,
                    tokValR, chan);
  } else {
    // If MemOp has not been created, create a new one.
    b.setInsertionPointAfter(tile.getCoreOp());
    auto newMem = b.create<MemOp>(b.getUnknownLoc(), tile);

    auto &dmaBlock = newMem.body().emplaceBlock();
    auto &bdBlock = newMem.body().emplaceBlock();
    auto &endBlock = newMem.body().emplaceBlock();

    fillMemOpBlocks(b, &dmaBlock, &bdBlock, &endBlock, buf, tokName, tokValA,
                    tokValR, chan);
  }
}

/// Return it if the given memref is casted to a tensor.
static Value getTensor(Value memref) {
  for (auto user : memref.getUsers())
    if (auto toTensorOp = dyn_cast<bufferization::ToTensorOp>(user))
      return toTensorOp.memref();
  return Value();
}

void DataflowToAIE::runOnOperation() {
  auto mod = getOperation();
  auto b = OpBuilder(mod);
  auto loc = b.getUnknownLoc();
  auto topFunc = getTopFunc(mod);

  // Map from ProcessOp result to its corresponding BufferOp.
  DenseMap<Value, BufferOp> bufMap;

  // Generate TileOp, BufferOps, and CoreOp.
  for (auto process : topFunc.getOps<dataflow::ProcessOp>()) {
    // Generate TileOp based on the placement results.
    b.setInsertionPointAfter(process);
    auto tile = b.create<TileOp>(loc, getCol(process), getRow(process));

    // Create BufferOp for each local buffer.
    // TODO: Make this more robust. The current implementation assumes the
    // BufferStateChangedMemref pass has been applied.
    for (auto &op : llvm::make_early_inc_range(process.body().getOps())) {
      if (auto alloc = dyn_cast<memref::AllocOp>(op)) {
        // This buffer may be accessed by other processes, thus we record the
        // mapping in "bufMap".
        auto buf = b.create<BufferOp>(loc, alloc.getType(), tile);
        if (auto tensor = getTensor(alloc.memref())) {
          auto result = process.getResult(tensor);
          bufMap[result] = buf;
        }
        alloc.replaceAllUsesWith(buf.getResult());
        alloc.erase();

      } else if (auto toMemrefOp = dyn_cast<bufferization::ToMemrefOp>(op)) {
        // This buffer should not be accessed by other processes.
        auto buf = b.create<BufferOp>(
            loc, toMemrefOp.getType().cast<MemRefType>(), tile);
        auto operand = process.getOperand(toMemrefOp.tensor());
        destBufsMap[bufMap[operand]].push_back(buf);

        toMemrefOp.replaceAllUsesWith(buf.getResult());
        toMemrefOp.erase();
      }
    }

    // Create BufferOp for each scalar argument.
    for (auto arg : process.body().getArguments()) {
      if (arg.getType().isa<TensorType>())
        continue;

      auto type = MemRefType::get({1}, arg.getType());
      auto buf = b.create<BufferOp>(loc, type, tile);
      auto insertPoint = b.saveInsertionPoint();

      b.setInsertionPointToStart(&process.body().front());
      auto value = b.create<mlir::AffineLoadOp>(loc, b.getConstantAffineMap(0),
                                                ValueRange(buf.getResult()));
      arg.replaceAllUsesWith(value);
      b.restoreInsertionPoint(insertPoint);
    }

    // Erase all process arguments as all of them have been converted.
    process.body().front().eraseArguments([](auto) { return true; });

    // Generate a CoreOp and inline the contents of the process.
    auto core = b.create<CoreOp>(loc, tile);
    auto &procBlocks = process.body().getBlocks();
    auto &coreBlocks = core.body().getBlocks();
    coreBlocks.splice(coreBlocks.begin(), procBlocks);

    // Replace the ternimator with an EndOp.
    auto returnOp = coreBlocks.back().getTerminator();
    b.setInsertionPoint(returnOp);
    b.create<xilinx::AIE::EndOp>(loc);
    returnOp->erase();
  }

  // Remove all process operations.
  // for (auto process :
  //      llvm::make_early_inc_range(topFunc.getOps<dataflow::ProcessOp>())) {
  //   process->dropAllUses();
  //   process->erase();
  // }
  bufMap.clear();
}

std::unique_ptr<Pass> polyaie::createDataflowToAIEPass() {
  return std::make_unique<DataflowToAIE>();
}
std::unique_ptr<Pass>
polyaie::createDataflowToAIEPass(const PolyAIEOptions &opts) {
  return std::make_unique<DataflowToAIE>(opts);
}
