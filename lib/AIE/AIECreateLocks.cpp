//===- AIECreateLocks.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "polyaie/AIE/AIEDialect.h"
#include "polyaie/AIE/AIETokenAnalysis.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "aie-create-locks"
using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

struct Token2LockLowering : public OpConversionPattern<UseTokenOp> {
  using OpConversionPattern<UseTokenOp>::OpConversionPattern;
  ModuleOp &module;
  DenseMap<Operation *, std::vector<std::pair<Value, int>>> &acqLocks;
  DenseMap<Operation *, std::vector<std::pair<Value, int>>> &relLocks;
  DenseMap<std::pair<Operation *, Operation *>, std::pair<Value, int>>
      &lockChains;

  Token2LockLowering(
      MLIRContext *context, ModuleOp &m,
      DenseMap<Operation *, std::vector<std::pair<Value, int>>> &acqLocks,
      DenseMap<Operation *, std::vector<std::pair<Value, int>>> &relLocks,
      DenseMap<std::pair<Operation *, Operation *>, std::pair<Value, int>>
          &lockChains,
      PatternBenefit benefit = 1)
      : OpConversionPattern<UseTokenOp>(context, benefit), module(m),
        acqLocks(acqLocks), relLocks(relLocks), lockChains(lockChains) {}

  LogicalResult
  matchAndRewrite(UseTokenOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();
    Operation *parentOp = op->getParentOp();

    int srcCol, srcRow;
    bool IsParentMemOp = false;
    if (CoreOp core = dyn_cast<CoreOp>(parentOp)) {
      srcCol = core.colIndex();
      srcRow = core.rowIndex();
    } else if (MemOp mem = dyn_cast<MemOp>(parentOp)) {
      srcCol = mem.colIndex();
      srcRow = mem.rowIndex();
      IsParentMemOp = true;
    } else if (auto shimDma = dyn_cast<ShimDMAOp>(parentOp)) {
      srcCol = shimDma.colIndex();
      srcRow = shimDma.rowIndex();
    } else {
      llvm_unreachable("A parent operation of UseTokenOp must be either CoreOp "
                       "or MemOp or ShimDMAOp");
    }

    if (op.acquire()) {
      // Acquire lock from pair
      LLVM_DEBUG(llvm::dbgs() << "Replacing Acquire: " << op << "\n");
      for (auto acqLock : acqLocks[op]) {
        Value lockFromPair = acqLock.first;
        int lockValueFromPair = acqLock.second;
        rewriter.create<UseLockOp>(op.getLoc(), lockFromPair, lockValueFromPair,
                                   LockAction::Acquire);
        LLVM_DEBUG(llvm::dbgs() << "Acquire from pair " << lockFromPair
                                << " with " << lockValueFromPair << "\n");
      }

      // Acquire lock from chain
      for (auto map : lockChains) {
        Value lockFromChain = map.second.first;
        int lockValueFromChain = map.second.second;
        Operation *acqOp = map.first.second;
        if (acqOp != Op)
          continue;

        rewriter.create<UseLockOp>(op.getLoc(), lockFromChain,
                                   lockValueFromChain, LockAction::Acquire);
        LLVM_DEBUG(llvm::dbgs() << "Acquire from chain " << lockFromChain
                                << " with " << lockValueFromChain << "\n");
      }
    } else if (op.release()) {
      // Release lock from pair
      LLVM_DEBUG(llvm::dbgs() << "Replacing Release: " << op << "\n");
      for (auto relLock : relLocks[op]) {
        Value lockFromPair = relLock.first;
        int lockValueFromPair = relLock.second;
        rewriter.create<UseLockOp>(op.getLoc(), lockFromPair, lockValueFromPair,
                                   LockAction::Release);
        LLVM_DEBUG(llvm::dbgs() << "Release from pair " << lockFromPair
                                << " with " << lockValueFromPair << "\n");
      }

      // Release lock from chain
      for (auto map : lockChains) {
        Value lockFromChain = map.second.first;
        int lockValueFromChain = map.second.second;
        Operation *relOp = map.first.first;
        if (relOp != Op)
          continue;

        rewriter.create<UseLockOp>(op.getLoc(), lockFromChain,
                                   lockValueFromChain, LockAction::Release);
        LLVM_DEBUG(llvm::dbgs() << "Release from chain " << lockFromChain
                                << " with " << lockValueFromChain << "\n");
      }
    }

    rewriter.eraseOp(Op);

    return success();
  }
};

static int getLockID(DenseMap<std::pair<Operation *, int>, int> &locks,
                     Operation *tileOp) {

  for (unsigned i = 0; i < 16; i++) {
    int usageCnt = locks[std::make_pair(tileOp, i)];
    if (usageCnt == 0) {
      locks[std::make_pair(tileOp, i)] = 1;
      return i;
    }
  }

  return -1;
}

struct AIECreateLocksPass : public AIECreateLocksBase<AIECreateLocksPass> {
  void runOnOperation() override {

    ModuleOp m = getOperation();
    OpBuilder builder = OpBuilder::atBlockEnd(m.getBody());

    TokenAnalysis TA(m);
    TA.runAnalysis();
    LLVM_DEBUG(TA.print(llvm::dbgs()));
    DenseMap<StringRef, SmallVector<Operation *, 4>> tokenAcqMap(
        TA.getTokenAcqMap());
    DenseMap<StringRef, SmallVector<Operation *, 4>> tokenRelMap(
        TA.getTokenRelMap());
    SmallVector<std::pair<Operation *, Operation *>, 4> tokenChains(
        TA.getTokenChains());
    SmallVector<std::pair<Operation *, Operation *>, 4> tokenPairs(
        TA.getTokenPairs());
    DenseMap<std::pair<int, int>, Operation *> tiles(TA.getTiles());

    DenseMap<std::pair<Operation *, int>, int> locks;
    DenseMap<std::pair<Operation *, Operation *>, std::pair<Value, int>>
        lockChains;
    DenseMap<Operation *, std::vector<std::pair<Value, int>>> acqLocks;
    DenseMap<Operation *, std::vector<std::pair<Value, int>>> relLocks;

    for (auto map : tokenChains) {
      Operation *release = map.first;
      Operation *acquire = map.second;

      Operation *relUser = TA.getTokenUserOp(release);
      Operation *acqUser = TA.getTokenUserOp(acquire);
      bool IsRelUserCore = isa<CoreOp>(relUser);
      bool IsAcqUserCore = isa<CoreOp>(acqUser);
      std::pair<int, int> relUserCoord = TA.getCoord(relUser);
      std::pair<int, int> acqUserCoord = TA.getCoord(acqUser);

      Operation *tileOp = TA.getShareableTileOp(relUser, acqUser);

      LLVM_DEBUG(llvm::dbgs() << "\n=== CHECKING TOKEN CHAIN ===\n";
                 release->print(llvm::dbgs());
                 if (IsRelUserCore) llvm::dbgs() << " @Core";
                 else llvm::dbgs() << " @DMA";
                 llvm::dbgs() << " (" << relUserCoord.first << ", "
                              << relUserCoord.second << ")" << '\n';
                 acquire->print(llvm::dbgs());
                 if (IsAcqUserCore) llvm::dbgs() << " @Core";
                 else llvm::dbgs() << " @DMA";
                 llvm::dbgs() << " (" << acqUserCoord.first << ", "
                              << acqUserCoord.second << ")" << '\n';);

      // ignore chain that involves a MemOp (DMA) user and CoreOp user and they
      // don't have a shareable tile. This might be caused by MemcpyOp lowering
      // -- there are two MemOps that use the same token and the same lock
      // action + value. Therefore, TokenAnalysis accidentally chains one MemOp
      // to a Core (from the MemcpyOp relationship) that does not have memory
      // affinity with it
      // TODO: verify if it is actually safe to ignore this case
      if (!tileOp && ((!IsRelUserCore && IsAcqUserCore) ||
                      (!IsAcqUserCore && IsRelUserCore)))
        continue;
      assert(tileOp &&
             "Sorry, the lock users of this chain do not have a common lock");

      TileOp tile = dyn_cast<TileOp>(tileOp);
      int lockID = getLockID(locks, tileOp);
      // int tileCol = tile.colIndex();
      // int tileRow = tile.rowIndex();

      // // use tile row, col and lockID to create a unique lock name
      // std::string name = "lock" + std::to_string(tileRow) + std::to_string(tileCol) + std::to_string(lockID);
      assert(lockID >= 0 && "No more locks to allocate!");
      LLVM_DEBUG(llvm::dbgs() << "Shared tile \n"; tileOp->print(llvm::dbgs()));
      LLVM_DEBUG(llvm::dbgs() << " LockID: " << lockID << '\n');

      builder.setInsertionPointAfter(tileOp);
      LockOp lock =
          builder.create<LockOp>(builder.getUnknownLoc(), tile, lockID);
      // %1 = aie.lock(%0, 15) {sym_name = "lock_15"}
      // lock->setAttr(SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));

      lockChains[std::make_pair(release, acquire)] = std::make_pair(lock, 1);

      for (auto pair : tokenPairs) {
        Operation *acqFromPair = pair.first;
        Operation *relFromPair = pair.second;

        if (relFromPair == release)
          acqLocks[acqFromPair].push_back(std::make_pair(lock, 0));

        if (acqFromPair == acquire)
          relLocks[relFromPair].push_back(std::make_pair(lock, 0));
      }
    }

    ConversionTarget target(getContext());
    target.addLegalOp<UseLockOp>();

    RewritePatternSet patterns(&getContext());
    patterns.insert<Token2LockLowering>(m.getContext(), m, acqLocks, relLocks,
                                        lockChains);

    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIE::createAIECreateLocksPass() {
  return std::make_unique<AIECreateLocksPass>();
}