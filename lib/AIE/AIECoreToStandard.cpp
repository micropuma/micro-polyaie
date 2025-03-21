//===- AIECoreToStandard.cpp ------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "polyaie/AIE/AIEDialect.h"
#include "polyaie/AIE/AIENetlistAnalysis.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::vector;
using namespace xilinx;
using namespace xilinx::AIE;
// using namespace mlir::LLVM;

template <typename MyAIEOp>
struct AIEOpRemoval : public OpConversionPattern<MyAIEOp> {
  using OpConversionPattern<MyAIEOp>::OpConversionPattern;
  using OpAdaptor = typename MyAIEOp::Adaptor;
  ModuleOp &module;

  AIEOpRemoval(MLIRContext *context, ModuleOp &m, PatternBenefit benefit = 1)
      : OpConversionPattern<MyAIEOp>(context, benefit), module(m) {}

  LogicalResult matchAndRewrite(MyAIEOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();

    rewriter.eraseOp(Op);
    return success();
  }
};

struct AIEDebugOpToStdLowering : public OpConversionPattern<DebugOp> {
  using OpConversionPattern<DebugOp>::OpConversionPattern;
  ModuleOp &module;

  AIEDebugOpToStdLowering(MLIRContext *context, ModuleOp &m,
                          PatternBenefit benefit = 1)
      : OpConversionPattern<DebugOp>(context, benefit), module(m) {}

  LogicalResult matchAndRewrite(DebugOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();

    std::string funcName = "debug_i32";
    auto func = module.lookupSymbol<func::FuncOp>(funcName);
    assert(func && "Could not find the intrinsic function!");
    SmallVector<Value, 1> args;
    args.push_back(op.arg());
    rewriter.create<func::CallOp>(rewriter.getUnknownLoc(), func, args);
    rewriter.eraseOp(Op);
    return success();
  }
};

struct AIEPutStreamToStdLowering : public OpConversionPattern<PutStreamOp> {
  using OpConversionPattern<PutStreamOp>::OpConversionPattern;
  ModuleOp &module;

  AIEPutStreamToStdLowering(MLIRContext *context, ModuleOp &m,
                            PatternBenefit benefit = 1)
      : OpConversionPattern<PutStreamOp>(context, benefit), module(m) {}

  LogicalResult matchAndRewrite(PutStreamOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();

    std::string funcName = "llvm.aie.put.";
    if (op.isWideStream())
      funcName += "wms";
    else if (op.isFloatStream())
      funcName += "fms";
    else
      funcName += "ms";

    llvm::dbgs() << "FINDING: " << funcName << "\n";
    auto putMSFunc = module.lookupSymbol<func::FuncOp>(funcName);
    assert(putMSFunc && "Could not find the intrinsic function!");
    SmallVector<Value, 2> args;
    args.push_back(op.channel());
    args.push_back(op.streamValue());
    rewriter.create<func::CallOp>(rewriter.getUnknownLoc(), putMSFunc, args);
    rewriter.eraseOp(Op);
    return success();
  }
};

struct AIEGetStreamToStdLowering : public OpConversionPattern<GetStreamOp> {
  using OpConversionPattern<GetStreamOp>::OpConversionPattern;
  ModuleOp &module;

  AIEGetStreamToStdLowering(MLIRContext *context, ModuleOp &m,
                            PatternBenefit benefit = 1)
      : OpConversionPattern<GetStreamOp>(context, benefit), module(m) {}

  LogicalResult matchAndRewrite(GetStreamOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    std::string funcName = "llvm.aie.get.";
    if (op.isWideStream())
      funcName += "wss";
    else if (op.isFloatStream())
      funcName += "fss";
    else
      funcName += "ss";

    auto getSSFunc = module.lookupSymbol<func::FuncOp>(funcName);
    assert(getSSFunc && "Could not find the intrinsic function!");
    SmallVector<Value, 2> args;
    args.push_back(op.channel());
    auto getSSCall = rewriter.create<func::CallOp>(rewriter.getUnknownLoc(),
                                                   getSSFunc, args);
    rewriter.replaceOp(op, getSSCall.getResult(0));
    return success();
  }
};

struct AIEPutCascadeToStdLowering : public OpConversionPattern<PutCascadeOp> {
  using OpConversionPattern<PutCascadeOp>::OpConversionPattern;
  ModuleOp &module;

  AIEPutCascadeToStdLowering(MLIRContext *context, ModuleOp &m,
                             PatternBenefit benefit = 1)
      : OpConversionPattern<PutCascadeOp>(context, benefit), module(m) {}

  LogicalResult matchAndRewrite(PutCascadeOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();

    std::string funcName = "llvm.aie.put.mcd";
    auto putMCDFunc = module.lookupSymbol<func::FuncOp>(funcName);
    assert(putMCDFunc && "Could not find the intrinsic function!");
    SmallVector<Value, 2> args;
    args.push_back(op.cascadeValue());
    rewriter.create<func::CallOp>(rewriter.getUnknownLoc(), putMCDFunc, args);
    rewriter.eraseOp(Op);
    return success();
  }
};

struct AIEGetCascadeToStdLowering : public OpConversionPattern<GetCascadeOp> {
  using OpConversionPattern<GetCascadeOp>::OpConversionPattern;
  ModuleOp &module;

  AIEGetCascadeToStdLowering(MLIRContext *context, ModuleOp &m,
                             PatternBenefit benefit = 1)
      : OpConversionPattern<GetCascadeOp>(context, benefit), module(m) {}

  LogicalResult matchAndRewrite(GetCascadeOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    std::string funcName = "llvm.aie.get.scd";
    auto getSCDFunc = module.lookupSymbol<func::FuncOp>(funcName);
    assert(getSCDFunc && "Could not find the intrinsic function!");
    auto getSCDCall = rewriter.create<func::CallOp>(rewriter.getUnknownLoc(),
                                                    getSCDFunc, ValueRange({}));
    rewriter.replaceOp(op, getSCDCall.getResult(0));
    return success();
  }
};

struct AIECoreToStandardFunc : public OpConversionPattern<CoreOp> {
  using OpConversionPattern<CoreOp>::OpConversionPattern;
  ModuleOp &module;
  BlockAndValueMapping &mapper;
  DenseMap<Operation *, SmallVector<BufferOp, 4>> &tileToBuffers;
  int tileCol = 0;
  int tileRow = 0;

  AIECoreToStandardFunc(
      MLIRContext *context, ModuleOp &m, BlockAndValueMapping &mapper,
      DenseMap<Operation *, SmallVector<BufferOp, 4>> &tileToBuffers,
      PatternBenefit benefit = 1, int tileCol = 1, int tileRow = 1)
      : OpConversionPattern<CoreOp>(context, benefit), module(m),
        mapper(mapper), tileToBuffers(tileToBuffers), tileCol(tileCol),
        tileRow(tileRow) {}

  LogicalResult matchAndRewrite(CoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {

    Operation *Op = op.getOperation();
    int col = op.colIndex();
    int row = op.rowIndex();

    // Only pull code for the indicated function
    if ((tileRow != row) || (tileCol != col)) {
      rewriter.eraseOp(Op);
      return success();
    }

    std::string coreName("core" + std::to_string(col) + std::to_string(row));
    auto coreFunc = rewriter.create<func::FuncOp>(
        rewriter.getUnknownLoc(), coreName,
        FunctionType::get(rewriter.getContext(), {}, {}));

    rewriter.cloneRegionBefore(op.body(), coreFunc.getBody(),
                               coreFunc.getBody().begin(), mapper);

    // Create a main function that just calls the core function above.
    auto mainFunc = rewriter.create<func::FuncOp>(
        rewriter.getUnknownLoc(), "_main",
        FunctionType::get(rewriter.getContext(), {}, {}));
    rewriter.setInsertionPointToStart(mainFunc.addEntryBlock());
    SmallVector<Value, 8> args;
    rewriter.create<func::CallOp>(rewriter.getUnknownLoc(), coreFunc,
                                  args); // call with no args.
    rewriter.create<func::ReturnOp>(rewriter.getUnknownLoc(),
                                    args); // return nothing

    DenseMap<Operation *, Value> newAllocated;

    for (auto map : tileToBuffers) {
      Operation *tileOp = map.first;
      SmallVector<BufferOp, 4> buffers(map.second);
      TileOp tile = dyn_cast<TileOp>(tileOp);
      int dstCol = tile.colIndex();
      int dstRow = tile.rowIndex();

      if (!isLegalMemAffinity(col, row, dstCol, dstRow))
        continue;

      rewriter.setInsertionPoint(coreFunc);
      for (auto buffer : buffers) {
        auto symName = buffer.name().getValue();
        rewriter.create<memref::GlobalOp>(rewriter.getUnknownLoc(), symName, rewriter.getStringAttr("public"),
                                          buffer.getType(), nullptr, false, nullptr);
      }
      rewriter.setInsertionPointToStart(&coreFunc.getBody().front());
      for (auto buffer : buffers) {
        MemRefType t = buffer.getType().cast<MemRefType>();
        auto symName = buffer.name().getValue();
        auto allocated = rewriter.create<memref::GetGlobalOp>(
            rewriter.getUnknownLoc(), t, symName);
        newAllocated[buffer] = allocated.result();
        // Assume that buffers are aligned so they can be vectorized.
        rewriter.create<memref::AssumeAlignmentOp>(rewriter.getUnknownLoc(),
                                                   allocated, 32);
        rewriter.replaceOp(buffer, allocated.result());
      }
    }

    coreFunc.getBody().walk([&](Operation *childOp) {
      rewriter.setInsertionPointAfter(childOp);

      if (EndOp end = dyn_cast<EndOp>(childOp)) {
        rewriter.create<func::ReturnOp>(rewriter.getUnknownLoc(),
                                        ValueRange({}));
        rewriter.eraseOp(childOp);
      } else if (UseLockOp useLock = dyn_cast<UseLockOp>(childOp)) {
        std::string funcName = "llvm.aie.lock.";
        if (useLock.acquire())
          funcName += "acquire.reg";
        else if (useLock.release())
          funcName += "release.reg";

        auto useLockFunc = module.lookupSymbol<func::FuncOp>(funcName);
        assert(useLockFunc && "Could not find the intrinsic function!");

        SmallVector<Value, 2> args;

        args.push_back(rewriter.create<arith::IndexCastOp>(
            useLock.getLoc(), IntegerType::get(rewriter.getContext(), 32),
            useLock.lock()));
        args.push_back(rewriter.create<arith::ConstantOp>(
            useLock.getLoc(), IntegerType::get(rewriter.getContext(), 32),
            rewriter.getI32IntegerAttr(useLock.getLockValue())));

        rewriter.create<func::CallOp>(rewriter.getUnknownLoc(), useLockFunc,
                                      args);
        rewriter.eraseOp(childOp);
      }
    });

    rewriter.eraseOp(Op);
    return success();
  }
};

struct AIECoreToStandardPass
    : public AIECoreToStandardBase<AIECoreToStandardPass> {
  void runOnOperation() override {

    ModuleOp m = getOperation();
    OpBuilder builder = OpBuilder::atBlockEnd(m.getBody());

    // Ensure that we don't have an incorrect target triple.  This may override
    // some bogus target triple in the original mlir.  In reality this should
    // pick the 'aie' target triple.
    m->setAttr(LLVM::LLVMDialect::getTargetTripleAttrName(),
               builder.getStringAttr(""));

    // Extract all CoreOps
    // Create an LLVM func for each CoreOp
    // Clone the region body of each CoreOp to the newly created LLVM func

    DenseMap<std::pair<int, int>, Operation *> tiles;
    DenseMap<Operation *, CoreOp> cores;
    DenseMap<Operation *, MemOp> mems;
    DenseMap<std::pair<Operation *, int>, LockOp> locks;
    DenseMap<Operation *, SmallVector<BufferOp, 4>> tileToBuffers;
    DenseMap<Operation *, SwitchboxOp> switchboxes;

    NetlistAnalysis NL(m, tiles, cores, mems, locks, tileToBuffers,
                       switchboxes);
    NL.collectTiles(tiles);
    NL.collectCores(cores);
    NL.collectBuffers(tileToBuffers);

    // Populate intrinsic functions
    // Intrinsic information:
    // peano/llvm-project/llvm/lib/Target/AIE/AIEInstrInfo.td Also take a look
    // at the tests: peano/llvm-project/llvm/test/CodeGen/AIE
    builder.setInsertionPointToStart(m.getBody());

    SmallVector<Type, 2> callArgTypes;

    Type int32Type = IntegerType::get(builder.getContext(), 32);
    Type int128Type = IntegerType::get(builder.getContext(), 128);
    Type int384Type = IntegerType::get(builder.getContext(), 384);
    Type floatType = FloatType::getF32(builder.getContext());

    // llvm.func @debug_i32(%val: !llvm.i32) -> ()
    builder
        .create<func::FuncOp>(
            builder.getUnknownLoc(), "debug_i32",
            FunctionType::get(builder.getContext(), {int32Type}, {}))
        .setPrivate();

    // llvm.func @llvm.aie.put.ms(%channel: !llvm.i1, %stream_val: !llvm.i32) ->
    // ()
    builder
        .create<func::FuncOp>(
            builder.getUnknownLoc(), "llvm.aie.put.ms",
            FunctionType::get(builder.getContext(), {int32Type, int32Type}, {}))
        .setPrivate();

    // llvm.func @llvm.aie.put.mws(%channel: !llvm.i1, %stream_val: !llvm.i128)
    // -> ()
    builder
        .create<func::FuncOp>(builder.getUnknownLoc(), "llvm.aie.put.wms",
                              FunctionType::get(builder.getContext(),
                                                {int32Type, int128Type}, {}))
        .setPrivate();

    // llvm.func @llvm.aie.put.mfs(%channel: !llvm.i1, %stream_val: !llvm.float)
    // -> ()
    builder
        .create<func::FuncOp>(
            builder.getUnknownLoc(), "llvm.aie.put.fms",
            FunctionType::get(builder.getContext(), {int32Type, floatType}, {}))
        .setPrivate();

    // llvm.func @llvm.aie.get.ss(%channel: !llvm.i1) -> !llvm.i32
    builder
        .create<func::FuncOp>(
            builder.getUnknownLoc(), "llvm.aie.get.ss",
            FunctionType::get(builder.getContext(), {int32Type}, {int32Type}))
        .setPrivate();

    // llvm.func @llvm.aie.get.wss(%channel: !llvm.i1) -> !llvm.i128
    builder
        .create<func::FuncOp>(
            builder.getUnknownLoc(), "llvm.aie.get.wss",
            FunctionType::get(builder.getContext(), {int32Type}, {int128Type}))
        .setPrivate();

    // llvm.func @llvm.aie.get.fss(%channel: !llvm.i1) -> !llvm.float
    builder
        .create<func::FuncOp>(
            builder.getUnknownLoc(), "llvm.aie.get.fss",
            FunctionType::get(builder.getContext(), {int32Type}, {floatType}))
        .setPrivate();

    // llvm.func @llvm.aie.put.scd(%scd_val: !llvm.i384) -> ()
    builder
        .create<func::FuncOp>(
            builder.getUnknownLoc(), "llvm.aie.put.mcd",
            FunctionType::get(builder.getContext(), {int384Type}, {}))
        .setPrivate();

    // llvm.func @llvm.aie.get.scd() -> !llvm.i384
    builder
        .create<func::FuncOp>(
            builder.getUnknownLoc(), "llvm.aie.get.scd",
            FunctionType::get(builder.getContext(), {}, {int384Type}))
        .setPrivate();

    // llvm.func @llvm.aie.lock.acquire.reg(%lock_id: !llvm.i32, %lock_val:
    // !llvm.i32) ->()
    builder
        .create<func::FuncOp>(
            builder.getUnknownLoc(), "llvm.aie.lock.acquire.reg",
            FunctionType::get(builder.getContext(), {int32Type, int32Type}, {}))
        .setPrivate();

    // llvm.func @llvm.aie.lock.release.reg(%lock_id: !llvm.i32, %lock_val:
    // !llvm.i32) ->()
    builder
        .create<func::FuncOp>(
            builder.getUnknownLoc(), "llvm.aie.lock.release.reg",
            FunctionType::get(builder.getContext(), {int32Type, int32Type}, {}))
        .setPrivate();

    BlockAndValueMapping mapper;
    ConversionTarget target(getContext());
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<cf::ControlFlowDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<VectorDialect>();
    target.addLegalDialect<arith::ArithmeticDialect>();
    target.addLegalOp<func::FuncOp, ModuleOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<AIEPutStreamToStdLowering,
                 AIEGetStreamToStdLowering,
                 AIEPutCascadeToStdLowering,
                 AIEGetCascadeToStdLowering,
                 AIEDebugOpToStdLowering
                >(m.getContext(), m);

    patterns.add<AIECoreToStandardFunc>(m.getContext(), m, mapper, tileToBuffers, 1,
        tileCol, tileRow);
    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();

    RewritePatternSet removepatterns(&getContext());
    removepatterns.add<AIEOpRemoval<AIE::TileOp>,
                       AIEOpRemoval<AIE::FlowOp>,
                       AIEOpRemoval<AIE::MemOp>,
                       AIEOpRemoval<AIE::ShimDMAOp>,
                       AIEOpRemoval<AIE::ShimMuxOp>,
                       AIEOpRemoval<AIE::SwitchboxOp>,
                       AIEOpRemoval<AIE::LockOp>,
                       AIEOpRemoval<AIE::BufferOp>,
                       AIEOpRemoval<AIE::ExternalBufferOp>
                      >(m.getContext(), m);

    if (failed(applyPartialConversion(m, target, std::move(removepatterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIE::createAIECoreToStandardPass() {
  return std::make_unique<AIECoreToStandardPass>();
}
