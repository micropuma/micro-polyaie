//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The PolyAIE Authors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/DialectConversion.h"
#include "polyaie/Transforms/Passes.h"
#include "polyaie/Utils.h"

using namespace mlir;
using namespace polyaie;
using namespace dataflow;

namespace {
struct ConvertToDataflow
    : public polyaie::ConvertToDataflowBase<ConvertToDataflow> {
  void runOnOperation() override;
};
} // namespace

// 将bufferization::ToTensorOp转变成dataflow::TensorLoadOp
namespace {
struct TensorLoadConversion
    : public OpConversionPattern<bufferization::ToTensorOp> {
  using OpConversionPattern<bufferization::ToTensorOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(bufferization::ToTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // 获取bufferization::ToTensorOp的memref
    // 并替换该memref的源subview
    auto mem = op.memref();
    if (auto subview = mem.getDefiningOp<memref::SubViewOp>()) {
      // 将bufferization::ToTensorOp变成dataflow::TensorLoadOp
      // 且dataflow::TensorLoadOp的memory layout 和subview一致
      // 替换掉源mem是subview.source，后续的canonicalize会将subview操作删除
      rewriter.replaceOpWithNewOp<dataflow::TensorLoadOp>(
          op, op.getType(), subview.static_offsets(), subview.static_sizes(),
          subview.static_strides(), subview.source());
      return success();

    } else if (auto cast = mem.getDefiningOp<memref::CastOp>()) {
      rewriter.replaceOpWithNewOp<dataflow::TensorLoadOp>(op, op.getType(),
                                                          cast.source());
      return success();

    } else if (mem.isa<BlockArgument>() ||
               isa<memref::AllocOp, memref::AllocaOp>(mem.getDefiningOp())) {
      // 如果没有经过subview操作，直接替换，十分naive
      rewriter.replaceOpWithNewOp<dataflow::TensorLoadOp>(op, op.getType(),
                                                          op.memref());
      return success();
    }
    return failure();
  }
};
} // namespace

// 将memref::CopyOp转变成dataflow::TensorStoreOp
namespace {
struct TensorStoreConversion : public OpConversionPattern<memref::CopyOp> {
  using OpConversionPattern<memref::CopyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // 一般情况下，memref::CopyOp的source是bufferization::ToMemrefOp
    // %7 = bufferization.to_memref %6 : memref<2x2xf32>
    // "dataflow.tensor_store"(%arg0, %6) {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : (memref<2x2xf32>, tensor<2x2xf32>) -> ()
    if (auto toMemrefOp =
            op.source().getDefiningOp<bufferization::ToMemrefOp>()) {
      auto mem = op.target();
      // 都是很统一的写法，获取subview的layout，然后改写
      if (auto subview = mem.getDefiningOp<memref::SubViewOp>()) {
        rewriter.replaceOpWithNewOp<dataflow::TensorStoreOp>(
            op, subview.static_offsets(), subview.static_sizes(),
            subview.static_strides(), subview.source(), toMemrefOp.tensor());
        return success();

      } else if (auto cast = mem.getDefiningOp<memref::CastOp>()) {
        rewriter.replaceOpWithNewOp<dataflow::TensorStoreOp>(
            op, cast.source(), toMemrefOp.tensor());
        return success();

      } else if (mem.isa<BlockArgument>() ||
                 isa<memref::AllocOp, memref::AllocaOp>(mem.getDefiningOp())) {
        // naive case
        rewriter.replaceOpWithNewOp<dataflow::TensorStoreOp>(
            op, mem, toMemrefOp.tensor());
        return success();
      }
    }
    return failure();
  }
};
} // namespace

// 将func::CallOp转变成dataflow::ProcessOp
// CallOp变成processOp，利用processOp的构造函数构造
// processOp做的body是CallOp的fun的body的inline化
// processOp的returnOp变成dataflow::ReturnOp
namespace {
struct ProcessConversion : public OpConversionPattern<func::CallOp> {
  using OpConversionPattern<func::CallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // 在mlir整个module的范畴搜寻func::FuncOp
    auto mod = op->getParentOfType<ModuleOp>();
    auto func = mod.lookupSymbol<func::FuncOp>(op.getCallee());

    // Replace call and function operation.
    // 默认没有提供kind参数的processOp是。processKind::AIE。
    auto process = rewriter.replaceOpWithNewOp<dataflow::ProcessOp>(
        op, op.getResultTypes(), op.getOperands());
    // process->setAttrs(func->getAttrs());

    // Inline the function body.
    // now is process op
    auto &bodyBlock = process.body().front();
    rewriter.inlineRegionBefore(func.getBody(), &bodyBlock);
    rewriter.eraseBlock(&bodyBlock);
    rewriter.eraseOp(func);

    // Replace return operation.
    // func::returnOp -> dataflow::returnOp
    auto returnOp = process.body().back().getTerminator();
    rewriter.setInsertionPoint(returnOp);
    rewriter.create<dataflow::ReturnOp>(returnOp->getLoc(),
                                        returnOp->getOperands());
    rewriter.eraseOp(returnOp);
    return success();
  }
};
} // namespace

// core pass implementation
void ConvertToDataflow::runOnOperation() {
  auto mod = getOperation();

  // 必须是顶层函数才有这个转化
  auto topFunc = getTopFunc<func::FuncOp>(mod);
  auto b = OpBuilder(mod);

  // 定义转换模版
  RewritePatternSet patterns(mod.getContext());
  ConversionTarget target(*mod.getContext());

  // 在dataflow中不支持的操作
  // func::CallOp要变成dataflow::ProcessOp
  // bufferization::ToTensorOp要变成dataflow::TensorLoadOp
  // memref::CopyOp要变成dataflow::TensorStoreOp
  // 显示转换，建立dataflow 模型，更接近AIE硬件特性
  target.addIllegalOp<func::CallOp>();
  target.addIllegalOp<bufferization::ToTensorOp>();
  target.addIllegalOp<memref::CopyOp>();

  // 添加转换模版，这三个转换模版使我们重点关注的对象
  patterns.add<TensorLoadConversion>(patterns.getContext());
  patterns.add<TensorStoreConversion>(patterns.getContext());
  patterns.add<ProcessConversion>(patterns.getContext());

  // 设定合法操作集合
  target.addLegalOp<dataflow::TensorLoadOp>();
  target.addLegalOp<dataflow::TensorStoreOp>();
  target.addLegalOp<dataflow::ProcessOp>();
  target.addLegalOp<dataflow::ReturnOp>();

  if (failed(applyPartialConversion(topFunc, target, std::move(patterns))))
    return signalPassFailure();

  // Mark leaf processes that drives tensor store ops.
  for (auto store : topFunc.getOps<dataflow::TensorStoreOp>()) {
    auto leafProcess = store.tensor().getDefiningOp<dataflow::ProcessOp>();
    leafProcess->setAttr("polyaie.leaf", b.getUnitAttr());
  }

  // Create a new dataflow function.
  // 整个top function的内容都被inline到了dataflow::FuncOp中
  // 单纯的给func::FuncOp套上dataflow::FuncOp的壳子
  b.setInsertionPoint(topFunc);
  SmallVector<NamedAttribute, 4> attrs;
  for (const auto &attr : topFunc->getAttrs()) {
    if (attr.getName() == SymbolTable::getSymbolAttrName() ||
        attr.getName() == function_interface_impl::getTypeAttrName())
      continue;
    attrs.push_back(attr);
  }
  auto dfFunc = b.create<dataflow::FuncOp>(topFunc.getLoc(), topFunc.getName(),
                                           topFunc.getFunctionType(), attrs);
  dfFunc.resolveArgAndResNames();

  // Inline the contents of the top function.
  auto &topFuncBlocks = topFunc.getBody().getBlocks();
  auto &dfFuncBlocks = dfFunc.body().getBlocks();
  dfFuncBlocks.splice(dfFuncBlocks.begin(), topFuncBlocks);

  // Replace the ternimator with an EndOp.
  auto returnOp = dfFuncBlocks.back().getTerminator();
  b.setInsertionPoint(returnOp);
  b.create<dataflow::ReturnOp>(returnOp->getLoc(), returnOp->getOperands());
  returnOp->erase();

  // We can erase the top function now.
  topFunc->erase();
}

std::unique_ptr<Pass> polyaie::createConvertToDataflowPass() {
  return std::make_unique<ConvertToDataflow>();
}
