//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The PolyAIE Authors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"
#include "mlir/Transforms/DialectConversion.h"
#include "polyaie/Transforms/Passes.h"
#include "polyaie/Utils.h"

using namespace mlir;
using namespace polyaie;

namespace {
struct TensorizeMemref : public polyaie::TensorizeMemrefBase<TensorizeMemref> {
  void runOnOperation() override;
};
} // namespace

namespace {
class MemrefTensorizeTypeConverter : public TypeConverter {
public:
  static Value materializeDefine(OpBuilder &b, MemRefType type,
                                 ValueRange inputs, Location loc) {
    assert(inputs.size() == 1);
    auto inputType = inputs[0].getType().dyn_cast<TensorType>();
    assert(inputType && inputType.getElementType() == type.getElementType() &&
           inputType.getShape() == type.getShape());
    return b.create<bufferization::ToMemrefOp>(loc, type, inputs[0]);
  }

  static Value materializeUse(OpBuilder &b, TensorType type, ValueRange inputs,
                              Location loc) {
    assert(inputs.size() == 1);
    auto inputType = inputs[0].getType().dyn_cast<MemRefType>();
    assert(inputType && inputType.getElementType() == type.getElementType() &&
           inputType.getShape() == type.getShape());
    return b.create<bufferization::ToTensorOp>(loc, type, inputs[0]);
  }

  MemrefTensorizeTypeConverter() {
    // Convert multi-elements memrefs to tensors.
    addConversion([](Type type) -> Type {
      auto memref = type.dyn_cast<MemRefType>();
      if (!memref || memref.getNumElements() == 1)
        return type;
      return RankedTensorType::get(memref.getShape(), memref.getElementType());
    });
    // Convert between memref and tensor.
    addArgumentMaterialization(materializeDefine);
    addSourceMaterialization(materializeDefine);
    addTargetMaterialization(materializeUse);
  }
};
} // namespace

void TensorizeMemref::runOnOperation() {
  auto mod = getOperation();
  auto b = OpBuilder(mod);
  auto topFunc = getTopFunc<FuncOp>(mod);

  // Tensorize memref types.
  MemrefTensorizeTypeConverter tensorizeConverter;
  RewritePatternSet patterns(mod.getContext());
  ConversionTarget target(*mod.getContext());

  populateFuncOpTypeConversionPattern(patterns, tensorizeConverter);
  target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
    return (tensorizeConverter.isSignatureLegal(op.getType()) &&
            tensorizeConverter.isLegal(&op.getBody())) ||
           op == topFunc;
  });
  populateCallOpTypeConversionPattern(patterns, tensorizeConverter);
  target.addDynamicallyLegalOp<CallOp>(
      [&](CallOp op) { return tensorizeConverter.isLegal(op); });
  populateReturnOpTypeConversionPattern(patterns, tensorizeConverter);

  target.markUnknownOpDynamicallyLegal([&](Operation *op) {
    return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
           isLegalForReturnOpTypeConversionPattern(op, tensorizeConverter);
  });

  if (failed(applyFullConversion(mod, target, std::move(patterns))))
    return signalPassFailure();

  // Remove the layout information from all AllocOp.
  mod.walk([&](memref::AllocOp alloc) {
    auto type = alloc.getType();
    auto newType = MemRefType::get(type.getShape(), type.getElementType());

    b.setInsertionPoint(alloc);
    auto newAlloc = b.create<memref::AllocOp>(alloc.getLoc(), newType);
    alloc.replaceAllUsesWith(newAlloc.memref());
    alloc.erase();
  });

  mod.walk([&](mlir::ReturnOp returnOp) {
    for (auto operand : returnOp.getOperands())
      if (auto toTensorOp = operand.getDefiningOp<bufferization::ToTensorOp>())
        toTensorOp->moveBefore(returnOp);
  });
}

std::unique_ptr<Pass> polyaie::createTensorizeMemrefPass() {
  return std::make_unique<TensorizeMemref>();
}
