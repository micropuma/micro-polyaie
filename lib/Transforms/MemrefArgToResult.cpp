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
struct MemrefArgToResult
    : public polyaie::MemrefArgToResultBase<MemrefArgToResult> {
  MemrefArgToResult() = default;
  MemrefArgToResult(const PolyAIEOptions &opts) {
    returnAllArg = opts.memrefArgToResultReturnAllArg;
  }

  void runOnOperation() override;
};
} // namespace

void MemrefArgToResult::runOnOperation() {
  auto mod = getOperation();
  auto b = OpBuilder(mod);
  auto loc = b.getUnknownLoc();
  auto topFunc = getTopFunc<FuncOp>(mod);

  for (auto call : llvm::make_early_inc_range(topFunc.getOps<CallOp>())) {
    auto func = mod.lookupSymbol<FuncOp>(call.getCallee());

    // mlir 默认每个函数都有一个返回值，所以这里不需要处理没有返回值的函数
    auto returnOp = func.back().getTerminator();
    // 初始的returnVals包含已有return返回值，同时后续会添加memref类型的返回值
    SmallVector<Value, 8> returnVals(returnOp->getOperands());
    SmallVector<Value, 8> resultMems;
    llvm::SmallDenseSet<Value, 8> stateChangedMems;

    // Figure out all arguments that need to be returned.
    // TODO: Currently we return all memref arguments, how to determine whether
    // an argument need to be returned?
    for (auto arg : func.getArguments()) {
      // Get the argument type and bypass non-memref arguments.
      auto argType = arg.getType().dyn_cast<MemRefType>();
      if (!argType)
        continue;

      auto memory = call.getOperand(arg.getArgNumber());

      if (llvm::any_of(arg.getUsers(), [&](Operation *op) {
            return isa<mlir::AffineStoreOp>(op);
          })) {
        returnVals.push_back(arg);
        resultMems.push_back(memory);
        stateChangedMems.insert(memory);
      } else if (returnAllArg) {
        returnVals.push_back(arg);
        resultMems.push_back(memory);
      }
    }

    // Update return operation and function signature.
    b.setInsertionPoint(returnOp);
    auto newReturn = b.create<ReturnOp>(returnOp->getLoc(), returnVals);
    returnOp->erase();
    func.setType(b.getFunctionType(func.getArgumentTypes(),
                                   newReturn.getOperandTypes()));

    // Update function call.
    b.setInsertionPointAfter(call);
    auto newCall = b.create<CallOp>(call.getLoc(), func, call.getOperands());
    auto numResults = call.getNumResults();
    // call 中原来是用旧的返回值，现在用新的返回值替换
    // 注意，返回值分为两类：原本已有和在func中经过affine::StoreOp操作的memref类型的返回值
    call.replaceAllUsesWith(newCall.getResults().take_front(numResults));
    call.erase();

    // Create copy operation if the state of memory is changed.
    // 利用zip的特性去除掉原本已经有的返回值，只处理memref类型的返回值
    for (auto zip :
         llvm::zip(newCall.getResults().drop_front(numResults), resultMems))
      if (stateChangedMems.count(std::get<1>(zip)))
        b.create<memref::CopyOp>(loc, std::get<0>(zip), std::get<1>(zip));
  }
}

std::unique_ptr<Pass> polyaie::createMemrefArgToResultPass() {
  return std::make_unique<MemrefArgToResult>();
}
std::unique_ptr<Pass>
polyaie::createMemrefArgToResultPass(const PolyAIEOptions &opts) {
  return std::make_unique<MemrefArgToResult>(opts);
}
