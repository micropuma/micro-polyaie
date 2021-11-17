//===----------------------------------------------------------------------===//
//
// Copyright 2021-2021 The PolyAIE Authors.
//
//===----------------------------------------------------------------------===//

#include "polyaie/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace polyaie;

namespace {
#define GEN_PASS_REGISTRATION
#include "polyaie/Transforms/Passes.h.inc"
} // namespace

void polyaie::registerPolyAIEPassPipeline() {
  PassPipelineRegistration<PolyAIEPipelineOptions>(
      "polyaie-pipeline", "Compile to AIE array",
      [](OpPassManager &pm, const PolyAIEPipelineOptions &opts) {
        pm.addPass(polyaie::createPreprocessPass(opts));
        pm.addPass(mlir::createAffineLoopNormalizePass());
        pm.addPass(mlir::createSimplifyAffineStructuresPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(polyaie::createCreateDataflowPass());
        // pm.addPass(polyaie::createConvertToAIEPass());
      });
}

void polyaie::registerPolyAIEPasses() {
  registerPasses();
  registerPolyAIEPassPipeline();
}
