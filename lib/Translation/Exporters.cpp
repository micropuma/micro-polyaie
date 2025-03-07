//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The PolyAIE Authors.
//
//===----------------------------------------------------------------------===//

#include "polyaie/Exporters.h"

// 
#include "../lib/AIE/Targets/AIEVecToCpp/TranslateAIEVecToCpp.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "polyaie/InitAllDialects.h"

using namespace mlir;
using namespace polyaie;
using namespace func;

LogicalResult polyaie::exportAIEKernel(ModuleOp module, raw_ostream &os) {
  for (auto func : module.getOps<FuncOp>()) {
    os << R"XXX(
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for MLIR-AIE AIE kernel.
//
//===----------------------------------------------------------------------===//

// #define __AIENGINE__ 1
// #define NOCPP

// #include <cardano.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {

)XXX";

    if (failed(xilinx::aievec::translateAIEVecToCpp(func, os)))
      return failure();

    os << "}\n";
  }

  return success();
}

void polyaie::registerExportHostKernel() {
  static TranslateFromMLIRRegistration registration(
      "export-host-kernel", exportHostKernel, [&](DialectRegistry &registry) {
        polyaie::registerAllDialects(registry);
      });
}

void polyaie::registerExportAIEKernel() {
  static TranslateFromMLIRRegistration registration(
      "export-aie-kernel", exportAIEKernel, [&](DialectRegistry &registry) {
        polyaie::registerAllDialects(registry);
      });
}

void polyaie::registerPolyAIEExporters() {
  registerExportHostKernel();
  registerExportAIEKernel();
}
