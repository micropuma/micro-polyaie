//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The PolyAIE Authors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "polyaie/InitAllDialects.h"
#include "polyaie/InitAllPasses.h"

int main(int argc, char **argv) {
  // 注册所有的dialect和pass
  // 在注册pass的时候，注册所有的pipeline以及可选项
  mlir::DialectRegistry registry;
  mlir::polyaie::registerAllDialects(registry);
  mlir::polyaie::registerAllPasses();

  return mlir::failed(mlir::MlirOptMain(argc, argv, "PolyAIE Optimization Tool",
                                        registry, true));
}
