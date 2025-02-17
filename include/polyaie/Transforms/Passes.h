//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The PolyAIE Authors.
//
//===----------------------------------------------------------------------===//

#ifndef POLYAIE_TRANSFORMS_PASSES_H
#define POLYAIE_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "polyaie/InitAllDialects.h"
#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace mlir {
namespace polyaie {

struct PolyAIEOptions : public PassPipelineOptions<PolyAIEOptions> {
  // 一些pass的配置
  // Configure the preprocess pass.
  Option<std::string> preprocessTopFuncName{
      *this, "top-func-name", llvm::cl::init("main"),
      llvm::cl::desc("Specify the top function of the program")};

  // Configure the preprocess pass.
  // 设置对loop做tile化的参数
  Option<int64_t> loopTileSize{
      *this, "tile-size", llvm::cl::init(32),
      llvm::cl::desc("Specify the size of loop tiling")};

  // Configure the split-top-func pass.
  // 将top function拆分为多个function，构建dataflow模型
  Option<int64_t> splitTopFuncNumAIE{
      *this, "num-aie", llvm::cl::init(256),
      llvm::cl::desc("Specify the available number of AIEs")};

  // Configure the memref-arg-to-result pass.
  Option<bool> memrefArgToResultReturnAllArg{
      *this, "return-all-arg", llvm::cl::init(false),
      llvm::cl::desc("Return all arguments as results")};

  // Configure the affine-super-vectorize pass.
  Option<int64_t> superVectorizeSize{
      *this, "vec-size", llvm::cl::init(1),
      llvm::cl::desc("Specify the size of super vectorization")};

  // Configure the placement pass.
  // aie tile的放置pass
  Option<std::string> placementAlgorithm{
      *this, "algorithm", llvm::cl::init("naive"),
      llvm::cl::desc("Specify the placement algorithm, possible values are: "
                     "naive(default), simulated-annealing")};

  // Configure the create-interface pass.
  // 是否和extern mem或是pl交互
  Option<bool> enableCreateInterface{
      *this, "enable-create-interface", llvm::cl::init(true),
      llvm::cl::desc("Enable to generate interface processes")};

  // Configure the link-extern-kernel pass.
  // 将function body替换为extern kernel，仿照linalg的设计
  Option<bool> enableLinkExternKernel{
      *this, "enable-link-extern-kernel", llvm::cl::init(false),
      llvm::cl::desc("Substitute each function body with an external kernel")};

  // 如果有外部已经编译好的kernel，可以直接链接
  Option<std::string> linkExternKernelObjectFile{
      *this, "object-file", llvm::cl::init("kernel.o"),
      llvm::cl::desc("Specify the path of the pre-compiled object file of "
                     "external kernels")};

  // 利用AIEVec 方言生成外部kernel
  Option<bool> linkExternKernelGenExternKernel{
      *this, "gen-extern-kernel", llvm::cl::init(true),
      llvm::cl::desc(
          "Whether to generate external kernels through AIEVec dialect")};
};

std::unique_ptr<Pass> createPreprocessPass();
std::unique_ptr<Pass> createPreprocessPass(const PolyAIEOptions &opts);
std::unique_ptr<Pass> createSplitTopFuncPass();
std::unique_ptr<Pass> createSplitTopFuncPass(const PolyAIEOptions &opts);

std::unique_ptr<Pass> createCreateMemrefSubviewPass();
std::unique_ptr<Pass> createHoistMemrefSubviewPass();
std::unique_ptr<Pass> createMemrefArgToResultPass();
std::unique_ptr<Pass> createMemrefArgToResultPass(const PolyAIEOptions &opts);
std::unique_ptr<Pass> createExtractMemrefDependencyPass();
std::unique_ptr<Pass> createBufferMemrefResultPass();
std::unique_ptr<Pass> createTensorizeMemrefPass();
std::unique_ptr<Pass> createDetectLoopReductionPass();

std::unique_ptr<Pass> createConvertToDataflowPass();
std::unique_ptr<Pass> createInsertGMIOAdapterPass();
std::unique_ptr<Pass> createPlacementPass();
std::unique_ptr<Pass> createPlacementPass(const PolyAIEOptions &opts);
std::unique_ptr<Pass> createCreateInterfacePass();
std::unique_ptr<Pass> createPrintDataflowPass();

// 先注释AIE部分，后续再添加
// std::unique_ptr<Pass> createDataflowToAIEPass();
// std::unique_ptr<Pass> createLinkExternKernelPass();
// std::unique_ptr<Pass> createLinkExternKernelPass(const PolyAIEOptions &opts);
// std::unique_ptr<Pass> createMaterializeBroadcastPass();
// std::unique_ptr<Pass> createFlowPacketToCircuitPass();
// std::unique_ptr<Pass> createDoubleBufferPass();
// std::unique_ptr<Pass> createPostprocessPass();
// std::unique_ptr<Pass> createCodeGenCleanupPass();

void registerPolyAIEPassPipeline();
void registerPolyAIEPasses();

#define GEN_PASS_CLASSES
#include "polyaie/Transforms/Passes.h.inc"

} // namespace polyaie
} // namespace mlir

#endif // POLYAIE_TRANSFORMS_PASSES_H
