//===- AIEDialect.h ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_AIE_DIALECT_H
#define MLIR_AIE_DIALECT_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Debug.h"
#include <map>

using namespace mlir;

#include "polyaie/AIE/AIEEnums.h.inc"

namespace mlir {
namespace OpTrait {
template <typename ConcreteType>
class FlowEndPoint : public OpTrait::TraitBase<ConcreteType, FlowEndPoint> {};
} // namespace OpTrait
} // namespace mlir

/// Include the generated interface declarations.
#include "polyaie/AIE/AIEInterfaces.h.inc"

namespace xilinx {
namespace AIE {

void registerAIETranslations();

// FIXME: use this
//#include "AIEDialect.h.inc"

// The Dialect
class AIEDialect : public mlir::Dialect {
public:
  explicit AIEDialect(mlir::MLIRContext *ctx);
  static StringRef getDialectNamespace() { return "aie"; }

  // /// Parse a type registered to this dialect. Overridding this method is
  // /// required for dialects that have custom types.
  // /// Technically this is only needed to be able to round-trip to textual IR.
  // mlir::Type parseType(DialectAsmParser &parser) const override;

  // /// Print a type registered to this dialect. Overridding this method is
  // /// only required for dialects that have custom types.
  // /// Technically this is only needed to be able to round-trip to textual IR.
  // void printType(mlir::Type type, DialectAsmPrinter &os) const override;
};

////////////////////////////////////////////////////////////////////////////////
/////////////////////// Custom Types for the Dialect ///////////////////////////
////////////////////////////////////////////////////////////////////////////////

// namespace detail {
// struct AIEListTypeStorage;
// }

// /// LLVM-style RTTI: one entry per subclass to allow dyn_cast/isa.
// enum AIETypeKind {
//   // The enum starts at the range reserved for this dialect.
//   AIE_TYPE = mlir::Type::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
//   AIE_LIST,
// };

// /// Type for Toy arrays.
// /// In MLIR Types are reference to immutable and uniqued objects owned by the
// /// MLIRContext. As such `AIEListType` only wraps a pointer to an uniqued
// /// instance of `AIEListTypeStorage` (defined in our implementation file) and
// /// provides the public facade API to interact with the type.
// class AIEListType : public mlir::Type::TypeBase<AIEListType, mlir::Type,
//                                                  detail::AIEListTypeStorage>
//                                                  {
// public:
//   using Base::Base;

//   /// Return the type of individual elements in the array.
//   mlir::Type getElementType();

//   /// Get the unique instance of this Type from the context.
//   static AIEListType get(mlir::Type elementType);

//   /// Support method to enable LLVM-style RTTI type casting.
//   static bool kindof(unsigned kind) { return kind == AIETypeKind::AIE_LIST; }
// };

////////////////////////////////////////////////////////////////////////////////
//////////////////// Custom Operations for the Dialect /////////////////////////
////////////////////////////////////////////////////////////////////////////////

//#include "AIEOpInterfaces.h.inc"

typedef std::pair<WireBundle, int> Port;
typedef std::pair<Port, Port> Connect;
typedef std::pair<int, int> TileID;

bool isValidTile(TileID src);

// Return the tile ID of the memory to the west of the given tile, if it exists.
mlir::Optional<TileID> getMemWest(TileID src);
// Return the tile ID of the memory to the east of the given tile, if it exists.
mlir::Optional<TileID> getMemEast(TileID src);
// Return the tile ID of the memory to the north of the given tile, if it
// exists.
mlir::Optional<TileID> getMemNorth(TileID src);
// Return the tile ID of the memory to the south of the given tile, if it
// exists.
mlir::Optional<TileID> getMemSouth(TileID src);

bool isInternal(int srcCol, int srcRow, int dstCol, int dstRow);
bool isWest(int srcCol, int srcRow, int dstCol, int dstRow);
bool isMemWest(int srcCol, int srcRow, int dstCol, int dstRow);
bool isEast(int srcCol, int srcRow, int dstCol, int dstRow);

bool isMemEast(int srcCol, int srcRow, int dstCol, int dstRow);
bool isNorth(int srcCol, int srcRow, int dstCol, int dstRow);
bool isMemNorth(int srcCol, int srcRow, int dstCol, int dstRow);
bool isSouth(int srcCol, int srcRow, int dstCol, int dstRow);
bool isMemSouth(int srcCol, int srcRow, int dstCol, int dstRow);

bool isLegalMemAffinity(int coreCol, int coreRow, int memCol, int memRow);
} // namespace AIE
} // namespace xilinx

// include TableGen generated Op definitions
#define GET_OP_CLASSES
#include "polyaie/AIE/AIE.h.inc"

namespace xilinx {
namespace AIE {

#define GEN_PASS_CLASSES
#include "polyaie/AIE/AIEPasses.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createAIEAssignBufferAddressesPass();
std::unique_ptr<OperationPass<ModuleOp>> createAIECoreToStandardPass();
std::unique_ptr<OperationPass<ModuleOp>> createAIECreateCoresPass();
std::unique_ptr<OperationPass<ModuleOp>> createAIECreateLocksPass();
std::unique_ptr<OperationPass<ModuleOp>> createAIEFindFlowsPass();
std::unique_ptr<OperationPass<ModuleOp>> createAIEHerdRoutingPass();
std::unique_ptr<OperationPass<ModuleOp>> createAIELocalizeLocksPass();
std::unique_ptr<OperationPass<ModuleOp>> createAIELowerMemcpyPass();
std::unique_ptr<OperationPass<ModuleOp>> createAIENormalizeAddressSpacesPass();
std::unique_ptr<OperationPass<ModuleOp>> createAIERouteFlowsPass();
std::unique_ptr<OperationPass<ModuleOp>> createAIERoutePacketFlowsPass();
std::unique_ptr<OperationPass<func::FuncOp>> createAIEVectorOptPass();
std::unique_ptr<OperationPass<ModuleOp>> createAIEPathfinderPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "polyaie/AIE/AIEPasses.h.inc"

} // namespace AIE
} // namespace xilinx

#endif
