//===-AIEVectorize.cpp - Vectorizer for AIE architecture --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//
// This file implements the functionality to massage the output from affine
// supervectorizer into a set of operations and datatypes corresponding to
// AIE vector abstraction.
//===----------------------------------------------------------------------===//

#include "polyaie/AIE/Dialect/AIEVec/AIEVecUtils.h"
#include "polyaie/AIE/Dialect/AIEVec/IR/AIEVecOps.h"
#include "polyaie/AIE/Dialect/AIEVec/Transforms/IntervalReuse.h"
#include "polyaie/AIE/Dialect/AIEVec/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;
using namespace arith;
using namespace vector;
using namespace xilinx;
using namespace xilinx::aievec;

#define DEBUG_TYPE "aie-vect"

namespace {
// A struct to pack the global state required for vectorization at one place.
// Local to this translation unit.
struct VectState {
  // A vector of all the reuse intervals created. Class IntervalReuse represents
  // the cluster of data access (with reuse potential) along the vectorized
  // dimension of each array, It clusters together reads that have a potential
  // of vector-level data reuse. Therefore, array accesses A[i][j:j+8] and
  // A[i+2][j:j+8] will map to different IntervalReuse objects.
  SmallVector<IntervalReuse *, 16> reuseIntervals;
  // Map from a transfer_read operation to the IntervalReuse object it belongs
  // to.
  DenseMap<Operation *, IntervalReuse *> opToIntervalMap;
  // Map from a transfer_read operation to its linearized access expression.
  // Linearized expression for access A[i][j], where A is of dimensionality MxN
  // is (i*N+j). We assume that the innermost dimension is the vectorized
  // dimension.
  DenseMap<Operation *, AffineExpr> linearizedAccess;
  // A map from an index (of array access) to an expr dim map (e.g., i->d0). We
  // need this to create the correct linearized expressions for all the array
  // accesses in the function.
  DenseMap<Value, AffineExpr> indexToExprDimMap;
  // For each transfer_read operation, a map from its container basic block to
  // the enclosing for/while loops. This helps us identify two instructions
  // that are nested together, even if they belong to different basic blocks.
  DenseMap<Block *, SmallVector<Operation *, 8>> blockToEnclosingLoops;
  // This is specific to 8x8 scheme. For an 8x8 scheme, every mul/fma is
  // replaced by two mul/fmas in AIE dialect. So we keep track of the pair.
  DenseMap<Operation *, Operation *> pairedOp;
  // If we fuse a representative mul/fma op with another fma op to exploit the
  // column topology of the AIE intrinsic, then cache, for the representative
  // op, the compile-time constant access distance between their two operands.
  // The first(second) offset of the pair represents the access distance
  // between the first(second) operands of the representative op and the the
  // fused op(s). This access distance will be used to compute the xstep/zstep
  // attribute.
  DenseMap<Operation *, std::pair<int32_t, int32_t>> opToColOffsets;
  // A set of operations that are msc (fmsub) ops. We do not differentiate
  // between mac and msc ops at vector dialect level. The only op in vector
  // dialect is just FMA op.
  llvm::SmallSet<Operation *, 8> mscOps;
  // Used to build and insert all the new operations created.
  OpBuilder builder;
  // The shift val for ups and srs intinsics. This value should be between 0
  // and 63.
  int8_t shift;
  // The zero offset, indicating the position of recurring 0 in the input
  // filter. The counting starts at 1. For example, if the filter array is
  // {1,2,3,0,4,5,6,0,7,8,9,0}, then zeroOffset=4.
  int32_t zeroOffset;
  // The duplicate count, indicating the number of times a value is duplicated
  // in the filter. The filter values must be duplicated at least twice for the
  // i8xi8 scheme. An example of filter for i8xi8 scheme is {0,0,1,1,2,2,3,3},
  // with dupFactor=2.
  int32_t dupFactor;

  // Constructors
  VectState(MLIRContext *context)
      : builder(context), shift(0), zeroOffset(0), dupFactor(2) {}
  VectState(MLIRContext *context, int8_t s, int32_t z, int32_t d)
      : builder(context), shift(s), zeroOffset(z), dupFactor(d) {}

  IntervalReuse *getIntervalForOperation(Operation *op);
};

// Get the IntervalReuse object for a given read operation
IntervalReuse *VectState::getIntervalForOperation(Operation *op) {
  assert(opToIntervalMap.count(op) &&
         "could not find the IntervalReuse object for op");
  return opToIntervalMap[op];
}

// A struct to store the attributes (start, lo/hi offset, step, square) for an
// AIE fma, mul, or select operation.
struct AIEOpAttributes {
  std::string select;
  SmallVector<std::string, 2> start;
  SmallVector<std::string, 2> offset, offset_hi;
  SmallVector<std::string, 2> step;
  SmallVector<std::string, 2> square;
};

// A struct that stores some of the attributes for a vector type
struct AIEVecAttributes {
  // The number of lanes along the vectorized dimension for the vector type.
  // For a multidimensional vector, it is the innermost dimension size.
  unsigned lanes;
  // For a 1D vector, capture its size in bits. For an nD vector, capture the
  // size of the innermost dimension in bits.
  int32_t vecSizeInBits;
  // Underlying scalar element type
  Type elementType;
  // The the element size in bits
  int32_t elementSizeInBits;
  // Does the vector load data from memory
  bool loadFromMemory;
  // Is the vector splat?
  bool isSplat;
  // Constructors
  AIEVecAttributes(unsigned l, unsigned vs, Type et, int32_t es)
      : lanes(l), vecSizeInBits(vs), elementType(et), elementSizeInBits(es),
        loadFromMemory(false), isSplat(false) {}
  AIEVecAttributes(unsigned l, unsigned vs, Type et, int32_t es, bool lm,
                   bool sp)
      : lanes(l), vecSizeInBits(vs), elementType(et), elementSizeInBits(es),
        loadFromMemory(lm), isSplat(sp) {}
};

// Structure to capture the lane/col topology, and the element type size of
// xbuff and ybuff. Captures all the necessary information to map the incoming
// mul/mac op to the vectorization scheme.
struct Scheme {
  // lanes and columns in the vector intrinsic
  int32_t lanes, cols;
  // size (in bits) of the underlying scalar element type of xbuff and zbuff
  int32_t xbits, zbits;
  // Constructor
  Scheme(int32_t l, int32_t c, int32_t x, int32_t z)
      : lanes(l), cols(c), xbits(x), zbits(z) {}
};
} // namespace

//===----------------------------------------------------------------------===//
// Helper Routines
//===----------------------------------------------------------------------===//

// Combine the result of vector-related utilities into a single utility.
static inline AIEVecAttributes getVectorStats(VectorType type) {
  return AIEVecAttributes(getVectorLaneSize(type), getVectorSizeInBits(type),
                          type.getElementType(), getElementSizeInBits(type));
}

// Get the vector stats for an operation's result.
static inline AIEVecAttributes getResultVecStats(Operation *op,
                                                 unsigned idx = 0) {
  VectorType vtype = op->getResult(idx).getType().cast<VectorType>();
  return getVectorStats(vtype);
}

// Get the vector stats for an operation's operand.
static inline AIEVecAttributes
getOperandVecStats(Operation *op, VectState *state, unsigned idx = 0) {
  assert(op->getNumOperands() > idx);
  Operation *defOp = op->getOperand(idx).getDefiningOp();
  VectorType vtype = defOp->getResult(0).getType().cast<VectorType>();
  auto ret = getVectorStats(vtype);
  // if the defining op is a transfer read, get the extent read from source
  if (auto readOp = dyn_cast<TransferReadOp>(defOp)) {
    IntervalReuse *iv = state->getIntervalForOperation(readOp);
    ret.vecSizeInBits = iv->getIntervalWidth(readOp);
    // Set load from memory to true
    ret.loadFromMemory = true;
    // Check if the load is splat
    ret.isSplat = readOp.getPermutationMap().isConstant();
  }
  return ret;
}

// Get the number of rows and columns in the vector scheme.
static inline std::pair<int32_t, int32_t> getNumRowsAndCols(Operation *op) {
  assert(op->getNumOperands() >= 2 && op->getNumResults() == 1);

  Operation *left = op->getOperand(0).getDefiningOp();
  Operation *right = op->getOperand(1).getDefiningOp();

  // Get the number of lanes
  VectorType vtype = op->getResult(0).getType().cast<VectorType>();
  int32_t lanes = getVectorLaneSize(vtype);

  // Get the data sizes for left and right operands
  VectorType ltype = left->getResult(0).getType().cast<VectorType>();
  VectorType rtype = right->getResult(0).getType().cast<VectorType>();
  int32_t lsize = getElementSizeInBits(ltype);
  int32_t rsize = getElementSizeInBits(rtype);

  int32_t width = (lsize == 8 && rsize == 8)    ? 128
                  : (lsize == 16 && rsize == 8) ? 64
                                                : 32;

  // Now the computation
  int32_t m = 1;
  if (lsize == 32)
    m *= 2;
  if (rsize == 32)
    m *= 2;
  int32_t cols = width / (m * lanes);
  return std::make_pair(lanes, cols);
}

// Fuse the access extent of two mul/fma operations. This means that for the
// corresponding lhs(rhs) operands of op1 and op2, check if they read from
// memory, and if they do, extend their access extent to their union. For
// example if the left operand of Op1 has access extent [0,256], and the left
// operand of Op2 has access extent [128,512], where these two accesses belong
// to the same ReuseInterval, then the union is [0,512]. This union will be the
// new access extent of the left operands of both Op1 and Op2.
static inline void fuseAccessExtent(Operation *Op1, Operation *Op2,
                                    VectState *state) {
  // Assert that the input operations are of expected type
  bool expectedTypes =
      isa<vector::FMAOp>(Op2) && isa<MulIOp, MulFOp, vector::FMAOp>(Op1);
  assert(expectedTypes && "incorrect operation types");

  // Iterate over the even and odd operands for both the operations
  for (int idx = 0; idx < 2; ++idx) {
    Operation *op1 = Op1->getOperand(idx).getDefiningOp();
    Operation *op2 = Op2->getOperand(idx).getDefiningOp();
    // If both op1 and op2 are transfer read ops, then we need to create an
    // interval that subsumes the extent read by both op1 an op2.
    if (isa<TransferReadOp>(op1) && isa<TransferReadOp>(op2)) {
      IntervalReuse *iv1 = state->getIntervalForOperation(op1);
      IntervalReuse *iv2 = state->getIntervalForOperation(op2);
      // Assert that both the ops belong to the same IntervalReuse object
      assert(iv1 == iv2);
      assert(iv1->getInterval(op1) == iv2->getInterval(op2));
      auto op1Extent = iv1->getAccessExtent(op1);
      auto op2Extent = iv2->getAccessExtent(op2);
      // Create the new extent that's a union of refExtent and opExtent
      auto newExtent =
          std::make_pair(std::min(op1Extent.first, op2Extent.first),
                         std::max(op1Extent.second, op2Extent.second));
      // And now update the read extents with the union
      iv1->setAccessExtent(op1, newExtent);
      iv2->setAccessExtent(op2, newExtent);
    }
  }
}

// To be a simple lane-wise multiplication, we check that
// (1) both lhs and rhs operands come from vector of same size,
// (2) no operand is splat, and
// (3) no type is float if Op is mul/fma.
static bool isSimpleVectIntrinsic(Operation *Op, VectState *state) {
  // The incoming operator should be mul/fma/sub/add op
  bool isMulOrFMAOp = isa<MulIOp, MulFOp, vector::FMAOp>(Op);
  bool isSubOrAddOp = isa<SubIOp, SubFOp, AddIOp, AddFOp>(Op);
  if (!isMulOrFMAOp && !isSubOrAddOp)
    return true;

  // Get the vec stats for result, left, and right operand
  AIEVecAttributes vstat = getResultVecStats(Op);
  AIEVecAttributes lstat = getOperandVecStats(Op, state, 0);
  AIEVecAttributes rstat = getOperandVecStats(Op, state, 1);

  bool sizeMatches = lstat.vecSizeInBits == rstat.vecSizeInBits &&
                     vstat.vecSizeInBits == rstat.vecSizeInBits &&
                     lstat.elementType == rstat.elementType &&
                     vstat.elementType == rstat.elementType;
  bool noSplat = !lstat.isSplat && !rstat.isSplat;
  bool noFloat = !vstat.elementType.isa<FloatType>() &&
                 !lstat.elementType.isa<FloatType>() &&
                 !rstat.elementType.isa<FloatType>();

  return sizeMatches && noSplat && (isSubOrAddOp || noFloat);
}

// Return true if this is a vector dialect op meeting the following conditions:
// (1) all the operands and results are vectorized; and
// (2) all the vector sizes are the same.
// (3) all the vectors have the same underlying scalar element type.
static bool isWellFormedVectorOp(Operation *Op) {
  // The op must have at least an operand or result
  if (Op->getNumOperands() == 0 && Op->getNumResults() == 0)
    return false;

  SmallVector<Value, 8> operandsAndResults;
  operandsAndResults.append(Op->operand_begin(), Op->operand_end());
  operandsAndResults.append(Op->result_begin(), Op->result_end());

  // Check 1. all the operands and results must be vector types
  for (auto val : operandsAndResults) {
    if (!val.getType().isa<VectorType>())
      return false;
  }

  VectorType refType = operandsAndResults.back().getType().cast<VectorType>();
  Type scalarType = refType.getElementType();
  unsigned refSize = getVectorLaneSize(refType);
  for (auto val : operandsAndResults) {
    VectorType vtype = val.getType().cast<VectorType>();
    // Check 2. All the vector sizes must be same
    if (refSize != getVectorLaneSize(vtype))
      return false;
    // Check 3. The underlying scalar type of all the vectors must be the same
    if (scalarType != vtype.getElementType())
      return false;
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Manipulate affine expressions
//===----------------------------------------------------------------------===//

// Make a flattened affine expression from the given exprs array. Functionally
// identical to makeCanonicalStridedLayoutExpr except that the returned
// AffineExpr is not simplified.
static AffineExpr makeFlattenedStridedExpr(ArrayRef<int64_t> sizes,
                                           ArrayRef<AffineExpr> exprs,
                                           MLIRContext *context) {
  assert(!sizes.empty() && !exprs.empty() &&
         "expected non-empty sizes and exprs");

  // Size 0 corner case is useful for canonicalizations.
  if (llvm::is_contained(sizes, 0))
    return getAffineConstantExpr(0, context);

  auto maps = AffineMap::inferFromExprList(exprs);
  assert(!maps.empty() && "Expected one non-empty map");
  unsigned nSymbols = maps[0].getNumSymbols();

  AffineExpr expr;
  bool dynamicPoisonBit = false;
  int64_t runningSize = 1;
  for (auto en : llvm::zip(llvm::reverse(exprs), llvm::reverse(sizes))) {
    int64_t size = std::get<1>(en);
    // Degenerate case, no size =-> no stride
    if (size == 0)
      continue;
    AffineExpr dimExpr = std::get<0>(en);
    AffineExpr stride = dynamicPoisonBit
                            ? getAffineSymbolExpr(nSymbols++, context)
                            : getAffineConstantExpr(runningSize, context);
    expr = expr ? expr + dimExpr * stride : dimExpr * stride;
    if (size > 0) {
      runningSize *= size;
      assert(runningSize > 0 && "integer overflow in size computation");
    } else {
      dynamicPoisonBit = true;
    }
  }
  return expr;
}

// Construct a linearized affine expression for the transfer_read op.
static AffineExpr constructLinearizedAffineExpr(TransferReadOp readOp,
                                                VectState *state) {
  // The global state stores a map from readOp to its linearized expression. If
  // the linear expression is already computed for this readOp, return it.
  if (state->linearizedAccess.count(readOp))
    return state->linearizedAccess[readOp];

  SmallVector<Value, 4> indices(readOp.getIndices().begin(),
                                readOp.getIndices().end());
  MemRefType memRefType = readOp.getSource().getType().cast<MemRefType>();
  MLIRContext *context = memRefType.getContext();

  SmallVector<AffineExpr, 8> exprVec;
  // Iterate over all the indices. If the index has an affine apply op
  // associated with it, we extract that. Otherwise we use the index from
  // default map.
  for (auto idxAndValue : llvm::enumerate(indices)) {
    auto value = idxAndValue.value();
    // If the access is a map via affine apply op (e.g., A[i+2], where the map
    // is d0 -> d0+2), push in the map after replacing all the dims with unique
    // index identifiers (e.g., let the unique identifier for index i be k0).
    if (AffineApplyOp apOf = value.getDefiningOp<AffineApplyOp>()) {
      AffineMap map = apOf.getAffineMap();
      assert(map.getNumResults() == 1 &&
             "Failed to create linearized affineExpr for complicated index");
      SmallVector<AffineExpr, 4> indexExprs;
      // Each operand of the map corresponds to a loop index. For each operand
      // (i.e., loop index), we create a unique dim expr.
      for (auto index : apOf.getMapOperands()) {
        if (auto cIdx = index.getDefiningOp<arith::ConstantOp>()) {
          auto idxVal = cIdx.getValue().cast<IntegerAttr>().getValue();
          unsigned idx = idxVal.getSExtValue();
          indexExprs.push_back(getAffineConstantExpr(idx, context));
        } else {
          if (!state->indexToExprDimMap.count(index))
            state->indexToExprDimMap[index] =
                getAffineDimExpr(state->indexToExprDimMap.size(), context);
          indexExprs.push_back(state->indexToExprDimMap[index]);
        }
      }
      // Now create a correct map expression using the unique dim exprs
      exprVec.push_back(map.getResult(0).replaceDims(indexExprs));
    }
    // If the index is an arith constant (e.g., A[3]), create an affine expr
    // from the constant value.
    else if (auto cOp = value.getDefiningOp<arith::ConstantOp>()) {
      auto idxVal = cOp.getValue().cast<IntegerAttr>().getValue();
      unsigned idx = idxVal.getSExtValue();
      exprVec.push_back(getAffineConstantExpr(idx, context));
    }
    // Default: the readop index is simply the loop index (e.g., A[i]).
    else {
      if (!state->indexToExprDimMap.count(value))
        state->indexToExprDimMap[value] =
            getAffineDimExpr(state->indexToExprDimMap.size(), context);
      exprVec.push_back(state->indexToExprDimMap[value]);
    }
  }

  assert(!exprVec.empty() && "Could not construct linearized affineExpr");

  // Linearize the exprVec as a strided access, but do not simplify
  auto ret = makeFlattenedStridedExpr(memRefType.getShape(), exprVec,
                                      memRefType.getContext());
  // Cache this readOp and linearized expr into the global map
  state->linearizedAccess[readOp] = ret;
  return ret;
}

// From a linearized affine expression, compute the base and the constant
// offset. If the access is A[i][j+2] for an N*N array A, the linearized
// expression will be A[i*N+j+2]. The base in this case will be (i*N+j), and the
// offset will be 2.
static std::pair<AffineExpr, int32_t> getBaseAndOffset(AffineExpr expr) {
  AffineExpr base = expr;
  int32_t offset = 0;
  // If expr is already a constant, the base is nullptr, and offset is expr
  if (auto constExpr = expr.dyn_cast<AffineConstantExpr>()) {
    base = nullptr;
    offset += constExpr.getValue();
  }
  // If this is a binary '+' expression, compute the constant offset. Currently
  // this is just a simple FSM. This must evolve as we explore more complex
  // access patterns.
  else if (auto binopExpr = expr.dyn_cast<AffineBinaryOpExpr>()) {
    if (binopExpr.getKind() == AffineExprKind::Add) {
      AffineExpr lhs = binopExpr.getLHS(), rhs = binopExpr.getRHS();
      if (auto constExpr = lhs.dyn_cast<AffineConstantExpr>()) {
        base = rhs;
        offset += constExpr.getValue();
      }
      if (auto constExpr = rhs.dyn_cast<AffineConstantExpr>()) {
        base = base == rhs ? nullptr : lhs;
        offset += constExpr.getValue();
      }
    }
  }
  return std::make_pair(base, offset);
}

//===----------------------------------------------------------------------===//
// AIE vector op generation routines
//===----------------------------------------------------------------------===//

// Generate and return an SRS op. Incoming `source` is an accumulator. The
// output should be a vector of element type `scalarType`.
static aievec::SRSOp generateSRSOp(Value source, Type scalarType,
                                   VectState *state, Location loc) {
  // The source should be accumulator type
  Type accType = source.getType();
  assert(isAccType(accType) && "srs source should be accumulator type");

  // Get the number of lanes
  unsigned lanes = accType.cast<AccType>().getLanes();
  // Now generate the new vector type for the SRS intrinsic
  VectorType srsType = createVectorType(lanes, scalarType);
  // Create the SRS op
  aievec::SRSOp srsOp =
      state->builder.create<aievec::SRSOp>(loc, srsType, source, state->shift);

  assert(srsOp && "could not create srs op");
  return srsOp;
}

// Generate and return a UPS op. Incoming `source` is a vector which needs
// to be moved to an accumulator.
static aievec::UPSOp generateUPSOp(Value source, VectState *state,
                                   Location loc) {
  Type sourceType = source.getType();
  assert(!isAccType(sourceType) && "ups source should not be accumulator");

  AccType accType = getAccType(sourceType);

  // Create a new UPS instruction
  aievec::UPSOp upsOp =
      state->builder.create<aievec::UPSOp>(loc, accType, source, state->shift);

  assert(upsOp && "could not create ups op");
  return upsOp;
}

// Generate and return a Concat op.
static aievec::ConcatOp generateConcatOp(SmallVector<Value> &sources,
                                         VectState *state, Location loc,
                                         VectorType concatType = nullptr) {
  assert(sources.size() > 1 && "must concat at least two vectors");

  VectorType vecType = sources.back().getType().cast<VectorType>();

  for (auto source : sources) {
    VectorType type = source.getType().cast<VectorType>();
    assert(type == vecType && "sources of concat op not of same type");
  }

  if (!concatType) {
    // Get the number of lanes and scalar type to create the concat result type
    unsigned lanes = sources.size() * getVectorLaneSize(vecType);
    Type scalarType = vecType.getElementType();
    concatType = createVectorType(lanes, scalarType);
  }

  // Create the concat op
  aievec::ConcatOp concatOp =
      state->builder.create<aievec::ConcatOp>(loc, concatType, sources);

  assert(concatOp && "could not create concat op");
  return concatOp;
}

// Generate and return a select operation. The start, offset, etc. for lanes
// are in opAttr.
static aievec::SelectOp generateSelectOp(Value xbuff, AIEOpAttributes &opAttr,
                                         unsigned lanes, VectState *state,
                                         Location loc, Value ybuff = nullptr) {
  // Assert that we have computed the attributes (start, offset, etc.) for both
  // lanes, and that select is non-empty.
  assert(!opAttr.select.empty());
  assert(opAttr.start.size() == opAttr.offset.size() &&
         opAttr.start.size() == 2);

  VectorType xtype = xbuff.getType().cast<VectorType>();
  // Verify that lanes is <= xtype lanes
  assert(lanes <= getVectorLaneSize(xtype));
  // Create the result type
  VectorType resultType = createVectorType(lanes, xtype.getElementType());

  // Create AIE dialect select op
  aievec::SelectOp selectOp = state->builder.create<aievec::SelectOp>(
      loc, resultType, xbuff, opAttr.select, opAttr.start[0], opAttr.offset[0],
      opAttr.offset_hi[0], opAttr.square[0], opAttr.start[1], opAttr.offset[1],
      opAttr.offset_hi[1], opAttr.square[1], ybuff);

  assert(selectOp && "could not create select op");
  return selectOp;
}

// Generate and return an Ext op. The lanes indicate the lanes in vector
// output, and idx defines which part of source is extracted.
static aievec::ExtOp generateExtOp(Value source, unsigned lanes, int8_t idx,
                                   VectState *state, Location loc) {
  VectorType stype = source.getType().cast<VectorType>();
  // Verify that lanes*idx is <= stype lanes
  assert(lanes * (idx + 1) <= getVectorLaneSize(stype));
  // Create the result type
  VectorType resultType = createVectorType(lanes, stype.getElementType());

  // Create AIE dialect ext op
  aievec::ExtOp extOp =
      state->builder.create<aievec::ExtOp>(loc, resultType, source, idx);

  assert(extOp && "could not create ext op");
  return extOp;
}

// Generate and return an Pack op.
static aievec::PackOp generatePackOp(Value source, VectState *state,
                                     Location loc) {
  // Create the result type
  VectorType stype = source.getType().cast<VectorType>();
  unsigned lanes = getVectorLaneSize(stype);
  Type i8Type = mlir::IntegerType::get(source.getContext(), 8);
  VectorType resultType = createVectorType(lanes, i8Type);

  // Create AIE dialect pack op
  aievec::PackOp packOp =
      state->builder.create<aievec::PackOp>(loc, resultType, source);

  assert(packOp && "could not create pack op");
  return packOp;
}

// Generate and return an Add op.
static aievec::AddOp generateAddOp(Operation *Op, AIEOpAttributes &opAttr,
                                   VectState *state) {
  // Assert that we computed the attributes for both the operands
  assert(opAttr.start.size() == opAttr.offset.size() &&
         opAttr.start.size() == 2);

  aievec::AddOp addOp = state->builder.create<aievec::AddOp>(
      Op->getLoc(), Op->getResult(0).getType(), Op->getOperand(0),
      Op->getOperand(1), opAttr.start[0], opAttr.offset[0], opAttr.offset_hi[0],
      opAttr.square[0], opAttr.start[1], opAttr.offset[1], opAttr.offset_hi[1],
      opAttr.square[1]);
  return addOp;
}

// Generate and return a Sub op.
static aievec::SubOp generateSubOp(Operation *Op, AIEOpAttributes &opAttr,
                                   VectState *state) {
  // Assert that we computed the attributes for both the operands
  assert(opAttr.start.size() == opAttr.offset.size() &&
         opAttr.start.size() == 2);

  aievec::SubOp subOp = state->builder.create<aievec::SubOp>(
      Op->getLoc(), Op->getResult(0).getType(), Op->getOperand(0),
      Op->getOperand(1), opAttr.start[0], opAttr.offset[0], opAttr.offset_hi[0],
      opAttr.square[0], opAttr.start[1], opAttr.offset[1], opAttr.offset_hi[1],
      opAttr.square[1]);
  return subOp;
}

// Generate and return an FMA operation in AIE dialect. This operation will
// have the start and offset fields for each operand. If the acc operand of
// fmaOp is a transfer_read operation, then we need to add an SRS instruction
// that will load the vector value into an accumulator.
static Operation *generateFMAOp(vector::FMAOp fmaOp, AIEOpAttributes &opAttr,
                                VectState *state, bool i8xi8_pairedOp = false) {
  // Assert that we have computed the attributes (start, offset, etc.) for both
  // left and right operands of the fma operation.
  assert(opAttr.start.size() == opAttr.offset.size() &&
         opAttr.start.size() == 2);

  // If the accumulator is not of type aievec::AccType for integer FMA
  // we need to generate a ups instruction.
  bool isInt = fmaOp.getLhs()
                   .getType()
                   .cast<VectorType>()
                   .getElementType()
                   .isa<IntegerType>();
  Value acc = fmaOp.getAcc();
  // If i8xi8_pairedOp is true, then we are trying to generated the paired FMA
  // op for i8xi8 scheme. Find the paired accumulator.
  if (i8xi8_pairedOp) {
    Operation *defOp = fmaOp.getAcc().getDefiningOp();
    if (state->pairedOp.count(defOp))
      acc = state->pairedOp[defOp]->getResult(0);
  }
  if (isInt && !isAccType(acc.getType())) {
    acc = generateUPSOp(acc, state, fmaOp->getLoc());
    LLVM_DEBUG(llvm::dbgs()
               << "\n\nCreated UPS op " << acc << " to move the output of "
               << fmaOp << " into accumulator");
  }
  // If the lhs operand vector is not >= twice the rhs operand vector, then use
  // concat operator.
  Value lhs = fmaOp.getLhs();
  if (!isSimpleVectIntrinsic(fmaOp, state)) {
    AIEVecAttributes lstat = getOperandVecStats(fmaOp, state, 0);
    assert(lstat.vecSizeInBits % 256 == 0);
    if (lstat.vecSizeInBits == 256) {
      VectorType concatType =
          createVectorType(512 / lstat.elementSizeInBits, lstat.elementType);
      SmallVector<Value> sources = {lhs, lhs};
      lhs = generateConcatOp(sources, state, fmaOp->getLoc(), concatType);
    }
  }

  // Check if this is an fmsub op, and if so, then we need to generate msc op
  bool isSub = state->mscOps.count(fmaOp);

  // Create AIE dialect fma/msc op
  Operation *xfmaOp = state->builder.create<aievec::FMAOp>(
      fmaOp->getLoc(), lhs, fmaOp.getRhs(), acc, opAttr.start[0],
      opAttr.offset[0], opAttr.offset_hi[0], opAttr.step[0], opAttr.square[0],
      opAttr.start[1], opAttr.offset[1], opAttr.offset_hi[1], opAttr.step[1],
      opAttr.square[1], isSub);

  assert(xfmaOp && "could not create fma op");
  return xfmaOp;
}

// Generate a MUL operation in AIE dialect. This operation will have the start
// and offset fields for each operand. The output will be in an accumulator.
template <typename T>
static Operation *generateMulOp(T mulOp, AIEOpAttributes &opAttr,
                                VectState *state) {
  // Assert that we have computed the attributes (start, offset, etc.) for both
  // left and right operands of the mul operation.
  assert(opAttr.start.size() == opAttr.offset.size() &&
         opAttr.start.size() == 2);

  // If the return type is not accumulator type already, create a new
  // accumulator type for return value of integer types.
  Type opType = mulOp.getType();
  if (!isAccType(opType) &&
      opType.cast<VectorType>().getElementType().isa<IntegerType>())
    opType = getAccType(opType);

  // If the lhs operand vector is not >= twice the rhs operand vector, then use
  // concat operator.
  Value lhs = mulOp.getLhs();
  if (!isSimpleVectIntrinsic(mulOp, state)) {
    AIEVecAttributes lstat = getOperandVecStats(mulOp, state, 0);
    assert(lstat.vecSizeInBits % 256 == 0);
    if (lstat.vecSizeInBits == 256) {
      VectorType concatType =
          createVectorType(512 / lstat.elementSizeInBits, lstat.elementType);
      SmallVector<Value> sources = {lhs, lhs};
      lhs = generateConcatOp(sources, state, mulOp->getLoc(), concatType);
    }
  }

  // Create AIE dialect mul op
  Operation *xmulOp = state->builder.create<aievec::MulOp>(
      mulOp->getLoc(), lhs, mulOp.getRhs(), opType, opAttr.start[0],
      opAttr.offset[0], opAttr.offset_hi[0], opAttr.step[0], opAttr.square[0],
      opAttr.start[1], opAttr.offset[1], opAttr.offset_hi[1], opAttr.step[1],
      opAttr.square[1]);

  assert(xmulOp && "could not create mul op");
  return xmulOp;
}

// For a transfer_read op, generate a corresponding UPD op. Multiple
// transfer_read ops will have the same UPD op if their read access extent is
// subsumed by the same interval. The updOps will have to be inserted at the
// head of region if the region has multiple blocks, or closer to the readOp
// otherwise.
static aievec::UPDOp
generateUPDOp(TransferReadOp readOp,
              DenseMap<std::tuple<IntervalReuse *, int32_t, int32_t>,
                       std::pair<aievec::UPDOp, int8_t>> &memToUpdMap,
              Region &region, VectState *state) {
  // Get the read access extent and interval of this read operation
  IntervalReuse *iv = state->getIntervalForOperation(readOp);
  auto extent = iv->getAccessExtent(readOp);
  auto interval = iv->getInterval(readOp);
  int32_t intervalWidth = interval.second - interval.first;
  assert(intervalWidth >= 128 && "Interval computation incorrect");

  // Create the upd vector type. To do so, we need the underlying element type.
  // We can divide the interval size by that to get the number of lanes in the
  // result vector of upd op.
  VectorType vecType = readOp.getVector().getType().cast<VectorType>();
  Type elementType = vecType.getElementType();
  int32_t elementSizeInBits = getElementSizeInBits(vecType);
  int intervalWidthInBytes = intervalWidth / elementSizeInBits;
  Type updVecType = createVectorType(intervalWidthInBytes, elementType);

  // Compute the mid value of the interval. This is useful because for
  // intervalWidth > 256, we can split the load into two steps: the bits to the
  // left/right of mid will be loaded using upd idx=0/idx=1 operator.
  int32_t mid = interval.first + intervalWidth / 2;
  // Compute the (aligned) extent of interval that this read requires to be
  // loaded.
  int32_t lb =
      intervalWidth <= 256 || extent.first < mid ? interval.first : mid;
  int32_t ub =
      intervalWidth <= 256 || extent.second > mid ? interval.second : mid;

  // Find if we have already created upd op idx=0/idx=1 for this interval
  aievec::UPDOp updOp = nullptr;
  // initial value 0 of updIndices means neither upd op idx=0 nor idx=1 were
  // created.
  int8_t updIndices = 0;
  auto key = std::make_tuple(iv, interval.first, interval.second);
  if (memToUpdMap.count(key)) {
    updOp = memToUpdMap[key].first;
    updIndices = memToUpdMap[key].second;
  }

  // This readOp could be A[i][j+2], where A is a 32-bit array. Assume that its
  // read access extent is subsumed by interval [0,512]. This 512-bit interval
  // load should be broken into two 256-bit UPD ops. We need to find the right
  // offset from A[i][j+2] where the reads will start. The first offset (in
  // bits) will be A[i][j+2]-2*32, and the second offset will be
  // A[i][j+2]+256-2*32. Essentially, the offsets should make the load well
  // aligned. Below, we compute this (-2*32) offset to make the loads aligned.
  SmallVector<Value, 4> indices(readOp.getIndices().begin(),
                                readOp.getIndices().end());
  // Get the linearized access expression for the read to compute the offset
  AffineExpr linearAccess = constructLinearizedAffineExpr(readOp, state);
  AffineExpr base;
  int32_t offset;
  // Get the base and offset from linear access expr
  std::tie(base, offset) = getBaseAndOffset(linearAccess);
  offset *= elementSizeInBits; // get the offset in bits

  // The insertion point depends on whether the region has a single block or
  // not. If it has a single block, that block will be the front block, so we
  // can insert the UPDOp closer to the readOp. However, if the region has
  // multiple blocks, we will insert all the UPDs to the front block of the
  // region so that the UPDs dominate the entire region.
  bool singleBlock = region.getBlocks().size() == 1;
  if (singleBlock)
    state->builder.setInsertionPoint(readOp);
  else
    state->builder.setInsertionPointToStart(&region.front());

  // If the extent <= 256 bits, we can directly copy data from mem into vector
  // without using a upd. So we try to chunk the interval into sub-intervals of
  // width >= 256 bits.
  int32_t incr = std::max(256, intervalWidth / 2);
  int8_t idx = 1;
  for (int32_t start = interval.first; start < interval.second;
       start += incr, ++idx) {
    // If idx=1, then this indicates a potential upd0 instruction. If idx=2, it
    // will be upd1 instruction.
    assert(idx <= 2 && "The only allowed values for UPD index are 0 and 1");
    int32_t end = std::min(interval.second, start + incr);
    // We are at sub-interval [start,end] of the vector interval. Check if this
    // sub-interval is subsumed by [lb,ub], and the upd op corresponding to this
    // sub-interval is not already generated.
    if (lb <= start && ub >= end && (updIndices & idx) == 0) {
      // Generate the upd instruction, and link it with a previous upd op
      // corresponding to the same read.
      updOp = state->builder.create<aievec::UPDOp>(
          readOp.getLoc(), updVecType, readOp.getSource(), indices,
          start - offset, idx - 1, updOp ? updOp.result() : nullptr);

      LLVM_DEBUG(llvm::dbgs() << "\n\nCreated UPD op " << updOp
                              << " for read op " << readOp);

      // If the transfer_read has some apply operations, then they also need to
      // be hoisted.
      for (auto &value : indices) {
        if (AffineApplyOp apOf = value.getDefiningOp<AffineApplyOp>()) {
          // Skip hoisting if already above in lexicographical order
          if (apOf->getBlock() == readOp->getBlock() &&
              apOf->isBeforeInBlock(updOp))
            continue;
          apOf.getOperation()->moveBefore(updOp);
        }
      }
      // Set the (idx-1)'th bit in updIndices to indicate that we have already
      // created a upd op for index idx.
      updIndices |= idx;
    }
  }

  // Link the generated updOp to possibly pre-existing UPD ops for the key
  memToUpdMap[key] = std::make_pair(updOp, updIndices);
  return updOp;
}

//===----------------------------------------------------------------------===//
// AIE vectorization routines
//===----------------------------------------------------------------------===//

// For this vectorized read operation, find the loop that corresponds to the
// vectorized dimension, and return its step size.
static int32_t computeVecorizedLoopStepSize(Operation *op, VectState *state) {
  auto readOp = dyn_cast<TransferReadOp>(op);
  // If this operation is not a read op, return the default step size of 1
  if (!readOp)
    return 1;

  int32_t step = 0;
  bool found = false;
  VectorType vectorType = readOp.getResult().getType().cast<VectorType>();
  SmallVector<Value, 4> indices(readOp.getIndices().begin(),
                                readOp.getIndices().end());
  assert(vectorType && !indices.empty());

  // Verify that enclosing loops have been computed for the read operation
  auto block = readOp->getBlock();
  assert(state->blockToEnclosingLoops.count(block) &&
         "enclosing loops should have been computed for the read operation");
  auto enclosingLoops = state->blockToEnclosingLoops[block];

  // The vectorized (i.e., last) index of the permutation must correspond to a
  // loop nest. If not, this is a splat read.
  AffineExpr expr = readOp.getPermutationMap().getResults().back();
  if (auto dimExpr = expr.dyn_cast<AffineDimExpr>()) {
    assert(dimExpr.getPosition() <= indices.size() &&
           "Failed to find the permutation index in index map");
    auto index = indices[dimExpr.getPosition()];
    // Iterate over all enclosing loops, and find the one that is variant in
    // index.
    for (auto loop : enclosingLoops) {
      auto iv = cast<AffineForOp>(loop).getInductionVar();
      auto invariants = getInvariantAccesses(iv, indices);
      if (!invariants.count(index)) {
        assert(
            !found &&
            "stepsize computation already has an entry along the variant dim");
        step = cast<AffineForOp>(loop).getStep();
        found = true;
      }
    }
  }
  assert(isPowerOfTwo(step) &&
         "non-power-of-two vectorization factor not supported");
  // The step increment in vectorized code is scaled by factor of vector lanes;
  // account for that.
  unsigned lanes = getVectorLaneSize(vectorType);
  return step / lanes;
}

// AIE vector loads are always aligned to 128-bit boundary. So if the operation
// reads from an unaligned memory location, return the starting position of the
// read in the vector. Each element of the vector is 'elementSizeInBits' bits
// wide.
int32_t computeStartInAIEVec(Operation *op, VectState *state) {
  // In case the operation is not a transfer_read, return default start
  if (!isa<TransferReadOp>(op))
    return 0;

  auto readOp = cast<TransferReadOp>(op);

  // Get the scalar element type's size in bits
  VectorType vtype = readOp.getVector().getType().cast<VectorType>();
  int32_t scalarSizeInBits = getElementSizeInBits(vtype);

  // Get the linearized access expr for this read
  AffineExpr linearAccess = constructLinearizedAffineExpr(readOp, state);
  AffineExpr base;
  int32_t offset;
  // get the base and offset from linear access expr
  std::tie(base, offset) = getBaseAndOffset(linearAccess);
  offset *= scalarSizeInBits; // compute offset in bits
  // Now find the reuse interval to which this readOp belongs
  IntervalReuse *iv = state->getIntervalForOperation(op);
  std::pair<int32_t, int32_t> interval = iv->getInterval(op);

  // The readOp reads from this interval, and the start of this interval is
  // aligned to 128 bits. The AIE vector corresponding to this read will hold
  // the value [inteval.first,interval.second]. Return the position of the first
  // element that is read.
  assert(offset >= interval.first && "Failed to compute the start");
  return (offset - interval.first) / scalarSizeInBits;
}

// For an i8xi8 scheme, we require two muls to compute the 16-lane output. Each
// mul has a replicated computation, where the output in lane i is replicated
// in lane i+2. Given that, we take the output of two mul ops. and merge them
// into a v32int16 vector. Then we shuffle (using select) to form two v16int16
// vectors that are replicas of each other. Finally, we can pick one of them
// (using ext), and then pack it into a v16int8 output.
static Operation *concatAndInterleave_i8xi8(Operation *source1,
                                            Operation *source2,
                                            VectState *state, Location loc) {
  // Both the input sources must be accumulators
  assert(isAccType(source1->getResult(0).getType()) &&
         isAccType(source2->getResult(0).getType()));

  // The source values are in accumulator. So generate SRS intrinsic to convert
  // the accumulator output to vector output. We want the output to be in
  // v16int16 vector, since select operation does not operate on v16int8
  // vector.
  Type i16Type =
      mlir::IntegerType::get(source1->getResult(0).getType().getContext(), 16);
  auto srsOp1 = generateSRSOp(source1->getResult(0), i16Type, state, loc);
  auto srsOp2 = generateSRSOp(source2->getResult(0), i16Type, state, loc);

  // Now we concat the result of the two SRS ops to form a 32-lane vector
  SmallVector<Value> sources = {srsOp1->getResult(0), srsOp2->getResult(0)};
  auto concatOp = generateConcatOp(sources, state, loc);

  // Select the right bits of output to again form the 16-lane vector. opAttr
  // will cache the step,offsets,square, etc. for both lanes.
  AIEOpAttributes opAttr;
  // 0xi is 1100, which indicates that two values must alternately come from
  // xoffset and yoffset.
  opAttr.select = "0xcccccccc";
  // xstart is 0. Since there are only 2 unique values in the first 4 values
  // of the vector, ystart is 4.
  opAttr.start.push_back("0");
  opAttr.start.push_back("4");
  for (size_t idx = 0; idx < 2; ++idx) {
    // Consider only the even indices in offset (e.g., c, 8, 4, 0). The
    // absolute difference between even indices should be 4 (based on the
    // scheme, this will get multiplied by 2. So technically, xoffset picks the
    // values starting at indices 0, 8, 16, 24 from the v32int16 vector,
    // whereas yoffset picks values starting at indices 0+4, 8+4, 16+4, 24+4).
    opAttr.offset.push_back("0x0c080400");
    // We don't care for the lower 16 values in the v32int16 vector post
    // shuffle
    opAttr.offset_hi.push_back("0x0");
    // The first value must be permuted to offset 0, and the next to 1
    opAttr.square.push_back("0x1010");
  }
  // And now perform the selection
  auto selectOp =
      generateSelectOp(concatOp->getResult(0), opAttr, 32, state, loc);
  // The values in the first 16 lanes in the v32int16 vector are replicated in
  // the last 16 lanes. So select the first 16 lanes to form a v16int16 vector.
  auto extOp = generateExtOp(selectOp->getResult(0), 16, 0, state, loc);
  // Pack the int16 values to int8 values to form the v16int8 output vector
  auto packOp = generatePackOp(extOp->getResult(0), state, loc);
  return packOp;
}

// Perform a multitude of checks to see if rhs operand of the incoming add/sub
// operator is a mul operator, so that we can fuse them to form an FMA
// operator.
static bool canFuseMulAndAddOrSubIntoFMAOp(Operation *Op) {
  // Check 1. This should be an add or sub operation
  assert((isa<AddIOp>(Op) || isa<AddFOp>(Op) || isa<SubIOp>(Op) ||
          isa<SubFOp>(Op)) &&
         "operation must be an add or sub op");

  // Check 2. Op must have two operands and one result
  assert(Op->getNumOperands() == 2 && Op->getNumResults() == 1);

  // Check 3. rhs operand of the Op should be a mul op (If any operand of add
  // op is mul op, it is guaranteed to be rhs operand by explicit
  // reassociation done earlier).
  Operation *mulOp = Op->getOperand(1).getDefiningOp();
  if (!isa<MulIOp, MulFOp>(mulOp))
    return false;

  // Check 4. mulOp must also have two operands and one result
  assert(mulOp->getNumOperands() == 2 && mulOp->getNumResults() == 1);

  // Determine the lhs, rhs, and accumulator values.
  Value lhs = mulOp->getOperand(0);
  Value rhs = mulOp->getOperand(1);
  Value acc = Op->getOperand(0);

  assert(lhs && rhs && acc &&
         "Failed to find the three operands of the FMA op");

  // Check 5. All lhs, rhs, and acc must be vector types
  if (!lhs.getType().isa<VectorType>() || !rhs.getType().isa<VectorType>() ||
      !acc.getType().isa<VectorType>())
    return false;

  // Check 6. All the ops should belong to the same block, otherwise we might
  // not be able to fuse them safely.
  if (lhs.getParentBlock() != rhs.getParentBlock() ||
      rhs.getParentBlock() != acc.getParentBlock())
    return false;

  // Check 7. All the vector sizes must be same
  VectorType lhsType = lhs.getType().cast<VectorType>();
  VectorType rhsType = rhs.getType().cast<VectorType>();
  VectorType accType = acc.getType().cast<VectorType>();

  unsigned lhsVecSize = getVectorLaneSize(lhsType);
  unsigned rhsVecSize = getVectorLaneSize(rhsType);
  unsigned accVecSize = getVectorLaneSize(accType);

  if (lhsVecSize != rhsVecSize || rhsVecSize != accVecSize)
    return false;

  // Check 8. The underlying scalar element type of all vectors must be the
  // same
  if (lhsType.getElementType() != rhsType.getElementType() ||
      rhsType.getElementType() != accType.getElementType())
    return false;

  // And after all this, we can fuse mul and add into fma
  return true;
}

// In the advanced FMA schemes, the two vector operands of a mul/fma op are not
// the same size. If the incoming operation involves multiplication,
// reassociate the operands involved in multiplication so that the left operand
// comes from bigger vector. The exception to this rule is the 8x8 scheme,
// where the right operand must be the bigger vector.
static void reassociateMulOpBasedOnVecSize(Operation *Op, VectState *state) {
  // Get the stats for left and right operand vectors
  AIEVecAttributes lstat = getOperandVecStats(Op, state, 0);
  AIEVecAttributes rstat = getOperandVecStats(Op, state, 1);

  // No need to do anything if both vectors are the same size
  if (lstat.vecSizeInBits == rstat.vecSizeInBits)
    return;

  // Check if this is an 8x8 scheme
  bool is8x8 = lstat.elementSizeInBits == 8 && rstat.elementSizeInBits == 8;

  // Flip the operands if necessary
  bool flip = is8x8 ? lstat.vecSizeInBits > rstat.vecSizeInBits
                    : rstat.vecSizeInBits > lstat.vecSizeInBits;
  if (flip) {
    LLVM_DEBUG(llvm::dbgs()
               << "\n\nReassociating op " << *Op
               << " to correctly place operand coming from bigger vector");
    Value left = Op->getOperand(0);
    Value right = Op->getOperand(1);
    Op->setOperand(0, right);
    Op->setOperand(1, left);
    LLVM_DEBUG(llvm::dbgs() << "\n\tOp after reassociation: " << *Op);
  }
}

// If Op involves multiplication, and any operand involved in the
// multiplication is splat, make it the second operand of mul, unless its the
// 8x8 scheme. In that case, make splat the first operand.
static void reassociateMulOpWithSplat(Operation *Op, VectState *state) {
  // Op must have at least two operands (two for mul, three for fma), and one
  // result.
  assert(Op->getNumOperands() == 2 || Op->getNumOperands() == 3);
  assert(Op->getNumResults() == 1);

  // Get the left and right operand vector properties
  AIEVecAttributes lstat = getOperandVecStats(Op, state, 0);
  AIEVecAttributes rstat = getOperandVecStats(Op, state, 1);

  // No need to do anything if both operands are splat
  if (lstat.isSplat && rstat.isSplat)
    return;

  // Check if this is an 8x8 scheme
  bool is8x8 = lstat.elementSizeInBits == 8 && rstat.elementSizeInBits == 8;

  // Now flip operands if required
  bool flip = is8x8 ? rstat.isSplat : lstat.isSplat;
  if (flip) {
    LLVM_DEBUG(llvm::dbgs() << "\n\nReassociating op " << *Op
                            << " to place splat as correct operand");
    Value left = Op->getOperand(0);
    Value right = Op->getOperand(1);
    Op->setOperand(0, right);
    Op->setOperand(1, left);
    LLVM_DEBUG(llvm::dbgs() << "\n\tOp after reassociation: " << *Op);
  }
}

// Rewrite a mul and add/sub op as a vector dialect FMA op
static void fuseMulAndAddOrSubIntoFMAOp(Operation *Op, VectState *state) {
  Value acc = Op->getOperand(0);
  Operation *mulOp = Op->getOperand(1).getDefiningOp();
  Value lhs = mulOp->getOperand(0);
  Value rhs = mulOp->getOperand(1);

  // Create a new FMA op
  state->builder.setInsertionPointAfter(Op);
  Operation *fmaOp =
      state->builder.create<vector::FMAOp>(Op->getLoc(), lhs, rhs, acc);

  // If Op is a sub op, we tag the generated fma op as msc op
  bool isSub = isa<SubIOp, SubFOp>(Op);
  if (isSub)
    state->mscOps.insert(fmaOp);

  LLVM_DEBUG(llvm::dbgs() << "\n\nFused " << (isSub ? "sub" : "add") << " op "
                          << *Op << "\n\tand mul op " << *mulOp
                          << "\n\tinto fma op " << *fmaOp);

  // Replace all the uses of Op with the fmaOp, and remove Op
  Op->replaceAllUsesWith(fmaOp);
  Op->erase();
  // If Op was the only consumer of mulOp, then there are no more uses of
  // mulOp. Remove it.
  if (mulOp->use_empty())
    mulOp->erase();
}

// Given the operation attributes (start, offset, step, square, etc.), generate
// an AIE mul/fma op for the incoming vector mul/fma Op. 'nextStart' is used
// for schemes that require two AIE dialect fma ops to be generated for one
// vector dialect fma op; the only difference between the attributes of the two
// AIE dialect fma ops is the start field.
static void generateMulOrFMAOp(Operation *Op, AIEOpAttributes &opAttr,
                               VectState *state, std::string nextStart = "") {
  // Assert that we computed the attributes for both the operands
  assert(opAttr.start.size() == opAttr.offset.size() &&
         opAttr.start.size() == 2);

  // Set insertion point of the AIE dialect mul/fma op
  state->builder.setInsertionPointAfter(Op);

  // Return true if any user of this op is not mul/fma op
  auto notMulOrFMAOp = [&](Operation *op) {
    return !isa<MulIOp, MulFOp, vector::FMAOp>(op);
  };

  // Generate an AIE dialect mul/fma op from a vector dialect mul/fma op
  auto genOp = [&](Operation *Op, AIEOpAttributes &opAttr, VectState *state,
                   bool i8xi8_pairedOp = false) {
    Operation *repOp = nullptr;
    // Create aievec::FMAOp corresponding to the vector::FMAOp
    if (vector::FMAOp fmaOp = dyn_cast<vector::FMAOp>(Op))
      repOp = generateFMAOp(fmaOp, opAttr, state, i8xi8_pairedOp);
    // Create aievec::MulOp corresponding to the vector::MulIOp
    else if (MulIOp mulOp = dyn_cast<MulIOp>(Op))
      repOp = generateMulOp<MulIOp>(mulOp, opAttr, state);
    // Create aievec::MulOp corresponding to the vector::MulFOp
    else if (MulFOp mulOp = dyn_cast<MulFOp>(Op))
      repOp = generateMulOp<MulFOp>(mulOp, opAttr, state);
    else
      llvm_unreachable("Operation not mul/fma op");
    return repOp;
  };

  Operation *repOp = genOp(Op, opAttr, state);
  LLVM_DEBUG(llvm::dbgs() << "\n\nGenerated AIE dialect mul/fma op " << *repOp);

  // i8xi8 scheme generates two AIE dialect mul/fma ops for each vector dialect
  // mul/fma op. Generate the paired mul/fma op if nextStart is not empty.
  if (!nextStart.empty()) {
    opAttr.start[1] = nextStart;
    Operation *pairedOp = genOp(Op, opAttr, state, true);
    LLVM_DEBUG(llvm::dbgs() << "\n\nGenerated the paired AIE dialect "
                            << "mul/fma op for 8x8 scheme " << *repOp);
    // Link the two mul/fma ops
    assert(!state->pairedOp.count(repOp));
    state->pairedOp[repOp] = pairedOp;
    // If any of the uses of incoming op is not a mul/fma op, then we need to
    // concatenate the paired ops and generate a v16xi8 vector.
    if (llvm::any_of(Op->getUsers(), notMulOrFMAOp))
      repOp = concatAndInterleave_i8xi8(repOp, pairedOp, state, Op->getLoc());
  }

  // Replace all the uses of the vector mul/fma op with the AIE mul/fma op, and
  // remove vector op from the IR.
  Op->replaceAllUsesWith(repOp);
  Op->erase();
}

// Compute the start and offset for xbuff/zbuff for 32x32 scheme.
static void computeBuffAttr_i32xi32(
    unsigned vecSize, // #lanes
    int32_t start,    // start in AIE vec
    int32_t accIncr,  // access change with each loop increment
    AIEOpAttributes &opAttr) {
  // Populate start
  std::string startStr = std::to_string(start);
  // Compute the offset resembling "0x76543210"
  std::string offsetStr = "0x";
  for (int i = vecSize - 1; i >= 0; --i)
    offsetStr.push_back(getHexValue(i * accIncr));

  // And now we have everything to push into opAttr
  opAttr.start.push_back(startStr);
  opAttr.offset.push_back(offsetStr);
  opAttr.offset_hi.push_back("");
  opAttr.square.push_back("");
  opAttr.step.push_back("");
}

// Compute the start, lo/hi offset, and square for xbuff for 16x16 scheme.
static void computeXbuffAttr_i16xi16(
    unsigned vecSize,  // #lanes
    int32_t start,     // computed start in AIE vec
    int32_t accIncr,   // access change with each loop increment
    int32_t colOffset, // xbuff access distance between vector cols
    AIEOpAttributes &opAttr) {
  // The colOffset must be either <=1, or a multiple of 2
  assert(colOffset >= -1 && (colOffset <= 1 || colOffset % 2 == 0) &&
         "cannot compute offset and square for xbuff");
  // We can only generate the offsets and square if either accIncr or column
  // offset is <= 1.
  assert((accIncr <= 1 || colOffset <= 1) &&
         "cannot generate offset and square for xbuff");

  std::string startStr, offsetStr, offsetHiStr, squareStr;

  // Arch restriction: xstart should be a multiple of 2.
  int32_t m2start = (start / 2) * 2;
  startStr = std::to_string(m2start);
  // m2Offset accounts for the extra 1 if the start is not a multiple of 2
  int32_t m2Offset = start - m2start;

  // Compute hi and lo offsets to something resembling "0x_7_6_5_4" and
  // "0x_3_2_1_0" respectively. The '_' are 0 if colOffset is 1.
  offsetStr = "0x";
  int32_t offset = std::max(colOffset, accIncr);
  for (int i = vecSize / 2 - 2; i >= 0; i -= 2) {
    offsetStr.push_back(offset <= 1 ? '0' : getHexValue((offset - 2) / 2));
    offsetStr.push_back(getHexValue((i * accIncr) / 2));
  }
  offsetHiStr = "0x";
  for (int i = vecSize - 2, e = vecSize / 2; i >= e; i -= 2) {
    offsetHiStr.push_back(offset <= 1 ? '0' : getHexValue((offset - 2) / 2));
    offsetHiStr.push_back(getHexValue((i * accIncr) / 2));
  }

  // Now compute the square for xbuff.
  int32_t cstep = std::min(2, std::abs(colOffset));
  int32_t astep = std::min(2, accIncr);
  assert(m2Offset == 0 || (astep <= 1 && cstep <= 1));

  SmallVector<int32_t> sqPattern = {astep + cstep, astep, cstep, 0};
  squareStr = "0x";
  for (auto sq : sqPattern)
    squareStr.push_back(getHexValue(sq + m2Offset));

  // And now we have everything to push into opAttr
  opAttr.start.push_back(startStr);
  opAttr.offset.push_back(offsetStr);
  opAttr.offset_hi.push_back(offsetHiStr);
  opAttr.square.push_back(squareStr);
  opAttr.step.push_back("");
}

// Compute the start, lo/hi offset, and step for zbuff for 16x16 scheme.
static void computeZbuffAttr_i16xi16(
    unsigned vecSize,   // #lanes
    int32_t start,      // computed start in AIE vec
    int32_t accIncr,    // access change with each loop increment
    int32_t zeroOffset, // offset of 0 value in the filter
    int32_t colOffset,  // zbuff access distance between vector cols
    AIEOpAttributes &opAttr) {
  std::string startStr, offsetStr, offsetHiStr, stepStr;
  // zstart must be 4b value.
  assert(start < 16 && "zstart must be 4b value");
  startStr = std::to_string(start);

  // If zbuff comes from splat, use default offsets
  if (accIncr == 0)
    offsetStr = offsetHiStr = "0";
  else {
    // Compute hi and lo offsets using general scheme
    offsetStr = "0x";
    for (int i = vecSize / 2 - 1; i >= 0; --i)
      offsetStr.push_back(getHexValue(i * accIncr));
    offsetHiStr = "0x";
    for (auto i = vecSize - 1, e = vecSize / 2; i >= e; --i)
      offsetStr.push_back(getHexValue(i * accIncr));
  }

  // Compute step between columns
  int32_t step = colOffset == -1 ? zeroOffset - 1 - start : colOffset;
  assert(step >= 0 && "zstep cannot be negative");
  stepStr = std::to_string(step);

  // And now we have everything to push into opAttr
  opAttr.start.push_back(startStr);
  opAttr.offset.push_back(offsetStr);
  opAttr.offset_hi.push_back(offsetHiStr);
  opAttr.square.push_back("");
  opAttr.step.push_back(stepStr);
}

// Compute the start, offset, square, and step for xbuff for 8x8 scheme. This
// is the data scheme, but since is is so restricted, we do a switcharoo, and
// use filter as xbuff. We assume that the filter elements are duplicated
// (duplication factor= 2). For example, the 2x2 filter should be
// {0,0,1,1,2,2,3,3}.
static void computeXbuffAttr_i8xi8(
    unsigned vecSize,  // #lanes
    int32_t start,     // computed start in AIE vec
    int32_t colOffset, // xbuff access distance between vector cols
    AIEOpAttributes &opAttr) {
  // Assert that colStep is a multiple of 4, where colStep is the difference
  // between idx[i][j] and idx[i][j+2].
  assert(
      colOffset >= 2 &&
      "each filter entry must be replicated at least twice for i8xi8 scheme");
  int32_t colStep = 2 * colOffset;
  assert(colStep % 4 == 0 && "xstep must be multiple of 4");

  std::string startStr, offsetStr, squareStr, stepStr;
  // Arch restriction: xstart must be a multiple of 4
  int32_t m4start = (start / 4) * 4;
  startStr = std::to_string(m4start);
  // m4Offset accounts for the excess if start is not a multiple of 4
  int32_t m4Offset = start - m4start;
  // Because of duplication, m4Offset can only be 0 or 2
  assert(m4Offset == 0 || m4Offset == 2);

  // Compute offsetStr to something resembling "0x_0_0_0_0", where _ is
  // (colStep-4)/4.
  offsetStr = "0x";
  for (int i = vecSize / 4 - 1; i >= 0; --i) {
    offsetStr.push_back(getHexValue(colStep / 4 - 1));
    offsetStr += "0";
  }
  stepStr = std::to_string(colStep);

  // Now compute the square for zbuff. We want a {0,x,0,x} pattern.
  int32_t offsetWithoutDup = colOffset / 2;
  int32_t rstep = offsetWithoutDup >= 2 ? 2
                  : colOffset == -1     ? 1
                                        : offsetWithoutDup;
  assert(m4Offset == 0 || rstep <= 1);

  SmallVector<int32_t> sqPattern = {rstep, 0, rstep, 0};
  squareStr = "0x";
  for (auto sq : sqPattern)
    squareStr.push_back(getHexValue(sq + m4Offset));

  // And now we have everything to push into opAttr
  opAttr.start.push_back(startStr);
  opAttr.offset.push_back(offsetStr);
  opAttr.offset_hi.push_back("");
  opAttr.square.push_back(squareStr);
  opAttr.step.push_back(stepStr);
}

// Compute the start, offset, square, and step for zbuff for 8x8 scheme. This
// is the coefficient scheme, but since the coefficient scheme is more relaxed,
// we use image as zbuff.
static void computeZbuffAttr_i8xi8(
    unsigned vecSize,  // #lanes
    int32_t start,     // computed start in AIE vec
    int32_t accIncr,   // access change with each loop increment
    int32_t colOffset, // zbuff access distance between vector cols
    AIEOpAttributes &opAttr, std::string &nextStart) {
  // The colOffset must be either <=1, or a multiple of 2
  assert((colOffset <= 1 || colOffset % 2 == 0) && "zbuff value not supported");

  std::string startStr, offsetStr, squareStr, stepStr;
  // Arch restriction: zstart is a multiple of 2
  int32_t m2start = (start / 2) * 2;
  startStr = std::to_string(m2start);
  // m2Offset accounts for the extra 1 if the start is not a multiple of 2
  int32_t m2Offset = start - m2start;

  // Compute offsetStr to something resembling "0x43322110". The usual pattern
  // is "0x_3_2_1_0", and the purpose is to fill the "_".
  offsetStr = "0x";
  for (int i = vecSize / 4 - 1; i >= 0; --i) {
    int32_t val = i * accIncr + (colOffset + 1) / 2;
    offsetStr.push_back(getHexValue(val));
    offsetStr.push_back(getHexValue(i * accIncr));
  }
  stepStr = std::to_string(2 * std::abs(colOffset));
  nextStart = std::to_string(m2start + 2 * accIncr * (vecSize / 4));

  // Now compute the square for zbuff. We want a {0,1+x,y,y+1+x} pattern, where
  // x is the square offset, and y is the accIncr.
  int32_t rstep = colOffset >= 2 ? 2 : std::abs(colOffset);
  assert(m2Offset == 0 || rstep <= 1);

  SmallVector<int32_t> sqPattern = {accIncr + rstep, accIncr, rstep, 0};
  squareStr = "0x";
  for (auto sq : sqPattern)
    squareStr.push_back(getHexValue(sq + m2Offset));

  // And now we have everything to push into opAttr
  opAttr.start.push_back(startStr);
  opAttr.offset.push_back(offsetStr);
  opAttr.offset_hi.push_back("");
  opAttr.square.push_back(squareStr);
  opAttr.step.push_back(stepStr);
}

// Find a length-k chain of FMA ops such that (1) the chain is linear; (2) the
// operand datawidth is 16 or 8 bits; (3) the access distance between lhs (rhs)
// operands of both FMAs is compile-time constant. These FMAs will be fused
// into a single FMA. Technically, k is equal to the number of columns in the
// FMA topology. If fused, cache the pair indicating the access difference
// between the operands for the two FMAs.
static void fuseFMAOps(Operation *refOp,
                       llvm::SmallSet<Operation *, 8> &fusedOpSet, int32_t cols,
                       VectState *state) {
  // The number of columns must be greater than 1. refOp must be mul/fma op,
  // and should not be covered by the simple vector scheme.
  if (cols <= 1 || !isa<MulIOp, MulFOp, vector::FMAOp>(refOp) ||
      isSimpleVectIntrinsic(refOp, state))
    return;

  // Get the start offsets for left and right operands of the reference
  // operator, i.e., start of the fusion chain.
  Operation *lOp = refOp->getOperand(0).getDefiningOp();
  Operation *rOp = refOp->getOperand(1).getDefiningOp();
  int32_t lstart = computeStartInAIEVec(lOp, state);
  int32_t rstart = computeStartInAIEVec(rOp, state);

  // The xbuff and zbuff offsets between the fused FMA ops. The default value
  // is -1
  int xOffset = -1, zOffset = -1;

  // We write a loop that tries to chase a linear chain of length col-1
  // starting at reference mul/fma op refOp. Let us consider a computational
  // chain of length 3 : {c = A[i]*B[i]; c += A[i+1]*B[i+1]; c +=
  // A[i+2]*B[i+2]}; We represent the start for each instruction as a pair
  // (lhs-operand-start, rhs-operand-start). The starts for the chain will be
  // {(0,0), (1,1), (2,2)}.  Since the consecutive starts are equidistant in
  // the chain, we consider this chain fusable, and cache the fused operations
  // in fusedp vector.
  Operation *curOp = refOp;
  SmallVector<Operation *, 8> fusedOps;

  for (auto len = 0; len < cols - 1; ++len) {
    // If this operation has more than one use, break loop.
    if (!curOp->hasOneUse())
      break;
    // Get the consumer of the curOp FMA
    Operation *usrOp = *curOp->getUsers().begin();
    // The user/consumer user operation must be a FMA, belonging to the same
    // basic block as curOp, and must not be covered by simple scheme.
    if (!isa<vector::FMAOp>(usrOp) || curOp->getBlock() != usrOp->getBlock() ||
        isSimpleVectIntrinsic(usrOp, state))
      break;
    // Both curOp and usrOp must be either fma or fmsub(msc)
    if (isa<vector::FMAOp>(curOp) &&
        state->mscOps.count(curOp) != state->mscOps.count(usrOp))
      break;
    // Compute the start/access distance for each operand of curOp and usrOp
    SmallVector<int32_t, 2> offsets;
    for (size_t idx = 0; idx < 2; ++idx) {
      // Get the vector attributes for this operand of curOp and usrOp
      AIEVecAttributes cstat = getOperandVecStats(curOp, state, idx);
      AIEVecAttributes ustat = getOperandVecStats(usrOp, state, idx);
      // We need to ensure that the accesses to this operand of curOp and usrOp
      // come from the same vector. To guarantee this, we peform two checks:
      // Check 1. The accesses must be similar
      if (cstat.vecSizeInBits != ustat.vecSizeInBits ||
          cstat.elementSizeInBits != ustat.elementSizeInBits ||
          cstat.loadFromMemory != ustat.loadFromMemory ||
          cstat.isSplat != ustat.isSplat)
        break;
      // Check 2. The accesses must come from the same vector/upd op
      Operation *cdefOp = curOp->getOperand(idx).getDefiningOp();
      Operation *udefOp = usrOp->getOperand(idx).getDefiningOp();
      bool related = cdefOp == udefOp;
      if (!related && cstat.loadFromMemory && ustat.loadFromMemory) {
        IntervalReuse *civ = state->getIntervalForOperation(cdefOp);
        IntervalReuse *uiv = state->getIntervalForOperation(udefOp);
        related =
            civ == uiv && civ->getInterval(cdefOp) == uiv->getInterval(udefOp);
      }
      if (!related)
        break;

      // We know that the accesses to this operand for both curOp and usrOp
      // come from the same AIE vector. So we can get the start value for the
      // operands.
      int32_t start1 = computeStartInAIEVec(cdefOp, state);
      int32_t start2 = computeStartInAIEVec(udefOp, state);
      int32_t offset = start2 - start1;
      // perform a set of checks to make sure that the distance can be encoded
      // in AIE intrinsic.
      // Check 1: the offset should be positive
      if (offset < 0)
        break;
      // Check 2: If offset is greater than 1, it should be a multiple of 2
      if (offset > 1 && offset % 2 != 0)
        break;
      // Check 3: If offset is >=2, then the reference op must have start=0
      int32_t refStart = idx == 0 ? lstart : rstart;
      if (!ustat.isSplat && offset > 1 && refStart != 0)
        break;
      // From this operand's perspective, we can fuse this usrOp with curOp.
      // Cache the start offset.
      offsets.push_back(offset);
    }
    // Verify that we computed offset for both operands
    if (offsets.size() < 2)
      break;
    // Ensure that the difference between consecutive xOffsets and zOffsets is
    // consistent throughout the chain.
    if ((xOffset != -1 && xOffset != offsets[0]) ||
        (zOffset != -1 && zOffset != offsets[1]))
      break;
    // Now the user FMA op can be fused with refOp
    xOffset = offsets[0];
    zOffset = offsets[1];
    fusedOps.push_back(usrOp);
    // usrOp now becomes curOp, so that we can chase the linear chain starting
    // at it.
    curOp = usrOp;
  }

  // If there are no ops fused, return
  if (fusedOps.empty())
    return;

  LLVM_DEBUG(llvm::dbgs() << "\n\nFused following fma ops with op " << *refOp);

  // If we reached here, we have decided to fuse a linear chain of FMAs, we
  // need to remove the fused FMAs from the IR.
  for (auto &op : fusedOps) {
    LLVM_DEBUG(llvm::dbgs() << "\n\tfma op " << *op);
    fusedOpSet.insert(op);
    // Since we are fusing op with refOp, fuse their access extents too
    fuseAccessExtent(refOp, op, state);
    // Now replace the uses of op with reference
    op->replaceAllUsesWith(refOp);
  }

  // Cache the column offsets for refOp
  assert(!state->opToColOffsets.count(refOp));
  state->opToColOffsets[refOp] = std::make_pair(xOffset, zOffset);
}

// Compute all the attributes for xbuff, based on the scheme.
static void computeXbuffAttributes(
    Scheme &scheme,    // vect scheme info
    int32_t start,     // computed start in AIE vec
    int32_t colOffset, // xbuff access distance between vector cols
    int32_t accIncr,   // xbuff access incr with each loop increment
    int32_t dupFactor, // duplication factor for i8xi8 filter
    AIEOpAttributes &opAttr) {
  // Branch to different schemes
  // Case 1: 32x32 real
  if (scheme.lanes == 8 && scheme.cols == 1 && scheme.xbits == 32 &&
      scheme.zbits == 32)
    computeBuffAttr_i32xi32(scheme.lanes, start, accIncr, opAttr);
  // Case 2: 16x16 real
  else if (scheme.lanes == 16 && scheme.cols == 2 && scheme.xbits == 16 &&
           scheme.zbits == 16) {
    // We only support a loop increment of <= 1
    assert((accIncr <= 1 || accIncr % 2 == 0) &&
           "loop step size value not supported");
    computeXbuffAttr_i16xi16(scheme.lanes, start, accIncr, colOffset, opAttr);
  }
  // Case 3: 8x8 real
  else if (scheme.lanes == 16 && scheme.cols == 8 && scheme.xbits == 8 &&
           scheme.zbits == 8) {
    // We only support a loop increment of <= 1
    assert(accIncr <= 1 && "loop step size greater than 1 not supported");
    // If we were not able to fuse any of the macs to exploit column topology,
    // then colOffset must be equal to dupFactor.
    if (colOffset == -1)
      colOffset = dupFactor;
    computeXbuffAttr_i8xi8(scheme.lanes, start, colOffset, opAttr);
  } else
    llvm_unreachable("Unsupported vectorization scheme");
}

// Compute all the attributes for zbuff, based on the scheme.
static void computeZbuffAttributes(
    Scheme &scheme,         // vect scheme info
    int32_t start,          // computed start in AIE vec
    int32_t colOffset,      // zbuff access distance between vector cols
    int32_t accIncr,        // zbuff access incr with each loop increment
    int32_t zeroOffset,     // zero offset of filter for i16xi16 scheme
    std::string &nextStart, // start of mul/mac pair in i8xi8 scheme
    AIEOpAttributes &opAttr) {
  // Branch to different schemes
  // Case 1: 32x32 real
  if (scheme.lanes == 8 && scheme.cols == 1 && scheme.xbits == 32 &&
      scheme.zbits == 32)
    computeBuffAttr_i32xi32(scheme.lanes, start, accIncr, opAttr);
  // Case 2: 16x16 real
  else if (scheme.lanes == 16 && scheme.cols == 2 && scheme.xbits == 16 &&
           scheme.zbits == 16) {
    // We only support a loop increment of <= 1
    assert(accIncr <= 1 && "loop step size greater than 1 not supported");
    // Get the zero offset in filter if the user provided it in the command
    // line. The zero offset is cyclic, so compute an offset that is > start.
    zeroOffset = zeroOffset == 0 ? scheme.lanes
                                 : start + zeroOffset - (start % zeroOffset);
    computeZbuffAttr_i16xi16(scheme.lanes, start, accIncr, zeroOffset,
                             colOffset, opAttr);
  }
  // Case 3: 8x8 real
  else if (scheme.lanes == 16 && scheme.cols == 8 && scheme.xbits == 8 &&
           scheme.zbits == 8) {
    // We only support a loop increment of <= 1
    assert(accIncr <= 1 && "loop step size greater than 1 not supported");
    computeZbuffAttr_i8xi8(scheme.lanes, start, accIncr, colOffset, opAttr,
                           nextStart);
  } else
    llvm_unreachable("Unsupported vectorization scheme");
}

// For this mul/FMA operator, generate AIE dialect mul/FMA op based on
// different vector schemes.
static void generateSchemeBasedMulOrFMAOp(Operation *Op, VectState *state) {
  // First check if this operation requires simple vector operation, and not an
  // advanced scheme.
  if (isSimpleVectIntrinsic(Op, state)) {
    // opAttr will cache the attributes (start, step, offsets, square, etc.)
    // for both lhs and rhs operands.
    AIEOpAttributes opAttr;
    // For simple scheme, we do not need any attribute
    for (size_t idx = 0; idx < 2; ++idx) {
      opAttr.start.push_back("");
      opAttr.offset.push_back("");
      opAttr.offset_hi.push_back("");
      opAttr.square.push_back("");
      opAttr.step.push_back("");
    }
    generateMulOrFMAOp(Op, opAttr, state);
    return;
  }

  // Otherwise generate mul or fma op based on advanced scheme. Get the rows,
  // cols, and datatype size for the vector scheme, and pack all that
  // information in the Scheme struct.
  int32_t lanes, cols;
  std::tie(lanes, cols) = getNumRowsAndCols(Op);
  // Get the data sizes for left and right operands of mul/fma
  int32_t xbits =
      getElementSizeInBits(Op->getOperand(0).getType().cast<VectorType>());
  int32_t zbits =
      getElementSizeInBits(Op->getOperand(1).getType().cast<VectorType>());
  Scheme scheme(lanes, cols, xbits, zbits);

  // If element size is < 32 bits ,we can fuse multiple FMAs together to
  // exploit the column topology of FMA intrinsic.
  auto colOffset = state->opToColOffsets.count(Op) ? state->opToColOffsets[Op]
                                                   : std::make_pair(-1, -1);

  // opAttr will cache the step,offsets,square, etc. for both lhs and rhs
  // operands.
  AIEOpAttributes opAttr;
  // For i8xi8 scheme, each vector dialect mul/fma op is converted to two AIE
  // dialect mul/fma op. The two AIE ops are identical, except for the start
  // field. nextStart indicates the start of the second op.
  std::string nextStart;
  // Compute relevant attributes (start, offsets, step, square, etc.) for each
  // operand, and store them in opAttr.
  for (size_t idx = 0; idx < 2; ++idx) {
    AIEVecAttributes stat = getOperandVecStats(Op, state, idx);
    Operation *op = Op->getOperand(idx).getDefiningOp();
    int32_t start = 0, accIncr = 1;
    // If the operand comes from transfer_read, compute the step and start
    // values.
    if (stat.loadFromMemory) {
      auto readOp = cast<TransferReadOp>(op);
      // How does the access change with each iteration of the vectorized loop?
      accIncr = stat.isSplat ? 0 : computeVecorizedLoopStepSize(readOp, state);
      // start in the AIE vector
      start = computeStartInAIEVec(op, state);
    }
    // Compute the xbuff and zbuff attributes
    if (idx == 0)
      computeXbuffAttributes(scheme, start, colOffset.first, accIncr,
                             state->dupFactor, opAttr);
    else
      computeZbuffAttributes(scheme, start, colOffset.second, accIncr,
                             state->zeroOffset, nextStart, opAttr);
  }
  // And now generate the mul/fma op
  generateMulOrFMAOp(Op, opAttr, state, nextStart);
}

// If the datatype allows it, fuse a mul or fma op with other fma ops to
// utilize the column topology of the AIE mul/fma intrinsic (e.g., 2 fmas can
// be fused for i16xi16 scheme, and 8 for i8xi8 scheme).
static void fuseFMAOpsForColumnTopology(func::FuncOp func, VectState *state) {
  // A set of FMA ops that were fused in the column topology
  llvm::SmallSet<Operation *, 8> fusedOpSet;

  // Fuse FMA ops to exploit column topology
  func.walk([&](mlir::Operation *op) {
    if (isa<MulIOp, MulFOp, vector::FMAOp>(op)) {
      // Only process fma ops that are not already fused with another mul/fma
      if (!fusedOpSet.count(op)) {
        // Get the rows and columns for this topology
        int32_t lanes, cols;
        std::tie(lanes, cols) = getNumRowsAndCols(op);
        // Try fusing a linear chain of FMA ops (max length = cols) starting at
        // op.
        fuseFMAOps(op, fusedOpSet, cols, state);
      }
    }
  });

  // Remove all the ops that were fused with other FMAs
  for (auto op : fusedOpSet)
    op->erase();
}

// Generate the AIE mul/fma op for each vector mul/fma op. This function is the
// crux of AIE vectorization. It accomplishes two main tasks: (1) For each
// mul/fma operation, compute the operand attributes. The attributes are start,
// offsets, square, step, etc. based on the scheme; and (2) Once all the
// attributes are computed, generate appropriate mul/fma operation in AIE
// dialect.
static void generateAIEMulOrFMAOpsInFunc(func::FuncOp func, VectState *state) {
  // For each mul/fma op, compute the scheme-dependent operand attributes, and
  // generate corresponding AIE dialect ops.
  func.walk([&](mlir::Operation *op) {
    if (isa<MulIOp, MulFOp, vector::FMAOp>(op))
      generateSchemeBasedMulOrFMAOp(op, state);
  });
}

// Given the operation attributes (start, offset, square, etc.), generate an
// AIE add/sub op for the incoming vector add/sub Op.
static void generateAddOrSubOp(Operation *Op, AIEOpAttributes &opAttr,
                               VectState *state) {

  // Set insertion point of the AIE dialect mul/fma op
  state->builder.setInsertionPointAfter(Op);

  // Generate an AIE dialect add/sub op
  Operation *repOp = nullptr;
  if (isa<SubIOp, SubFOp>(Op)) {
    repOp = generateSubOp(Op, opAttr, state);
    LLVM_DEBUG(llvm::dbgs() << "\n\nGenerated AIE dialect sub op " << *repOp);
  } else {
    repOp = generateAddOp(Op, opAttr, state);
    LLVM_DEBUG(llvm::dbgs() << "\n\nGenerated AIE dialect sub op " << *repOp);
  }

  // Replace all the uses of the vector add/sub op with the AIE add/sub op, and
  // remove Op from the IR.
  Op->replaceAllUsesWith(repOp);
  Op->erase();
}

// For this add/sub operator, generate AIE dialect add/sub op based on
// different vector schemes.
static void generateSchemeBasedAddOrSubOp(Operation *Op, VectState *state) {
  // opAttr will cache the attributes (start, offsets, square, etc.) for both
  // lhs and rhs operands.
  AIEOpAttributes opAttr;

  // First check if this operation requires simple vector operation, and not an
  // advanced scheme.
  if (isSimpleVectIntrinsic(Op, state)) {
    // For simple scheme, we do not need any attribute
    for (size_t idx = 0; idx < 2; ++idx) {
      opAttr.start.push_back("");
      opAttr.offset.push_back("");
      opAttr.offset_hi.push_back("");
      opAttr.square.push_back("");
    }
    generateAddOrSubOp(Op, opAttr, state);
    return;
  }

  // Otherwise generate add/sub op based on advanced scheme.
  // Compute relevant attributes (start, offsets, square, etc.) for each
  // operand, and store them in opAttr.
  for (size_t idx = 0; idx < 2; ++idx) {
    AIEVecAttributes stat = getOperandVecStats(Op, state, idx);
    assert(stat.elementSizeInBits >= 16 &&
           "advanced scheme for add op on int8 data type not supported");

    int32_t start = 0, accIncr = 1;
    std::string startStr;
    std::string offsetStr, offsetHiStr;
    std::string squareStr;

    // If the operand comes from transfer_read, compute the loop step and start
    // values.
    if (stat.loadFromMemory) {
      Operation *op = Op->getOperand(idx).getDefiningOp();
      auto readOp = cast<TransferReadOp>(op);
      // How does the access change with each iteration of the vectorized loop?
      accIncr = stat.isSplat ? 0 : computeVecorizedLoopStepSize(readOp, state);
      // start in the AIE vector
      start = computeStartInAIEVec(op, state);
    }
    // Now the usual processing. For i32 datatype, use the regular lane
    // selection.
    if (stat.elementSizeInBits == 32) {
      startStr = std::to_string(start);
      offsetStr = "0x";
      for (int i = 7; i >= 0; --i)
        offsetStr.push_back(getHexValue(i * accIncr));
      // If there are >8 lanes, we need to compute offset_hi
      if (stat.lanes > 8) {
        assert(stat.lanes == 16 && "Cannot generate offset for add/sub op");
        // Cannot have loop stride > 1
        assert(accIncr <= 1 && "Cannot generate offset for given loop stride");
        offsetHiStr = "0x";
        for (int i = 15; i >= 8; --i)
          offsetStr.push_back(getHexValue(i * accIncr));
      }
    } else if (stat.elementSizeInBits == 16) {
      assert(accIncr <= 1 && "cannot generate offset for given loop stride");
      // start must be a multiple of 2 for i16 data type
      int32_t m2Offset = start % 2;
      startStr = std::to_string(start - m2Offset);
      // We must compute the offset and offset_hi only if the access is not
      // splat. For splat, we can use trivial offsets.
      if (accIncr == 0)
        offsetStr = offsetHiStr = "0";
      else {
        offsetStr = "0x";
        for (int i = 6; i >= 0; i -= 2) {
          offsetStr.push_back('0');
          offsetStr.push_back(getHexValue((i * accIncr) / 2));
        }
        offsetHiStr = "0x";
        for (int i = 14; i >= 8; i -= 2) {
          offsetHiStr.push_back('0');
          offsetHiStr.push_back(getHexValue((i * accIncr) / 2));
        }
      }
      // We use a simplistic square that covers only two cases: access is
      // splat, and access is regular with stride that's power of 2.
      if (m2Offset == 0 && accIncr == 0)
        squareStr = "0";
      else {
        assert(m2Offset == 0 || accIncr == 0);
        squareStr = "0x";
        int32_t astep = std::min(1, accIncr);
        SmallVector<int32_t> sqPattern = {3 * astep, 2 * astep, astep, 0};
        for (auto sq : sqPattern)
          squareStr.push_back(getHexValue(sq + m2Offset));
      }
    } else
      llvm_unreachable("Cannot generate advanced add op for given datatype");

    // We have computed all the fields. Cache the attributes.
    opAttr.start.push_back(startStr);
    opAttr.offset.push_back(offsetStr);
    opAttr.offset_hi.push_back(offsetHiStr);
    opAttr.square.push_back(squareStr);
  }
  // And now generate the add/sub op
  generateAddOrSubOp(Op, opAttr, state);
}

// The main focus of this function is to compute the right start/offset fields
// for the adds involving splat. If none of the operands of the add op is
// splat, we must generate simple scheme add op.
static void generateAIEAddOrSubOpsInFunc(func::FuncOp func, VectState *state) {
  func.walk([&](mlir::Operation *op) {
    if (isa<AddIOp, AddFOp, SubIOp, SubFOp>(op))
      generateSchemeBasedAddOrSubOp(op, state);
  });
}

// Generate UPD ops to subsume all the transfer_read ops of affine dialect. To
// generate the UPD ops, we first visit the innermost for op, and for each
// transfer_read instruction nested inside that op, create a set of UPD ops,
// and then insert them in the front bb of that for op's region.
static void insertUPDOpsInLoop(AffineForOp forOp, VectState *state) {
  // Recursively generate UPD ops in the nested for op's.
  for (AffineForOp nestedOp : forOp.region().getOps<AffineForOp>())
    insertUPDOpsInLoop(nestedOp, state);

  // A map from an interval to the UPD op. The key gives the interval that
  // should be loaded into the AIE vec, and the value indicates the UPD op
  // achieving that. The value also has an 8-bit field, whose first/second bit
  // is set if upd op idx=0/idx=1 is already created for this interval.
  DenseMap<std::tuple<IntervalReuse *, int32_t, int32_t>,
           std::pair<aievec::UPDOp, int8_t>>
      memToUpdMap;
  // A map from a read operation to its corresponding UPD operation. The idea
  // is that multiple read ops will derive from the same bigger vector
  // register.
  DenseMap<Operation *, aievec::UPDOp> readOpToUpdMap;
  // Iterate over all the transfer_read ops within this loop
  Region &region = forOp.region();
  for (TransferReadOp readOp : region.getOps<TransferReadOp>()) {
    aievec::UPDOp updOp = generateUPDOp(readOp, memToUpdMap, region, state);
    readOpToUpdMap[readOp] = updOp;
  }

  // Now replace all the uses of a transfer_read op with its UPD op
  for (auto &map : readOpToUpdMap) {
    Operation *op = map.first;
    op->replaceAllUsesWith(map.second);
    op->erase();
  }
}

// Replace all the transfer_read ops with UPD ops in the function.
static void insertUPDOpsInFunc(func::FuncOp func, VectState *state) {
  for (AffineForOp forOp : func.getOps<AffineForOp>()) {
    insertUPDOpsInLoop(forOp, state);
  }
}

// Incoming Op is an operation in AIE dialect whose result is an accumulator.
// Check all its uses, and if any user of Op is a non-AIE operation, insert an
// SRS instruction to move the value from accumulator to vector.
static void insertSRSOp(Operation *Op, VectState *state) {
  // This operation must have at least one use, and at least one result
  if (Op->use_empty() || Op->getNumResults() == 0)
    return;

  // The operation must be in AIE dialect, and its result should be accumulator
  // type.
  assert(isAIEOp(Op) && llvm::any_of(Op->getResultTypes(), isAccType));

  // Check if any user of this operation is a non-AIE op. If any user of this
  // operation is non-AIE op, then we need to generate SRS op to move value
  // from accumulator to vector
  auto isNonAIEOp = [&](Operation *op) { return !isAIEOp(op); };
  if (!llvm::any_of(Op->getUsers(), isNonAIEOp))
    return;

  // Given an accumulator, one can use different srs intrinsic to generate
  // different output types. Create a map from SRS output type to the SRS op.
  DenseMap<Type, aievec::SRSOp> typeToSRSOpMap;

  // Set the insertion point for the AIE dialect SRS op
  state->builder.setInsertionPointAfter(Op);

  // Iterate over all the users of this operation that are not in AIE dialect,
  // and replace the use of Op in them with srsOp
  for (auto user : Op->getUsers()) {
    // Skip AIE ops
    if (isAIEOp(user))
      continue;

    // Get the underlying scalar element type of user op. If the user is a
    // write op, it won't have a result. So get the element type from memref.
    Type scalarType = nullptr;
    if (auto writeOp = dyn_cast<TransferWriteOp>(user)) {
      // Get the element type from the memref output
      MemRefType memRefType = writeOp.getSource().getType().cast<MemRefType>();
      scalarType = memRefType.getElementType();
    } else
      scalarType = getElementTypeOrSelf(*user->getResultTypes().begin());
    assert(scalarType && "failed to form SRS op");
    // Iterate over all the operands of this user, and find the ones that
    // correspond to the Op.
    for (auto operand : user->getOperands()) {
      if (operand.getDefiningOp() == Op) {
        aievec::SRSOp srsOp = nullptr;
        if (!typeToSRSOpMap.count(scalarType)) {
          srsOp =
              generateSRSOp(Op->getResult(0), scalarType, state, Op->getLoc());
          LLVM_DEBUG(llvm::dbgs() << "\n\nCreated SRS op " << srsOp
                                  << " for the acc output of operation " << Op);
          typeToSRSOpMap[scalarType] = srsOp;
        } else
          srsOp = typeToSRSOpMap[scalarType];
        assert(srsOp && "Failed to create SRS intrinsic");
        // And now we replace the operand with srsOp
        user->replaceUsesOfWith(operand, srsOp);
      }
    }
  }
}

// Generate SRS op whenever we move data from an accumulator AIE dialect to a
// vector.
static void insertSRSOpsInFunc(func::FuncOp func, VectState *state) {
  func.walk([&](mlir::Operation *op) {
    // Check if we need to insert an SRS op if (1) the op is in AIE dialect,
    // and (2) the result of the op is accumulator type.
    if (isAIEOp(op) && llvm::any_of(op->getResultTypes(), isAccType))
      insertSRSOp(op, state);
  });
}

// Set existing read/write op to in-bounds, indicating that it always reads
// from/writes to a full buffer. We make this assumption for our vectorization
// framework.
template <typename TransferOp> static void setInBounds(TransferOp op) {
  if (op.getTransferRank() == 0)
    return;
  SmallVector<bool, 4> bools(op.getTransferRank(), true);
  OpBuilder b(op.getContext());
  op->setAttr(op.getInBoundsAttrName(), b.getBoolArrayAttr(bools));
}

// Remove redundant vector load/stores (i.e., transfer ops) that could be
// generated post unolling. The redundant operations are removed in two steps:
// first, we do a store to load forwarding. This removes the loads that
// immediately succeed a store to the same location. Then it removes multiple
// stores to the same memory location without an interfering store to that
// memref. The only preserves the last write. These transformations are already
// implemented in 'transferOpflowOpt' function. But these transformations only
// work on reads/writes that are within bounds. We safely assume that for AIE
// vectorization, all the transfer reads/writes are within bounds.
static void redundantLoadStoreOptimization(ModuleOp module) {
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    // Mark all the transfer ops that have empty in_bounds as inbound
    func.walk([&](Operation *Op) {
      if (auto readOp = dyn_cast<TransferReadOp>(Op)) {
        if (!readOp.getInBounds())
          setInBounds<TransferReadOp>(readOp);
      } else if (auto writeOp = dyn_cast<TransferWriteOp>(Op)) {
        if (!writeOp.getInBounds())
          setInBounds<TransferWriteOp>(writeOp);
      }
    });
    // Now that all the transfer ops are marked inbound, remove redundant
    // vector loads/stores
    transferOpflowOpt(func);
  }
}

// Run a pre pipeline of cleanup passes (canonicalizer). Remove redundant
// load/store operations in case the code was generated via unrolling
static void preCanonicalizeIR(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(createCanonicalizerPass());
  assert(!failed(pm.run(module)));
  redundantLoadStoreOptimization(module);
}

// Run a post pipeline of cleanup and optimization passes (canonicalizer, LICM,
// CSE, etc). At the end, lower the output from affine to scf, so that we can
// use EmitC functionality to generate the loops.
static void postCanonicalizeIR(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createLoopInvariantCodeMotionPass());
  pm.addPass(createLowerAffinePass());
  assert(!failed(pm.run(module)));
}

// Iterate over the loop nestings to form loop nesting bands. Then for each
// block within those bands, the enclosingLoops is set to the loop band.
static void
computeEnclosingLoopsPerBlock(AffineForOp forOp, VectState *state,
                              SmallVector<Operation *, 8> &enclosingLoops) {
  // Form the loop band for nested for ops
  for (AffineForOp nestedOp : forOp.region().getOps<AffineForOp>()) {
    enclosingLoops.push_back(nestedOp);
    computeEnclosingLoopsPerBlock(nestedOp, state, enclosingLoops);
    enclosingLoops.pop_back();
  }

  // Iterate over all the transfer_read operations enclosed within the current
  // region, and store the for loop nesting for the read op.
  for (TransferReadOp readOp : forOp.region().getOps<TransferReadOp>()) {
    // Find the block corresponding to this transfer_read
    Block *block = readOp->getBlock();
    state->blockToEnclosingLoops[block] = enclosingLoops;
  }
}

// We reorder the operands involved in multiplication so that (1) the splat
// operand is always the second operand, and (2) the bigger vector is the first
// operand. This allows us to form FMA intrinsic for AIE. The only exception to
// this rule is the 8x8 bit scheme, where the xbuff is a bit more restrictive,
// so we prefer splat as left operand of multiplication for 8x8 scheme.
static void reassociateMulOpInFunc(func::FuncOp func, VectState *state) {
  func.walk([&](mlir::Operation *op) {
    // Only reassociate vector mul ops that are well formed. This also includes
    // the multiplication component in fma ops.
    if (isa<MulIOp, MulFOp, vector::FMAOp>(op) && isWellFormedVectorOp(op)) {
      // 1. Reassociate so that splat is in the correct place
      reassociateMulOpWithSplat(op, state);
      // 2. Reassociate so that bigger vector is the first operand
      reassociateMulOpBasedOnVecSize(op, state);
    }
  });
}

// This is a very simple function that looks for add op of the form {a=b*c; d =
// a+e;}, and reassociates it so that the operand that computes a mult is the
// right operand of add op. This is a syntactic transformation that uses the
// commutativity of add op, and is only applied so that we can leverage the
// same code functionality for generating mac and msc ops.
static void reassociateAddOpInFunc(func::FuncOp func, VectState *state) {
  func.walk([&](mlir::Operation *op) {
    // Only reassociate vector add ops that are well formed.
    if (isa<AddIOp, AddFOp>(op) && isWellFormedVectorOp(op)) {
      // addOp must have two operands and one result
      assert(op->getNumOperands() == 2 && op->getNumResults() == 1);

      // Determine which operand is the multiply
      Operation *rhsOp = op->getOperand(1).getDefiningOp();
      // If rhs is mul operand, no need to proceed further
      if (!isa<MulIOp, MulFOp>(rhsOp)) {
        Operation *lhsOp = op->getOperand(0).getDefiningOp();
        // If lhs is the mul operand, do the switcharoo
        if (isa<MulIOp, MulFOp>(lhsOp)) {
          LLVM_DEBUG(llvm::dbgs() << "\n\nReassociating addOp " << *op
                                  << " to place mul as rhs operand");
          Value left = op->getOperand(0);
          Value right = op->getOperand(1);
          op->setOperand(0, right);
          op->setOperand(1, left);
          LLVM_DEBUG(llvm::dbgs() << "\n\taddOp after reassociation: " << *op);
        }
      }
    }
  });
}

// For i8xi8 scheme, the lhs operand vector size could be <= 256 bits, but the
// intrinsic requires the lhs operand vector to be at least 512 bits.
// Therefore, we check each read op, and (1) if it only appears in the LHS of a
// mul/fma op, and (2) its interval width is <= 256 bits, we tag the vector
// corresponding to it. Then we can try to coalesce two consecutive tagged
// intervals (i.e., vectors) in each ReuseInterval object. This removes the
// need of extra vector and two concat ops.
static void coalesceLHSOpVectorsInFunc(func::FuncOp func, VectState *state) {
  // Iterate over all the transfer read ops in this function
  func.walk([&](TransferReadOp op) {
    // Iterate over all the users of this read operation. We want to identify
    // if this read op only appears as an LHS operand of a mul/fma op.
    bool onlyLHS = true;
    for (auto user : op->getUsers()) {
      if (!isa<MulIOp, MulFOp, vector::FMAOp>(user) ||
          user->getOperand(0).getDefiningOp() != op) {
        onlyLHS = false;
        break;
      }
    }
    // If this read op only appears as LHS operand of mul/fma op, we find the
    // IntervalReuse object this op belongs to, and tag the interval (i.e.,
    // vector) subsuming this read op's access extent.
    if (onlyLHS) {
      IntervalReuse *iv = state->getIntervalForOperation(op);
      iv->markLHSOperandVec(op);
    }
  });

  // All the tagging is done. Now iterate over all the IntervalReuse objects.
  // If any of those has tagged vector, try to coalesce the tagged vectors.
  for (auto interval : state->reuseIntervals) {
    interval->coalesceIntervals();
  }
}

// For each read operation, compute the potential vector-level data reuse we
// can exploit for it.
static void computeReuse(TransferReadOp readOp, VectState *state) {
  // Construct a linearized access expression for the transfer_read
  AffineExpr linearAccess = constructLinearizedAffineExpr(readOp, state);
  AffineExpr base;
  int32_t offset;
  // Decompose the linear access into a base and constant offset value
  std::tie(base, offset) = getBaseAndOffset(linearAccess);

  // Get the step size of the vectorized loop that encloses this read operation
  int32_t step = computeVecorizedLoopStepSize(readOp, state);

  // If the permutation map is 0, the read operation is splat
  bool isSplat = readOp.getPermutationMap().isConstant();

  // Check if this readOp is the lhs or rhs operand of a mul/fma op. If it is,
  // then the vector size corresponding to its access extent should at least be
  // 256 bits. Otherwise, AIE vectors are at least 128 bits.
  unsigned minVecSize = 128;
  for (auto user : readOp->getUsers()) {
    if (isa<MulIOp, MulFOp, vector::FMAOp>(user)) {
      if (user->getOperand(0).getDefiningOp() == readOp ||
          user->getOperand(1).getDefiningOp() == readOp) {
        minVecSize = 256;
        break;
      }
    }
  }

  bool found = false;
  // Iterate over all the IntervalReuse objects created thus far. Each object
  // represents a group of reads that have a potential of vector-level data
  // reuse. If we find an interval that (1) accesses an array with same base,
  // and (2) has other operations enclosed within the same same set of loops as
  // this operation, then we have the cluster of read ops that this op must be
  // grouped with.
  for (auto interval : state->reuseIntervals) {
    // Check if reuse is discovered
    if (interval->potentialReuse(readOp, base, state->blockToEnclosingLoops)) {
      // If the reuse is found with other operations in interval, add this
      // operation to interval.
      interval->insertInterval(readOp, state->opToIntervalMap, offset, step,
                               isSplat, minVecSize);
      found = true;
      break;
    }
  }
  // If no reuse is found, create a new IntervalReuse object with just this
  // operation's read access extent.
  if (!found) {
    IntervalReuse *iv = new IntervalReuse(readOp, base);
    iv->insertInterval(readOp, state->opToIntervalMap, offset, step, isSplat,
                       minVecSize);
    state->reuseIntervals.push_back(iv);
  }
}

// Compute the reuse interval for all the transfer_read operations. The
// transfer_read operations capture the vector load. Since AIE only allows for
// aligned vector loads, we need to compose multiple transfer reads together to
// form intervals of certain width (128, 256, 512, or 1024), and create an AIE
// vector from each interval.
static void computeReuseInFunc(func::FuncOp func, VectState *state) {
  // First compute the loops surrounding each load/store operation. This is
  // necessary to identify loads/stores that are nested together.
  for (AffineForOp forOp : func.getOps<AffineForOp>()) {
    SmallVector<Operation *, 8> enclosingLoops;
    enclosingLoops.push_back(forOp);
    computeEnclosingLoopsPerBlock(forOp, state, enclosingLoops);
  }

  // Now we can cluster all the transfer_read ops that have a potential of
  // vector-level data reuse.
  func.walk([&](TransferReadOp op) { computeReuse(op, state); });
}

// Rewrite a sequence of mul and add/sub {a = b*c; d = a+e;} as an FMA op {d =
// b*c+e;}. This step only rewrites the FMA op in vector dialect.
static void rewriteFMAOpsInFunc(func::FuncOp func, VectState *state) {
  // Find a root add op that is well formed, and start from there
  func.walk([&](Operation *Op) {
    if (isa<AddIOp, AddFOp, SubIOp, SubFOp>(Op) && isWellFormedVectorOp(Op)) {
      // Perform a series of checks to see if we can find a mul and add/sub
      // that can be fused into a FMA. If found, fuse.
      if (canFuseMulAndAddOrSubIntoFMAOp(Op))
        fuseMulAndAddOrSubIntoFMAOp(Op, state);
    }
  });
}

// Assuming commutativity and associativity of add and mul ops, reassociate ops
// so that code generation becomes feasible/easier.
static void reassociateOpsInFunc(func::FuncOp func, VectState *state) {
  // We assume that pointwise multiplication is commutative. So correct the
  // order of operands involved in multiplication so that we can form AIE
  // mul/fma intrinsic.
  reassociateMulOpInFunc(func, state);
  // We assume that pointwise addition is commutative. If any operand of the
  // add op is a mul op, then we reassociate it to be the right operand of add
  // op. This change ensures that in the next step, when we form FMA ops, we
  // reuse the functionality for mac/msc ops.
  reassociateAddOpInFunc(func, state);
}

struct AIEVectorize : public AIEVectorizeBase<AIEVectorize> {
  AIEVectorize() = default;
  void runOnOperation() override;
};

/// Generate AIE vector intrinsics for the current module. Assumption: the
/// input to this function is the mlir output generated after vectorizing the
/// scalar mlir input with affine superVectorizer. The vectorization factor
/// should be appropriately set to a power of 2 (e.g., 8 for i32xi32 scheme, 16
/// for i16xi16 scheme and i8xi8 scheme).
void AIEVectorize::runOnOperation() {
  // Verify the bounds of the incoming arguments
  assert(shiftParam < 64 && "SRS shift parameter should be between 0 and 63");
  assert(zeroOffset < 128 &&
         "Zero offset in the filter should be between 0 and 127");
  assert(dupFactor < 128 &&
         "Duplicate offset in the filter should be between 0 and 127");

  ModuleOp module = getOperation();

  // Canonicalize the incoming IR, mostly to simplify affine/compose apply ops
  preCanonicalizeIR(module);

  // Iterate over all the functions in this module, and vectorize them
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    // Create a new global state
    VectState *state =
        new VectState(func.getContext(), shiftParam, zeroOffset, dupFactor);

    // Compute the reuse for all the transfer_read operations, and form the
    // initial vector sizes.
    computeReuseInFunc(func, state);
    // We leverage the assumption that pointwise addition and multiplication
    // are commutative and associative to reassociate the operands of some
    // operators. This IR massaging makes it feasible to generate aie dialect
    // fma/msc intrinsics.
    reassociateOpsInFunc(func, state);
    // Rewrite vector dialect add and mul operation chains as vector dialect
    // fma operation if feasible.
    rewriteFMAOpsInFunc(func, state);
    // Coalesce vectors that only appear as LHS operands of mul/fma op if their
    // size is <= 256 bits.
    coalesceLHSOpVectorsInFunc(func, state);
    // Check for opportunities of fusing FMA ops to exploit the column topology
    // of the AIE vector intrinsic.
    fuseFMAOpsForColumnTopology(func, state);
    // For each vector dialect mul/fma op, compute the start and offset values
    // of its operands. Finally, generate AIE dialect mul/FMA ops.
    generateAIEMulOrFMAOpsInFunc(func, state);
    // Insert SRS ops to move data from accumulator to vector when the producer
    // is an AIE dialect op that writes to an accumulator, and the consumer
    // isn't an AIE dialect op.
    insertSRSOpsInFunc(func, state);
    // For each vector dialect add/sub op, compute the start and offset values
    // of its operands. Finally, generate AIE dialect add/sub ops. This should
    // be done after srs ops are generated, so that the input to the add op is
    // always vectors.
    generateAIEAddOrSubOpsInFunc(func, state);
    // Generate UPD ops that subsume all the transfer_read ops in affine
    // dialect. This happens after generating aie dialect add/sub ops because
    // those ops need to query transfer reads to know if their operand is
    // splat.
    insertUPDOpsInFunc(func, state);
  }

  // Canonicalize the IR of all the functions in the module by running a set of
  // cleanup passes.
  postCanonicalizeIR(module);
}

std::unique_ptr<Pass> xilinx::aievec::createAIEVectorizePass() {
  return std::make_unique<AIEVectorize>();
}
