//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The PolyAIE Authors.
//
//===----------------------------------------------------------------------===//

#ifndef POLYAIE_DATAFLOW_DATAFLOW_H
#define POLYAIE_DATAFLOW_DATAFLOW_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/GraphTraits.h"

#include "polyaie/Dataflow/DataflowDialect.h.inc"
#include "polyaie/Dataflow/DataflowEnums.h.inc"

#define GET_OP_CLASSES
#include "polyaie/Dataflow/Dataflow.h.inc"

namespace llvm {
using namespace mlir;
using namespace polyaie;

// Specialize GraphTraits to treat FuncOp as a graph of ProcessOps as nodes and
// uses as edges.
// 将一个funcOp抽象成一个图，其中的node是ProcessOp，使用关系用边表示
// 这一部分是辅助print graph的
template <> struct GraphTraits<dataflow::FuncOp> {
  using GraphType = dataflow::FuncOp;
  using NodeRef = Operation *;

  // 判断是否是ProcessOp
  static bool isProcess(Operation *op) { 
    return isa<dataflow::ProcessOp>(op); 
  }

  // 用decltype获取函数指针类型，用来辅助过滤出process op
  // 针对一个operation的user进行过滤，只保留process op
  using ChildIteratorType =
      filter_iterator<Operation::user_iterator, decltype(&isProcess)>;
  static ChildIteratorType child_begin(NodeRef n) {
    return {n->user_begin(), n->user_end(), &isProcess};
  }
  static ChildIteratorType child_end(NodeRef n) {
    return {n->user_end(), n->user_end(), &isProcess};
  }

  // 用于获取funcOp的所有process op的迭代器
  // op_iterator是一个模板类，参数1定义了要迭代的op类型，参数2定义了迭代器的类型
  // 如下实例是在region 中迭代process op
  using nodes_iterator =
      mlir::detail::op_iterator<dataflow::ProcessOp, Region::OpIterator>;
  static nodes_iterator nodes_begin(dataflow::FuncOp f) {
    return f.body().getOps<dataflow::ProcessOp>().begin();
  }
  static nodes_iterator nodes_end(dataflow::FuncOp f) {
    return f.body().getOps<dataflow::ProcessOp>().end();
  }
};
} // namespace llvm

#endif // POLYAIE_DATAFLOW_DATAFLOW_H
