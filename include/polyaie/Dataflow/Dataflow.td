//===----------------------------------------------------------------------===//
//
// Copyright 2021-2022 The PolyAIE Authors.
//
//===----------------------------------------------------------------------===//

#ifndef POLYAIE_DATAFLOW_DATAFLOW_TD
#define POLYAIE_DATAFLOW_DATAFLOW_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Dataflow dialect and operations base
//===----------------------------------------------------------------------===//
def DataflowDialect : Dialect {
  let name = "dataflow";
  let summary = "A extension of the MLIR MemRef dialect";
  let description = [{
    This dialect contains operations and attributes that are useful in the
    PolyAIE compilation flow.
  }];
  let cppNamespace = "::mlir::polyaie::dataflow";
}

class DataflowOp<string mnemonic, list<Trait> traits = []> :
    Op<DataflowDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Dataflow Operations
//===----------------------------------------------------------------------===//

// 一个funcop表征一个函数，这个函数是一个handshaked函数，即每个value只有一个use
def FuncOp : DataflowOp<"func", [IsolatedFromAbove, FunctionOpInterface, Symbol,
    RegionKindInterface, OpAsmOpInterface]> {
  let summary = "Dataflow dialect function.";
  let description = [{
    The func operation represents a handshaked function. This is almost exactly
    like a standard FuncOp, except that it has some extra verification
    conditions. In particular, each Value must only have a single use.
  }];

  let regions = (region AnyRegion : $body);

  let skipDefaultBuilders = 1;

  let builders =
       [OpBuilder<(ins "StringRef":$name, "FunctionType":$type,
                      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>];

  let extraClassDeclaration = [{
    // Add an entry block to an empty function, and set up the block arguments
    // to match the signature of the function.
    Block *addEntryBlock();

    /// Returns the type of this function.
    FunctionType getFunctionType() {
      return getFunctionTypeAttr().getValue().cast<FunctionType>();
    }

    /// Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph; }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Returns the names of the arguments to this function.
    ArrayAttr getArgNames() {
      return (*this)->getAttrOfType<ArrayAttr>("argNames");
    }

    /// Returns the names of the results from this function.
    ArrayAttr getResNames() {
      return (*this)->getAttrOfType<ArrayAttr>("resNames");
    }

    /// Returns the argument name at the given index.
    StringAttr getArgName(unsigned idx) {
      return getArgNames()[idx].cast<StringAttr>();
    }

    /// Returns the result name at the given index.
    StringAttr getResName(unsigned idx) {
      return getResNames()[idx].cast<StringAttr>();
    }

    /// Resolve argument and result names. This can be used during building of
    /// a handshake.func operation to ensure that names provided by an incoming
    /// operation fulfills the name requirements of the handshake function.
    void resolveArgAndResNames();

    /// Hook for FunctionOpInterface, called after verifying that the 'type'
    /// attribute is present and checks if it holds a function type.  Ensures
    /// getType, getNumFuncArguments, and getNumFuncResults can be called
    /// safely.
    LogicalResult verifyType() {
      auto type = getFunctionTypeAttr().getValue();
      if (!type.isa<FunctionType>())
        return emitOpError(
            "requires '" + mlir::function_interface_impl::getTypeAttrName() +
            "' attribute of function type");
      return success();
    }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    static ::llvm::StringRef getDefaultDialect() {
      return "handshake";
    }
  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
  // let verifier = [{ return ::verify$cppClass(*this); }];
  // let printer = [{ return ::print$cppClass(p, *this); }];
  // let parser = [{ return ::parse$cppClass(parser, result); }];
}

// 为进程定义枚举类型
// 进程类型分为：AIE核上进程、全局内存IO进程、fpga IO进程
def AIE:  I32EnumAttrCase<"AIE", 0>;
def GMIO:  I32EnumAttrCase<"GMIO", 1>;
def PLIO:  I32EnumAttrCase<"PLIO", 2>;
def ProcessKind: I32EnumAttr<"ProcessKind", "Kind of Process",
    [AIE, GMIO, PLIO]> {
  let cppNamespace = "mlir::polyaie";
}

// 定义数据流抽象中的进程概念
// 一个process是isolated from above，即
// %result = arith.constant 10 : i32
// foo.region_op {
//  foo.yield %result : i32
// }
// 是不允许的
def ProcessOp : DataflowOp<"process", [IsolatedFromAbove]> {
  let summary = "process operation";
  let description = [{
    This operation represents a process in a dataflow graph.
  }];

  let arguments = (ins ProcessKind:$kind, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes, "ValueRange":$operands,
                   CArg<"ProcessKind", "ProcessKind::AIE">:$kind)>
  ];

  let hasVerifier = 1;
  // let verifier = "return ::verify(*this);";
  let extraClassDeclaration = [{
    /// Given an internal value, return the corresponding process operand. If
    /// the internal value is not a block argument, return nullptr.
    Value getOperandFromInternalVal(Value internalVal);

    /// Given an internal value, return the corresponding process result. If the
    /// internal value is not returned as a result, return nullptr.
    Value getResultFromInternalVal(Value internalVal);

    /// Given a process operand, return the corresponding block argument.
    Value getArgumentFromOperand(OpOperand &operand);

    /// Given a process result, return the corresponding returned value.
    Value getReturnValFromResult(OpResult result);
  }];
}

// 是一个region op的显示terminator op
def ReturnOp : DataflowOp<"return", [Terminator]> {
  let summary = "return operation of a process";
  let description = [{
    The operation terminates a process region and returns values generated by
    the process operation.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let hasVerifier = 1;
  // let verifier = "return ::verify(*this);";
}

// 定义一个tensor的获取操作，从内存中获取一个tensor
// 选用的trait指定了输入和输出的类型是相同的
def TensorLoadOp : DataflowOp<"tensor_load",
    [SameOperandsAndResultElementType]> {
  let summary = "tensor load operation";
  let description = [{
    This operation loads a tensor from a memory.
  }];

  let arguments = (ins I64ArrayAttr:$offsets, I64ArrayAttr:$sizes,
                       I64ArrayAttr:$strides, AnyMemRef:$memory);
  let results = (outs AnyTensor:$tensor);

  let builders = [ OpBuilder<(ins "Type":$type, "Value":$memory)> ];
  let hasVerifier = 1;
  // let verifier = "return ::verify(*this);";
}

def TensorStoreOp : DataflowOp<"tensor_store", [SameOperandsElementType]> {
  let summary = "tensor store operation";
  let description = [{
    This operation stores a tensor to a memory.
  }];

  let arguments = (ins I64ArrayAttr:$offsets, I64ArrayAttr:$sizes,
                       I64ArrayAttr:$strides, AnyMemRef:$memory,
                       AnyTensor:$tensor);

  let builders = [ OpBuilder<(ins "Value":$memory, "Value":$tensor)> ];
  let hasVerifier = 1;
  // let verifier = "return ::verify(*this);";
}

//===----------------------------------------------------------------------===//
// Runtime Operations - should be factored out to a new dialect
// 这一部分的操作主要是描述运行时的行为的，是dataflow dialect和aie dialect交集ops 
//===----------------------------------------------------------------------===//

// 定义host dma的类型
def HostToDDR:  I32EnumAttrCase<"HostToDDR", 0>;
def HostToAIE:  I32EnumAttrCase<"HostToAIE", 1>;
def DDRToHost:  I32EnumAttrCase<"DDRToHost", 2>;
def AIEToHost:  I32EnumAttrCase<"AIEToHost", 3>;
def HostDMAKind: I32EnumAttr<"HostDMAKind", "Kind of Host DMA",
    [HostToDDR, HostToAIE, DDRToHost, AIEToHost]> {
  let cppNamespace = "mlir::polyaie";
}

// 描述主机侧的dma传输，是和外部存储通过DDR做交互，或是和AIE核做交互。
// 相反方向也是同一描述方式，没有描述AIE to AIE的DMA传输
def HostDMAOp : DataflowOp<"runtime.host_dma", [SameOperandsElementType]> {
  let summary = "Host DMA operation";
  let description = [{
    This operation represents a host DMA to/from external memory, such as DDRs.
  }];

  let arguments = (ins I64ArrayAttr:$offsets, I64ArrayAttr:$sizes,
                       I64ArrayAttr:$strides, HostDMAKind:$kind,
                       AnyMemRef:$target, AnyMemRef:$source);

  let hasVerifier = 1;
  // let verifier = "return ::verify(*this);";
}

//===----------------------------------------------------------------------===//
// AIE Operations - should be factored out to AIE dialect
//===----------------------------------------------------------------------===//

// 这部分是方便Dataflow dialect接入AIE dialect
// 针对AIE核心与external memory交互，抽象接口，该接口可以是ShimNoc或ShimPL
def InterfaceOp : DataflowOp<"aie.interface", []> {
  let summary = "Represent a logical tile interface";
  let description = [{
    This operation indicates all "externalBuffers" are accessed through the
    logical interface of "tile". Note that the "tile" must be a shim tile,
    either a ShimNoc or a ShimPL.
  }];

  let arguments = (ins Index:$tile, Variadic<AnyMemRef>:$externalBuffers);

  // 先注释AIE部分，后续再添加
  // let hasVerifier = 1;
  // let verifier = "return ::verify(*this);";
}

// 定义AIE核心之间的broadcast操作
def BroadcastOp : DataflowOp<"aie.broadcast", [SameTypeOperands]> {
  let summary = "Memory broadcast operation";
  let description = [{
    Broadcast the source buffer to each target buffer.
  }];

  let arguments = (ins AnyMemRef:$source, Variadic<AnyMemRef>:$targets);

  // 先注释AIE部分，后续再添加
  // let hasVerifier = 1;
  // let verifier = "return ::verify(*this);";
}

#endif // POLYAIE_DATAFLOW_DATAFLOW_TD
