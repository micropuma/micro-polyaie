//===- AIEInterfaces.td ------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2020 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def FlowEndPoint : OpInterface<"FlowEndPoint"> {
  let description = [{
    Interface for operations that have interconnect-like properties,
    enabling them to host flows for routing.
  }];
  let cppNamespace = "::xilinx::AIE";
  let methods = [
    InterfaceMethod<[{
      }],
      "int", "colIndex", (ins )
    >,
    InterfaceMethod<[{
      }],
      "int", "rowIndex", (ins )
    >
  ];
}

def Interconnect : OpInterface<"Interconnect"> {
  let description = [{
    Interface for operations that have interconnect-like properties,
    enabling them to host flows for routing.
  }];
  let cppNamespace = "::xilinx::AIE";

  let methods = [
    InterfaceMethod<[{
      }],
      "::mlir::Region &", "connections", (ins )
    >,
    InterfaceMethod<[{
      }],
      "int", "colIndex", (ins )
    >,
    InterfaceMethod<[{
      }],
      "int", "rowIndex", (ins )
    >,
    InterfaceMethod<[{
        int FooOp::getNumSourceConnections(WireBundle bundle) {
      }],
      "int", "getNumSourceConnections", (ins "WireBundle":$bundle)
    >,
    InterfaceMethod<[{
        int FooOp::getNumDestConnections(WireBundle bundle) {
      }],
      "int", "getNumDestConnections", (ins "WireBundle":$bundle)
    >
  ];
}
