add_mlir_library(MLIRPolyAIEPasses
  Preprocess.cpp
  SplitTopFunc.cpp
  CreateMemrefSubview.cpp
  HoistMemrefSubview.cpp
  MemrefArgToResult.cpp
  ExtractMemrefDependency.cpp
  BufferMemrefResult.cpp
  TensorizeMemref.cpp
  DetectLoopReduction.cpp
  ConvertToDataflow.cpp
  InsertGMIOAdapter.cpp
  Placement.cpp
  CreateInterface.cpp
  PrintDataflow.cpp

  
  DataflowToAIE.cpp
  LinkExternKernel.cpp
  MaterializeBroadcast.cpp
  FlowPacketToCircuit.cpp
  DoubleBuffer.cpp
  Postprocess.cpp
  CodeGenCleanup.cpp
  ReorderOperations.cpp
  Passes.cpp

  DEPENDS
  MLIRPolyAIEPassesIncGen
  )
