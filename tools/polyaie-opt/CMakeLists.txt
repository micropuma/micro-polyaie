get_property(all_libs GLOBAL PROPERTY MLIR_ALL_LIBS)

# 设置可执行文件输出目录为 build/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(polyaie-opt
  polyaie-opt.cpp
  )

llvm_update_compile_flags(polyaie-opt)

target_link_libraries(polyaie-opt
  PRIVATE
  ${all_libs}

  # # 先注释AIE部分，后续再添加
  AIE
  MLIRAIEVec
  MLIRAIEVecTransforms
  MLIRPolyAIESupport
  MLIRDataflow
  MLIRPolyAIEPasses
  )
