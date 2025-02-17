get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

# 设置可执行文件输出目录为 build/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(LLVM_LINK_COMPONENTS
  Support
  )

add_executable(polyaie-translate
  polyaie-translate.cpp
  )

llvm_update_compile_flags(polyaie-translate)

target_link_libraries(polyaie-translate
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  # AIE
  # MLIRAIEVec
  MLIRDataflow
  # MLIRTargetAIEVecCpp
  # 先注释AIE部分，后续再添加
  # MLIRPolyAIEExporters
  MLIRPolyAIESupport
  )
