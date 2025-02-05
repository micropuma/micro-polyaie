cd build/

/usr/bin/cmake -G Ninja .. \
    -DLLVM_DIR=/home/douliyang/large/mlir-workspace/mlir-aie/llvm/build/lib/cmake/llvm \
    -DMLIR_DIR=/home/douliyang/large/mlir-workspace/mlir-aie/llvm/build/lib/cmake/mlir \
    -DAIE_DIR=/home/douliyang/large/mlir-workspace/mlir-aie/build/lib/cmake/aie \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DLLVM_USE_LINKER=lld \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++
