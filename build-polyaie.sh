#!/bin/bash

if [ -d "build" ]; then
	echo "Removing existing build directory..."
	rm -rf build
fi

mkdir build && cd build

cmake -G Ninja .. \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DLLVM_USE_LINKER=lld \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++

ninja -j $(nproc) 2>&1 | tee build.log

