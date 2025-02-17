#!/bin/bash

# 设置脚本在发生错误时退出
set -e

# 进入 LLVM 源码目录
cd llvm

# 如果 build 目录存在，删除它
if [ -d "build" ]; then
    echo "Removing existing build directory..."
    rm -rf build
fi

# 创建新的 build 目录
mkdir build && cd build

# 运行 CMake 配置
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DLLVM_USE_LINKER=lld \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++

# 运行 Ninja 进行编译
ninja -j $(nproc)

# 运行 MLIR 测试
ninja check-mlir

echo "MLIR build and tests completed successfully!"
