# PolyAIE Project

15958114660

## gemm Example
```sh
$ python scripts/pb-flow.py --polymer --loop-transforms --dataset SMALL --skip-vitis --skip-csim example/polybench

$ sed -E 's/arith.//g; s/f64/f32/g; s/andi/and/g; s/alloca/alloc/g; s/llvm.linkage[[:space:]]=[[:space:]]#llvm.linkage<external>,[[:space:]]//g' gemm.pre.kern.plmr.ca.lt.mlir > gemm.phism.mlir
$ polyaie-opt -polyaie-pipeline="top-func-name=kernel_gemm algorithm=simulated-annealing vec-size=1" gemm.phism.mlir 1> gemm.phism.polyaie.mlir 2> gemm.phism.polyaie.dot
$ polyaie-translate -export-host-kernel -debug-host-kernel=false -dry-run-host-kernel=true gemm.phism.polyaie.mlir > gemm.host.cpp

$ dot -Tpng gemm.phism.polyaie.dot > gemm.phism.polyaie.df.png && dot -Tpng -Kfdp gemm.phism.polyaie.dot > gemm.phism.polyaie.layout.png
$ sed -E '/memcpy/d; /alloc/d' gemm.phism.polyaie.mlir > gemm.phism.polyaie.mliraie.mlir

$ source /tools/Xilinx/Vitis/2020.1/settings64.sh
$ /home/hanchenye/workspace/polyaie/mlir-aie/build/bin/aiecc.py -j10 \
    --sysroot=/home/hanchenye/workspace/polyaie/mlir-aie/platforms/vck190_bare/petalinux/sysroot/sysroots/aarch64-xilinx-linux \
    gemm.phism.polyaie.mliraie.mlir \
    -I/home/hanchenye/workspace/polyaie/mlir-aie/build/runtime_lib/ \
    -I/home/hanchenye/workspace/polyaie/tmp/polybench-small/utilities/ -I${PWD} \
    /home/hanchenye/workspace/polyaie/tmp/polybench-small/utilities/polybench.cpp \
    /home/hanchenye/workspace/polyaie/mlir-aie/build/runtime_lib/test_library.cpp \
    gemm.cpp -o gemm.elf

$ /usr/bin/clang --target=aarch64-linux-gnu -std=c++11 \
    --sysroot=/home/hanchenye/workspace/polyaie/mlir-aie/platforms/vck190_bare/petalinux/sysroot/sysroots/aarch64-xilinx-linux \
    -I/home/hanchenye/workspace/polyaie/mlir-aie/platforms/vck190_bare/petalinux/sysroot/sysroots/aarch64-xilinx-linux/usr/include/c++/9.2.0 \
    -I/home/hanchenye/workspace/polyaie/mlir-aie/platforms/vck190_bare/petalinux/sysroot/sysroots/aarch64-xilinx-linux/usr/include/c++/9.2.0/aarch64-xilinx-linux \
    -I/home/hanchenye/workspace/polyaie/mlir-aie/platforms/vck190_bare/petalinux/sysroot/sysroots/aarch64-xilinx-linux/usr/include/c++/9.2.0/backward \
    -L/home/hanchenye/workspace/polyaie/mlir-aie/platforms/vck190_bare/petalinux/sysroot/sysroots/aarch64-xilinx-linux/usr/lib/aarch64-xilinx-linux/9.2.0 \
    -B/home/hanchenye/workspace/polyaie/mlir-aie/platforms/vck190_bare/petalinux/sysroot/sysroots/aarch64-xilinx-linux/usr/lib/aarch64-xilinx-linux/9.2.0 \
    -Iacdc_project -fuse-ld=lld -rdynamic -lxaiengine -lmetal -lopen_amp -ldl \
    -I/home/hanchenye/workspace/polyaie/mlir-aie/build/runtime_lib/ \
    -I/home/hanchenye/workspace/polyaie/tmp/polybench-small/utilities/ -I${PWD} \
    /home/hanchenye/workspace/polyaie/tmp/polybench-small/utilities/polybench.cpp \
    /home/hanchenye/workspace/polyaie/mlir-aie/build/runtime_lib/test_library.cpp \
    gemm.cpp -o gemm.elf
```

## 2mm Example
```sh
$ sed -E 's/arith.//g; s/f64/f32/g; s/andi/and/g; s/alloca/alloc/g; s/llvm.linkage[[:space:]]=[[:space:]]#llvm.linkage<external>,[[:space:]]//g' 2mm.pre.kern.plmr.ca.lt.mlir > 2mm.phism.mlir
$ polyaie-opt -polyaie-pipeline="top-func-name=kernel_2mm algorithm=simulated-annealing vec-size=1" 2mm.phism.mlir 1> 2mm.phism.polyaie.mlir 2> 2mm.phism.polyaie.dot
$ polyaie-translate -export-host-kernel -debug-host-kernel=false -dry-run-host-kernel=true 2mm.phism.polyaie.mlir > 2mm.host.cpp

$ dot -Tpng 2mm.phism.polyaie.dot > 2mm.phism.polyaie.df.png && dot -Tpng -Kfdp 2mm.phism.polyaie.dot > 2mm.phism.polyaie.layout.png
$ sed -E '/memcpy/d; /alloc/d' 2mm.phism.polyaie.mlir > 2mm.phism.polyaie.mliraie.mlir

$ source /tools/Xilinx/Vitis/2020.1/settings64.sh
$ /home/hanchenye/workspace/polyaie/mlir-aie/build/bin/aiecc.py -j10 \
    --sysroot=/home/hanchenye/workspace/polyaie/mlir-aie/platforms/vck190_bare/petalinux/sysroot/sysroots/aarch64-xilinx-linux \
    2mm.phism.polyaie.mliraie.mlir \
    -I/home/hanchenye/workspace/polyaie/mlir-aie/build/runtime_lib/ \
    -I/home/hanchenye/workspace/polyaie/tmp/polybench-small/utilities/ -I${PWD} \
    /home/hanchenye/workspace/polyaie/tmp/polybench-small/utilities/polybench.cpp \
    /home/hanchenye/workspace/polyaie/mlir-aie/build/runtime_lib/test_library.cpp \
    2mm.cpp -o 2mm.elf

$ /usr/bin/clang --target=aarch64-linux-gnu -std=c++11 \
    --sysroot=/home/hanchenye/workspace/polyaie/mlir-aie/platforms/vck190_bare/petalinux/sysroot/sysroots/aarch64-xilinx-linux \
    -I/home/hanchenye/workspace/polyaie/mlir-aie/platforms/vck190_bare/petalinux/sysroot/sysroots/aarch64-xilinx-linux/usr/include/c++/9.2.0 \
    -I/home/hanchenye/workspace/polyaie/mlir-aie/platforms/vck190_bare/petalinux/sysroot/sysroots/aarch64-xilinx-linux/usr/include/c++/9.2.0/aarch64-xilinx-linux \
    -I/home/hanchenye/workspace/polyaie/mlir-aie/platforms/vck190_bare/petalinux/sysroot/sysroots/aarch64-xilinx-linux/usr/include/c++/9.2.0/backward \
    -L/home/hanchenye/workspace/polyaie/mlir-aie/platforms/vck190_bare/petalinux/sysroot/sysroots/aarch64-xilinx-linux/usr/lib/aarch64-xilinx-linux/9.2.0 \
    -B/home/hanchenye/workspace/polyaie/mlir-aie/platforms/vck190_bare/petalinux/sysroot/sysroots/aarch64-xilinx-linux/usr/lib/aarch64-xilinx-linux/9.2.0 \
    -Iacdc_project -fuse-ld=lld -rdynamic -lxaiengine -lmetal -lopen_amp -ldl \
    -I/home/hanchenye/workspace/polyaie/mlir-aie/build/runtime_lib/ \
    -I/home/hanchenye/workspace/polyaie/tmp/polybench-small/utilities/ -I${PWD} \
    /home/hanchenye/workspace/polyaie/tmp/polybench-small/utilities/polybench.cpp \
    /home/hanchenye/workspace/polyaie/mlir-aie/build/runtime_lib/test_library.cpp \
    2mm.cpp -o 2mm.elf
```

## Quick Start
### 0. Clone PolyAIE
```sh
$ git clone --recursive git@github.com:hanchenye/polyaie.git
$ cd polyaie
```

### 1. Install MLIR
```sh
$ cd llvm
$ mkdir build && cd build
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_LINK_LLVM_DYLIB=ON \
    -DLLVM_BUILD_UTILS=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_USE_LINKER=lld \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++
$ ninja && ninja check-mlir
```

### 2. Install MLIR-AIE
Follow the instructions from https://xilinx.github.io/mlir-aie/.
```sh
$ cd mlir-aie

$ # (Optional) Build VCK190 platform for generating executables.
$ source /tools/Xilinx/Vitis/2020.1/settings64.sh # Your Vitis settings script.
$ source ~/tools/Xilinx/PetaLinux/settings.sh # Your PetaLinux settings script.
$ cd platforms/vck190_bare && make all && cd ../..

$ # Build MLIR-AIE compilation flow.
$ mkdir build && cd build
$ /usr/bin/cmake -GNinja .. \
    -DLLVM_DIR=${PWD}/../../llvm/build/lib/cmake/llvm \
    -DMLIR_DIR=${PWD}/../../llvm/build/lib/cmake/mlir \
    -DCMAKE_MODULE_PATH=${PWD}/../../cmakeModules \
    -DVitisSysroot=${PWD}/../platforms/vck190_bare/petalinux/sysroot/sysroots/aarch64-xilinx-linux \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DLLVM_USE_LINKER=lld \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++
$ ninja && ninja check-aie
```

### 3. Install PolyAIE
```sh
$ mkdir build && cd build
$ /usr/bin/cmake -G Ninja .. \
    -DMLIR_DIR=${PWD}/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=${PWD}/../llvm/build/lib/cmake/llvm \
    -DAIE_DIR=${PWD}/../mlir-aie/build/lib/cmake/aie \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DLLVM_USE_LINKER=lld \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++
$ ninja check-polyaie
$ export PATH=$PATH:${PWD}/bin
```

### 4. Clone and install Phism
Follow the instructions from https://github.com/kumasento/phism.
```sh
$ git clone --recursive git@github.com:kumasento/phism.git
$ cd phism
$ ./scripts/build-llvm.sh
$ ./scripts/build-polygeist.sh
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PWD}/polymer/build/pluto/lib
$ ./scripts/build-polymer.sh
$ ./scripts/build-phism.sh
$ export PYTHONPATH=$PYTHONPATH:/home/hanchenye/workspace/polyaie/phism
```
