#!/usr/bin/env bash
# 这是一个没有aie后端的test script，仅仅支持affine dialect转到dataflow dialect

DRY_RUN="false"
DEBUG_TILE="false"

RETURN_ALL_ARG="false"
TILE_SIZE="32"
VEC_SIZE="1"
ALGORITHM="simulated-annealing"
CREATE_INTERF="false"

EXTERN_KERNEL="false"
OBJECT_FILE="kernel.o"
GEN_EXTERN_KERNEL="false"
VITIS_DIR=/tools/Xilinx/Vitis/2020.1

POLYAIE_OPT=$PWD/../build/bin/polyaie-opt
POLYAIE_TRANSLATE=$PWD/../build/bin/polyaie-translate

# Get the absolute path of the current directory.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
TMP_DEBUG_DIR=${DIR}/tmp-debug
rm -rf ${TMP_DEBUG_DIR}
mkdir -p ${TMP_DEBUG_DIR}

TMP_DIR=${DIR}/tmp
rm -rf ${TMP_DIR}
mkdir -p ${TMP_DIR}

# Run polyaie to generate the AIE IR of GEMM.
PIPELINE_OPTS="top-func-name=gemm "
PIPELINE_OPTS+="return-all-arg=${RETURN_ALL_ARG} "
PIPELINE_OPTS+="tile-size=${TILE_SIZE} "
PIPELINE_OPTS+="vec-size=${VEC_SIZE} "
PIPELINE_OPTS+="algorithm=${ALGORITHM} "
PIPELINE_OPTS+="enable-create-interface=${CREATE_INTERF} "
PIPELINE_OPTS+="enable-link-extern-kernel=${EXTERN_KERNEL} "
PIPELINE_OPTS+="object-file=${OBJECT_FILE} "
PIPELINE_OPTS+="gen-extern-kernel=${GEN_EXTERN_KERNEL}"



${POLYAIE_OPT} ${DIR}/gemm.mlir \
  -polyaie-pipeline="${PIPELINE_OPTS}" \
  --mlir-print-ir-after-all \
  -debug-only=dialect-conversion \
  -o ${TMP_DIR}/gemm.polyaie.mlir \
  2>&1 | tee ${TMP_DIR}/gemm.polyaie-debug.log

${POLYAIE_TRANSLATE} ${TMP_DIR}/gemm.polyaie.mlir \
  -export-host-kernel \
  -dry-run-host-kernel=${DRY_RUN} \
  -debug-tile=${DEBUG_TILE} \
  > ${TMP_DIR}/gemm.host.cpp

${POLYAIE_OPT} -polyaie-codegen-cleanup \
  ${TMP_DIR}/gemm.polyaie.mlir \
  > ${TMP_DIR}/gemm.polyaie.mliraie.mlir

${POLYAIE_OPT} ${DIR}/gemm-simple.mlir \
  -polyaie-pipeline="${PIPELINE_OPTS}" \
  --mlir-print-ir-after-all \
  -debug-only=dialect-conversion \
  -o ${TMP_DEBUG_DIR}/gemm.polyaie.mlir \
  2>&1 | tee ${TMP_DEBUG_DIR}/gemm.polyaie-debug.log

${POLYAIE_TRANSLATE} ${TMP_DEBUG_DIR}/gemm.polyaie.mlir \
  -export-host-kernel \
  -dry-run-host-kernel=${DRY_RUN} \
  -debug-tile=${DEBUG_TILE} \
  > ${TMP_DEBUG_DIR}/gemm.host.cpp

${POLYAIE_OPT} -polyaie-codegen-cleanup \
  ${TMP_DEBUG_DIR}/gemm.polyaie.mlir \
  > ${TMP_DEBUG_DIR}/gemm.polyaie.mliraie.mlir
