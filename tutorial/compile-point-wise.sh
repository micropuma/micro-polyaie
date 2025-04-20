#!/usr/bin/env bash
# 这是一个没有aie后端的test script，仅仅支持affine dialect转到dataflow dialect

DRY_RUN="false"
DEBUG_TILE="false"

RETURN_ALL_ARG="false"
TILE_SIZE="32"
VEC_SIZE="8"
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

TMP_CONV_DIR=${DIR}/tmp_pointwise
rm -rf ${TMP_CONV_DIR}
mkdir -p ${TMP_CONV_DIR}

# Run polyaie to generate the AIE IR of GEMM.
PIPELINE_OPTS="top-func-name=pointwise_mult "
PIPELINE_OPTS+="return-all-arg=${RETURN_ALL_ARG} "
PIPELINE_OPTS+="tile-size=${TILE_SIZE} "
PIPELINE_OPTS+="vec-size=${VEC_SIZE} "
PIPELINE_OPTS+="algorithm=${ALGORITHM} "
PIPELINE_OPTS+="enable-create-interface=${CREATE_INTERF} "
PIPELINE_OPTS+="enable-link-extern-kernel=${EXTERN_KERNEL} "
PIPELINE_OPTS+="object-file=${OBJECT_FILE} "
PIPELINE_OPTS+="gen-extern-kernel=${GEN_EXTERN_KERNEL}"

# ===========================================
${POLYAIE_OPT} ${DIR}/pointwise.mlir \
  -polyaie-pipeline="${PIPELINE_OPTS}" \
  --mlir-print-ir-after-all \
  -o ${TMP_CONV_DIR}/gemm.polyaie.mlir \
  2>&1 | tee ${TMP_CONV_DIR}/gemm.polyaie-debug.log

${POLYAIE_TRANSLATE} ${TMP_CONV_DIR}/gemm.polyaie.mlir \
  -export-host-kernel \
  -dry-run-host-kernel=${DRY_RUN} \
  -debug-tile=${DEBUG_TILE} \
  > ${TMP_CONV_DIR}/gemm.host.cpp

${POLYAIE_OPT} -polyaie-codegen-cleanup \
  ${TMP_CONV_DIR}/gemm.polyaie.mlir \
  > ${TMP_CONV_DIR}/gemm.polyaie.mliraie.mlir

${POLYAIE_OPT} -polyaie-reorder-operation \
  ${TMP_CONV_DIR}/gemm.polyaie.mliraie.mlir \
  > ${TMP_CONV_DIR}/gemm.polyaie.final.mlir