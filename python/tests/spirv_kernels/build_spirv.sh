#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CL_FILE="${SCRIPT_DIR}/vector_add.cl"
BC_FILE="${SCRIPT_DIR}/vector_add.bc"
SPV_FILE="${SCRIPT_DIR}/vector_add.spv"

echo "[INFO] Compiling OpenCL C to LLVM bitcode..."

clang \
  -x cl \
  -target spir64-unknown-unknown \
  -cl-std=CL2.0 \
  -Xclang -finclude-default-header \
  -emit-llvm \
  -c "${CL_FILE}" \
  -o "${BC_FILE}"

echo "[INFO] Translating LLVM bitcode to SPIR-V..."

LLVM_SPIRV_BIN=""

for candidate in llvm-spirv llvm-spirv-20 llvm-spirv-18 llvm-spirv-17 llvm-spirv-16 llvm-spirv-15; do
    if command -v "$candidate" >/dev/null 2>&1; then
        LLVM_SPIRV_BIN="$candidate"
        break
    fi
done

if [ -z "$LLVM_SPIRV_BIN" ]; then
    echo "[ERROR] llvm-spirv not found."
    echo "Try:"
    echo "  apt-cache search llvm-spirv"
    echo "  sudo apt install llvm-spirv-18"
    exit 1
fi

"${LLVM_SPIRV_BIN}" "${BC_FILE}" -o "${SPV_FILE}"
echo "[OK] Generated: ${SPV_FILE}"