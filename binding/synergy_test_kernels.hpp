#pragma once

#include <cstdint>
#include <syclinterface/dpctl_sycl_types.h>

/**
 * @file synergy_test_kernels.hpp
 * @brief Native SYCL test-kernel factories used by the Python bindings.
 *
 * This header declares small native kernel factory functions used by the
 * Cython bridge to create precompiled SYCL kernels and expose them to Python
 * as ``dpctl.program.SyclKernel`` objects.
 *
 * These kernels are mainly intended for testing and examples. They make it
 * possible to validate the native SYnergy submission path without relying on
 * runtime kernel compilation from OpenCL source or SPIR-V.
 */

/**
 * @brief Create a native vector-add test kernel.
 *
 * The returned kernel is wrapped as a DPCTL kernel reference so that it can be
 * exposed to Python as a ``dpctl.program.SyclKernel``. This kernel is used to
 * test native SYnergy submission, USM argument passing and optional energy
 * profiling.
 *
 * @param AdapterHandle Integer handle to a ``SYnergy_Queue_Adapter`` instance.
 * @return DPCTLSyclKernelRef Wrapped vector-add kernel reference, or ``nullptr``
 * if the kernel cannot be created.
 */
extern "C" DPCTLSyclKernelRef SYnergyTest_CreateVecAddKernel(
    std::uintptr_t AdapterHandle
);

/**
 * @brief Create a native vector-product test kernel.
 *
 * The returned kernel is wrapped as a DPCTL kernel reference and exposed to
 * Python as a ``dpctl.program.SyclKernel``. This kernel is intended for simple
 * native workload tests and benchmarking experiments.
 *
 * @param AdapterHandle Integer handle to a ``SYnergy_Queue_Adapter`` instance.
 * @return DPCTLSyclKernelRef Wrapped vector-product kernel reference, or
 * ``nullptr`` if the kernel cannot be created.
 */
extern "C" DPCTLSyclKernelRef SYnergyTest_CreateVecprodKernel(
    std::uintptr_t AdapterHandle
);