#pragma once

#include <cstdint>
#include <syclinterface/dpctl_sycl_types.h>

extern "C" DPCTLSyclKernelRef SYnergyTest_CreateVecAddKernel(
    std::uintptr_t AdapterHandle
);

extern "C" DPCTLSyclKernelRef SYnergyTest_CreateVecprodKernel(
    std::uintptr_t AdapterHandle
);