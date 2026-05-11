#pragma once

#include <cstdint>
#include <syclinterface/dpctl_sycl_types.h>

extern "C" DPCTLSyclKernelRef SYnergyTest_CreateBusyKernel(
    std::uintptr_t AdapterHandle
);