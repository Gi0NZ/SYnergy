#pragma once

#include <cstddef>
#include <cstdint>

#include <syclinterface/dpctl_sycl_queue_interface.h>
#include <syclinterface/dpctl_sycl_types.h>

/**
 * Submit SYnergy su range semplice.
 *
 * Equivalente concettuale di DPCTLQueue_SubmitRange,
 * ma usa synergy::queue invece di sycl::queue.
 */
extern "C" DPCTLSyclEventRef SYnergyQueue_SubmitRange(
    std::uintptr_t AdapterHandle,
    const DPCTLSyclKernelRef KRef,
    void** Args,
    const DPCTLKernelArgType* ArgTypes,
    std::size_t NArgs,
    const std::size_t Range[3],
    std::size_t NDims,
    const DPCTLSyclEventRef* DepEvents,
    std::size_t NDepEvents,
    unsigned int UncoreFrequency,
    unsigned int CoreFrequency,
    int UseFrequencyScaling
);

/**
 * Submit SYnergy su nd_range.
 *
 * Equivalente concettuale di DPCTLQueue_SubmitNDRange,
 * ma usa synergy::queue invece di sycl::queue.
 */
extern "C" DPCTLSyclEventRef SYnergyQueue_SubmitNDRange(
    std::uintptr_t AdapterHandle,
    const DPCTLSyclKernelRef KRef,
    void** Args,
    const DPCTLKernelArgType* ArgTypes,
    std::size_t NArgs,
    const std::size_t gRange[3],
    const std::size_t lRange[3],
    std::size_t NDims,
    const DPCTLSyclEventRef* DepEvents,
    std::size_t NDepEvents,
    unsigned int UncoreFrequency,
    unsigned int CoreFrequency,
    int UseFrequencyScaling
);