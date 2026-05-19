/*#include "synergy_submit_backend.hpp"

#include "synergy_queue_adapter.hpp"

#include "syclinterface/dpctl_sycl_type_casters.hpp"

#include <sycl/sycl.hpp>

#include <cstdlib>

#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <utility>

using namespace sycl;
using namespace dpctl::syclinterface;

namespace {

SYnergy_Queue_Adapter* adapter_from_handle(std::uintptr_t handle) {
    if (handle == 0) {
        throw std::invalid_argument("SYnergy adapter handle is null.");
    }

    return reinterpret_cast<SYnergy_Queue_Adapter*>(handle);
}

void debug_submit_message(
    const char* name,
    std::size_t n_dims,
    int use_frequency_scaling,
    unsigned int uncore_frequency,
    unsigned int core_frequency
) {
    if (std::getenv("SYNERGY_SUBMIT_DEBUG") != nullptr) {
        std::cerr
            << "[SYnergy submit backend] "
            << name
            << " called | NDims=" << n_dims
            << " | frequency_scaling=" << use_frequency_scaling
            << " | uncore_frequency=" << uncore_frequency
            << " | core_frequency=" << core_frequency
            << std::endl;
    }
}

void set_synergy_dependent_events(
    handler& cgh,
    const DPCTLSyclEventRef* dep_events,
    std::size_t n_dep_events
) {
    for (std::size_t i = 0; i < n_dep_events; ++i) {
        auto event_ptr = dpctl::syclinterface::unwrap<sycl::event>(dep_events[i]);

        if (event_ptr) {
            cgh.depends_on(*event_ptr);
        }
    }
}

template <typename T, int NDims>
bool set_local_accessor_for_type(
    handler& cgh,
    std::size_t idx,
    const range<NDims>& r
) {
    local_accessor<T, NDims> la(r, cgh);
    cgh.set_arg(idx, la);
    return true;
}

template <int NDims>
bool set_local_accessor_for_dim(
    handler& cgh,
    std::size_t idx,
    DPCTLKernelArgType type_id,
    const range<NDims>& r
) {
    switch (type_id) {
        case DPCTL_INT8_T:
            return set_local_accessor_for_type<std::int8_t, NDims>(cgh, idx, r);

        case DPCTL_UINT8_T:
            return set_local_accessor_for_type<std::uint8_t, NDims>(cgh, idx, r);

        case DPCTL_INT16_T:
            return set_local_accessor_for_type<std::int16_t, NDims>(cgh, idx, r);

        case DPCTL_UINT16_T:
            return set_local_accessor_for_type<std::uint16_t, NDims>(cgh, idx, r);

        case DPCTL_INT32_T:
            return set_local_accessor_for_type<std::int32_t, NDims>(cgh, idx, r);

        case DPCTL_UINT32_T:
            return set_local_accessor_for_type<std::uint32_t, NDims>(cgh, idx, r);

        case DPCTL_INT64_T:
            return set_local_accessor_for_type<std::int64_t, NDims>(cgh, idx, r);

        case DPCTL_UINT64_T:
            return set_local_accessor_for_type<std::uint64_t, NDims>(cgh, idx, r);

        case DPCTL_FLOAT32_T:
            return set_local_accessor_for_type<float, NDims>(cgh, idx, r);

        case DPCTL_FLOAT64_T:
            return set_local_accessor_for_type<double, NDims>(cgh, idx, r);

        default:
            return false;
    }
}

bool set_synergy_local_accessor_arg(
    handler& cgh,
    std::size_t idx,
    const MDLocalAccessor* md
) {
    if (md == nullptr) {
        return false;
    }

    switch (md->ndim) {
        case 1:
            return set_local_accessor_for_dim<1>(
                cgh,
                idx,
                md->dpctl_type_id,
                range<1>{md->dim0}
            );

        case 2:
            return set_local_accessor_for_dim<2>(
                cgh,
                idx,
                md->dpctl_type_id,
                range<2>{md->dim0, md->dim1}
            );

        case 3:
            return set_local_accessor_for_dim<3>(
                cgh,
                idx,
                md->dpctl_type_id,
                range<3>{md->dim0, md->dim1, md->dim2}
            );

        default:
            return false;
    }
}

bool set_synergy_kernel_arg(
    handler& cgh,
    std::size_t idx,
    void* arg,
    DPCTLKernelArgType arg_type
) {
    switch (arg_type) {
        case DPCTL_INT8_T:
            cgh.set_arg(idx, *reinterpret_cast<std::int8_t*>(arg));
            return true;

        case DPCTL_UINT8_T:
            cgh.set_arg(idx, *reinterpret_cast<std::uint8_t*>(arg));
            return true;

        case DPCTL_INT16_T:
            cgh.set_arg(idx, *reinterpret_cast<std::int16_t*>(arg));
            return true;

        case DPCTL_UINT16_T:
            cgh.set_arg(idx, *reinterpret_cast<std::uint16_t*>(arg));
            return true;

        case DPCTL_INT32_T:
            cgh.set_arg(idx, *reinterpret_cast<std::int32_t*>(arg));
            return true;

        case DPCTL_UINT32_T:
            cgh.set_arg(idx, *reinterpret_cast<std::uint32_t*>(arg));
            return true;

        case DPCTL_INT64_T:
            cgh.set_arg(idx, *reinterpret_cast<std::int64_t*>(arg));
            return true;

        case DPCTL_UINT64_T:
            cgh.set_arg(idx, *reinterpret_cast<std::uint64_t*>(arg));
            return true;

        case DPCTL_FLOAT32_T:
            cgh.set_arg(idx, *reinterpret_cast<float*>(arg));
            return true;

        case DPCTL_FLOAT64_T:
            cgh.set_arg(idx, *reinterpret_cast<double*>(arg));
            return true;

        case DPCTL_VOID_PTR:
            cgh.set_arg(idx, arg);
            return true;

        case DPCTL_LOCAL_ACCESSOR:
            return set_synergy_local_accessor_arg(
                cgh,
                idx,
                reinterpret_cast<MDLocalAccessor*>(arg)
            );

        default:
            return false;
    }
}

void set_synergy_kernel_args(
    handler& cgh,
    void** args,
    const DPCTLKernelArgType* arg_types,
    std::size_t n_args
) {
    for (std::size_t i = 0; i < n_args; ++i) {
        const bool ok = set_synergy_kernel_arg(
            cgh,
            i,
            args[i],
            arg_types[i]
        );

        if (!ok) {
            throw std::invalid_argument(
                "Unsupported kernel argument type in SYnergy submit backend."
            );
        }
    }
}

template <typename CommandGroup>
event submit_with_synergy_policy(
    synergy::queue& q,
    CommandGroup&& command_group,
    unsigned int uncore_frequency,
    unsigned int core_frequency,
    int use_frequency_scaling
) {
    if (use_frequency_scaling) {
        return q.submit(
            uncore_frequency,
            core_frequency,
            std::forward<CommandGroup>(command_group)
        );
    }

    return q.submit(std::forward<CommandGroup>(command_group));
}

DPCTLSyclEventRef wrap_event(sycl::event&& e) {
    return ::dpctl::syclinterface::wrap<sycl::event>(
        new sycl::event(std::move(e))
    );
}
} // namespace

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
) {
    try {
        
        //sezione di testing, mi assicuro che venga chiamato il backend synergy da python
        debug_submit_message(
            "SYnergyQueue_SubmitRange",
            NDims,
            UseFrequencyScaling,
            UncoreFrequency,
            CoreFrequency
        );

        auto* adapter = adapter_from_handle(AdapterHandle);
        auto* kernel = ::dpctl::syclinterface::unwrap<sycl::kernel>(KRef);

        if (kernel == nullptr) {
            throw std::invalid_argument("Kernel reference is null.");
        }

        auto& q = adapter->native_queue();

        switch (NDims) {
            case 1: {
                event e = submit_with_synergy_policy(
                    q,
                    [&](handler& cgh) {
                        set_synergy_dependent_events(
                            cgh,
                            DepEvents,
                            NDepEvents
                        );
                        set_synergy_kernel_args(cgh, Args, ArgTypes, NArgs);
                        cgh.parallel_for(range<1>{Range[0]}, *kernel);
                    },
                    UncoreFrequency,
                    CoreFrequency,
                    UseFrequencyScaling
                );

                return wrap_event(std::move(e));
            }

            case 2: {
                event e = submit_with_synergy_policy(
                    q,
                    [&](handler& cgh) {
                        set_synergy_dependent_events(
                            cgh,
                            DepEvents,
                            NDepEvents
                        );
                        set_synergy_kernel_args(cgh, Args, ArgTypes, NArgs);
                        cgh.parallel_for(
                            range<2>{Range[0], Range[1]},
                            *kernel
                        );
                    },
                    UncoreFrequency,
                    CoreFrequency,
                    UseFrequencyScaling
                );

                return wrap_event(std::move(e));
            }

            case 3: {
                event e = submit_with_synergy_policy(
                    q,
                    [&](handler& cgh) {
                        set_synergy_dependent_events(
                            cgh,
                            DepEvents,
                            NDepEvents
                        );
                        set_synergy_kernel_args(cgh, Args, ArgTypes, NArgs);
                        cgh.parallel_for(
                            range<3>{Range[0], Range[1], Range[2]},
                            *kernel
                        );
                    },
                    UncoreFrequency,
                    CoreFrequency,
                    UseFrequencyScaling
                );

                return wrap_event(std::move(e));
            }

            default:
                throw std::invalid_argument(
                    "Range must have 1, 2 or 3 dimensions."
                );
        }
    } catch (const std::exception& e) {
        std::cerr << "SYnergyQueue_SubmitRange failed: "
                  << e.what()
                  << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "SYnergyQueue_SubmitRange failed: unknown exception"
                  << std::endl;
        return nullptr;
    }
}

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
) {
    try {
        auto* adapter = adapter_from_handle(AdapterHandle);
        auto* kernel = ::dpctl::syclinterface::unwrap<sycl::kernel>(KRef);

        if (kernel == nullptr) {
            throw std::invalid_argument("Kernel reference is null.");
        }

        auto& q = adapter->native_queue();

        switch (NDims) {
            case 1: {
                event e = submit_with_synergy_policy(
                    q,
                    [&](handler& cgh) {
                        set_synergy_dependent_events(
                            cgh,
                            DepEvents,
                            NDepEvents
                        );
                        set_synergy_kernel_args(cgh, Args, ArgTypes, NArgs);
                        cgh.parallel_for(
                            nd_range<1>{{gRange[0]}, {lRange[0]}},
                            *kernel
                        );
                    },
                    UncoreFrequency,
                    CoreFrequency,
                    UseFrequencyScaling
                );

                return wrap_event(std::move(e));
            }

            case 2: {
                event e = submit_with_synergy_policy(
                    q,
                    [&](handler& cgh) {
                        set_synergy_dependent_events(
                            cgh,
                            DepEvents,
                            NDepEvents
                        );
                        set_synergy_kernel_args(cgh, Args, ArgTypes, NArgs);
                        cgh.parallel_for(
                            nd_range<2>{
                                {gRange[0], gRange[1]},
                                {lRange[0], lRange[1]}
                            },
                            *kernel
                        );
                    },
                    UncoreFrequency,
                    CoreFrequency,
                    UseFrequencyScaling
                );

                return wrap_event(std::move(e));
            }

            case 3: {
                event e = submit_with_synergy_policy(
                    q,
                    [&](handler& cgh) {
                        set_synergy_dependent_events(
                            cgh,
                            DepEvents,
                            NDepEvents
                        );
                        set_synergy_kernel_args(cgh, Args, ArgTypes, NArgs);
                        cgh.parallel_for(
                            nd_range<3>{
                                {gRange[0], gRange[1], gRange[2]},
                                {lRange[0], lRange[1], lRange[2]}
                            },
                            *kernel
                        );
                    },
                    UncoreFrequency,
                    CoreFrequency,
                    UseFrequencyScaling
                );

                return wrap_event(std::move(e));
            }

            default:
                throw std::invalid_argument(
                    "NDRange must have 1, 2 or 3 dimensions."
                );
        }
    } catch (const std::exception& e) {
        std::cerr << "SYnergyQueue_SubmitNDRange failed: "
                  << e.what()
                  << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "SYnergyQueue_SubmitNDRange failed: unknown exception"
                  << std::endl;
        return nullptr;
    }


    
}
    */