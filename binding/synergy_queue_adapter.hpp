#pragma once

#include <sycl/sycl.hpp>
#include <synergy.hpp>
#include <string>

/**
 * @brief Adapter C++ per esporre synergy::queue a Python tramite pybind11.
 *
 * Questa classe incapsula una synergy::queue e fornisce solo i metodi
 * che vogliamo rendere accessibili lato Python.
 * 
 * 
 */

struct SYnergyCapabilities{
    bool cuda_support = false;
    bool rocm_support = false;
    bool level_zero_support = false;
    bool geopm_support = false;
    
    bool device_profiling = false;
    bool kernel_profiling = false;
    bool host_profiling = false;
    bool use_profiling_energy = false;
};




class SYnergy_Queue_Adapter {
private:
    synergy::queue q_;

public: 
    explicit SYnergy_Queue_Adapter(sycl::queue q)
        : q_(q) {}

    ~SYnergy_Queue_Adapter() {
        try{
            q_.wait_and_throw();
        } catch (...){   
        }
    }

    void wait() {
        q_.wait_and_throw();
    }

    std::string device_name() const {
        return q_.get_device().get_info<sycl::info::device::name>();
    }

    std::string backend_name() const {
        auto backend = q_.get_backend();
        switch (backend)
        {
        case sycl::backend::ext_oneapi_cuda:
            return "cuda";
        case sycl::backend::opencl:
            return "opencl";
        case sycl::backend::ext_oneapi_level_zero:
            return "level_zero";
        default:
            return "unknown";
        }
    }

#ifdef SYNERGY_DEVICE_PROFILING
    double device_energy_consumption() {
        return q_.device_energy_consumption();
    }
#endif

    synergy::queue& native_queue() {
        return q_;
    }

    SYnergyCapabilities capabilities() const {
    SYnergyCapabilities caps;

#ifdef SYNERGY_CUDA_SUPPORT
    caps.cuda_support = true;
#endif

#ifdef SYNERGY_ROCM_SUPPORT
    caps.rocm_support = true;
#endif

#ifdef SYNERGY_LZ_SUPPORT
    caps.level_zero_support = true;
#endif

#ifdef SYNERGY_GEOPM_SUPPORT
    caps.geopm_support = true;
#endif

#ifdef SYNERGY_DEVICE_PROFILING
    caps.device_profiling = true;
#endif

#ifdef SYNERGY_KERNEL_PROFILING
    caps.kernel_profiling = true;
#endif

#ifdef SYNERGY_HOST_PROFILING
    caps.host_profiling = true;
#endif

#ifdef SYNERGY_USE_PROFILING_ENERGY
    caps.use_profiling_energy = true;
#endif

    return caps;
}
};