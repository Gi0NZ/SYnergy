#pragma once

#include <sycl/sycl.hpp>

#include <device.hpp>
#include <runtime.hpp>

#include <string>
#include <vector>


class SYnergy_Device_Adapter {
private:
    sycl::device sycl_device_;
    synergy::device synergy_device_;

public:
    explicit SYnergy_Device_Adapter(sycl::device dev)
        : sycl_device_(dev),
          synergy_device_(synergy::detail::runtime::synergy_device_from(dev)) {}

    std::string name() const {
        return sycl_device_.get_info<sycl::info::device::name>();
    }

    std::string backend_name() const {
       auto backend = sycl_device_.get_backend();

        if (backend == sycl::backend::ext_oneapi_level_zero) {
            return "level_zero";
        }

        if (backend == sycl::backend::opencl) {
            return "opencl";
        }

    #ifdef SYNERGY_CUDA_SUPPORT
        if (sycl_device_.is_gpu()) {
            return "cuda";
        }
    #endif

        return "unknown";
    }

    bool is_gpu() const {
        return sycl_device_.is_gpu();
    }

    bool is_cpu() const {
        return sycl_device_.is_cpu();
    }

    sycl::device native_sycl_device() const {
        return sycl_device_;
    }

    std::vector<unsigned int> supported_core_frequencies() {
        return synergy_device_.supported_core_frequencies();
    }

    std::vector<unsigned int> supported_uncore_frequencies() {
        return synergy_device_.supported_uncore_frequencies();
    }

    unsigned int current_core_frequency(bool cached = true) {
        return synergy_device_.get_core_frequency(cached);
    }

    unsigned int current_uncore_frequency(bool cached = true) {
        return synergy_device_.get_uncore_frequency(cached);
    }

    void set_core_frequency(unsigned int freq) {
        synergy_device_.set_core_frequency(freq);
    }

    void set_uncore_frequency(unsigned int freq) {
        synergy_device_.set_uncore_frequency(freq);
    }

    void set_frequencies(unsigned int core, unsigned int uncore) {
        synergy_device_.set_all_frequencies(core, uncore);
    }

    double power_usage() {
        return synergy_device_.get_power_usage();
    }

    double energy_usage() {
        return synergy_device_.get_energy_usage();
    }
};