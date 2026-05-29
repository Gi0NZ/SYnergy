#pragma once

#include <sycl/sycl.hpp>

#include <device.hpp>
#include <runtime.hpp>

#include <string>
#include <vector>


/**
 * @brief Native C++ adapter exposing SYnergy device operations to Python.
 *
 * ``SYnergy_Device_Adapter`` wraps a SYCL device and retrieves the
 * corresponding ``synergy::device`` from the SYnergy runtime. The adapter is
 * exposed to Python through pybind11 and is used internally by the Python
 * ``SYnergyDevice`` facade.
 *
 * This class is responsible for device-level functionality such as device
 * information, backend detection, supported frequency queries, current
 * frequency queries, frequency scaling and low-level power/energy readings.
 *
 * Kernel submission is intentionally not handled here. Submission and profiling
 * are managed by ``SYnergy_Queue_Adapter`` and the Cython submit bridge, while
 * frequency management remains a device-level responsibility.
 */
class SYnergy_Device_Adapter {
private:
    /**
     * @brief Original SYCL device provided by dpctl/Python.
     */
    sycl::device sycl_device_;

    /**
     * @brief Native SYnergy device associated with the SYCL device.
     */
    synergy::device synergy_device_;

public:
    /**
     * @brief Construct a native SYnergy device adapter.
     *
     * The constructor stores the SYCL device and asks the SYnergy runtime to
     * retrieve the corresponding ``synergy::device``. If the selected device is
     * visible to SYCL/dpctl but unsupported by SYnergy, the runtime may throw
     * an exception. This exception is handled at Python level by
     * ``SYnergyDevice``.
     *
     * @param dev SYCL device selected by dpctl or passed from Python.
     */
    explicit SYnergy_Device_Adapter(sycl::device dev)
        : sycl_device_(dev),
          synergy_device_(synergy::detail::runtime::synergy_device_from(dev)) {}

    /**
     * @brief Return the human-readable device name.
     *
     * @return Device name reported by SYCL.
     */
    std::string name() const {
        return sycl_device_.get_info<sycl::info::device::name>();
    }

    /**
     * @brief Return the backend name associated with the device.
     *
     * The method maps the SYCL backend to a Python-friendly string. CUDA is
     * detected when SYnergy was compiled with CUDA support and the device is a
     * GPU.
     *
     * @return Backend name such as ``"cuda"``, ``"opencl"``,
     * ``"level_zero"`` or ``"unknown"``.
     */
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

    /**
     * @brief Check whether the selected device is a GPU.
     *
     * @return True if the SYCL device is a GPU, false otherwise.
     */
    bool is_gpu() const {
        return sycl_device_.is_gpu();
    }

    /**
     * @brief Check whether the selected device is a CPU.
     *
     * @return True if the SYCL device is a CPU, false otherwise.
     */
    bool is_cpu() const {
        return sycl_device_.is_cpu();
    }

    /**
     * @brief Return the underlying SYCL device.
     *
     * This method is useful for internal native code that needs direct access
     * to the original SYCL device.
     *
     * @return Wrapped SYCL device.
     */
    sycl::device native_sycl_device() const {
        return sycl_device_;
    }

    /**
     * @brief Return the list of supported GPU core frequencies.
     *
     * @return Vector containing supported core frequencies in MHz.
     */
    std::vector<unsigned int> supported_core_frequencies() {
        return synergy_device_.supported_core_frequencies();
    }

    /**
     * @brief Return the list of supported GPU uncore frequencies.
     *
     * On NVIDIA devices, the uncore frequency typically corresponds to the
     * memory clock frequency exposed by the backend.
     *
     * @return Vector containing supported uncore frequencies in MHz.
     */
    std::vector<unsigned int> supported_uncore_frequencies() {
        return synergy_device_.supported_uncore_frequencies();
    }

    /**
     * @brief Return the current GPU core frequency.
     *
     * @param cached If true, return the cached value stored by SYnergy. If
     * false, query the backend directly.
     * @return Current core frequency in MHz.
     */
    unsigned int current_core_frequency(bool cached = true) {
        return synergy_device_.get_core_frequency(cached);
    }

    /**
     * @brief Return the current GPU uncore frequency.
     *
     * On NVIDIA devices, this value typically maps to the memory clock.
     *
     * @param cached If true, return the cached value stored by SYnergy. If
     * false, query the backend directly.
     * @return Current uncore frequency in MHz.
     */
    unsigned int current_uncore_frequency(bool cached = true) {
        return synergy_device_.get_uncore_frequency(cached);
    }

    /**
     * @brief Set the GPU core frequency.
     *
     * The requested value should belong to the list returned by
     * ``supported_core_frequencies``.
     *
     * @param freq Target core frequency in MHz.
     */
    void set_core_frequency(unsigned int freq) {
        synergy_device_.set_core_frequency(freq);
    }

    /**
     * @brief Set the GPU uncore frequency.
     *
     * The requested value should belong to the list returned by
     * ``supported_uncore_frequencies``.
     *
     * @param freq Target uncore frequency in MHz.
     */
    void set_uncore_frequency(unsigned int freq) {
        synergy_device_.set_uncore_frequency(freq);
    }

    /**
     * @brief Set both GPU core and uncore frequencies.
     *
     * This method applies both values through the native SYnergy device API.
     *
     * @param core Target core frequency in MHz.
     * @param uncore Target uncore frequency in MHz.
     */
    void set_frequencies(unsigned int core, unsigned int uncore) {
        synergy_device_.set_all_frequencies(core, uncore);
    }

    /**
     * @brief Return the current device power usage.
     *
     * @return Power usage reported by the native SYnergy backend.
     */
    double power_usage() {
        return synergy_device_.get_power_usage();
    }

    /**
     * @brief Return the current device energy usage.
     *
     * @return Energy usage reported by the native SYnergy backend.
     */
    double energy_usage() {
        return synergy_device_.get_energy_usage();
    }
};