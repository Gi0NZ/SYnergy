#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include <sycl/sycl.hpp>
#include <synergy.hpp>

/**
 * @brief Capability flags exposed by the native SYnergy backend.
 *
 * This structure summarizes which SYnergy features were enabled at compile
 * time. It is returned to Python through pybind11 and converted into a
 * dictionary by the Python facade.
 */
struct SYnergyCapabilities {
    bool cuda_support = false;
    bool rocm_support = false;
    bool level_zero_support = false;
    bool geopm_support = false;

    bool device_profiling = false;
    bool kernel_profiling = false;
    bool host_profiling = false;
    bool use_profiling_energy = false;
};


/**
 * @brief Native C++ adapter exposing ``synergy::queue`` to Python.
 *
 * ``SYnergy_Queue_Adapter`` owns a native ``synergy::queue`` created from an
 * existing SYCL queue. The adapter is exposed to Python through pybind11 and
 * provides only the native operations needed by the Python ``SYnergyQueue``
 * facade.
 *
 * The adapter is intentionally thin. Kernel submission is handled by the
 * Cython submit bridge, while this class provides access to the native
 * ``synergy::queue``, synchronization, backend information, energy profiling
 * helpers and compile-time capability flags.
 */
class SYnergy_Queue_Adapter {
private:
    /**
     * @brief Native SYnergy queue used by the adapter.
     */
    synergy::queue q_;

public:
    /**
     * @brief Create the queue properties required by the SYnergy workflow.
     *
     * The queue is created as in-order and with profiling enabled. These
     * properties simplify event-based profiling and make the execution order
     * easier to reason about from the Python layer.
     *
     * @return SYCL property list containing ``in_order`` and
     * ``enable_profiling``.
     */
    static sycl::property_list make_synergy_properties() {
        return sycl::property_list{
            sycl::property::queue::in_order{},
            sycl::property::queue::enable_profiling{}
        };
    }

    /**
     * @brief Construct a SYnergy queue adapter from an existing SYCL queue.
     *
     * The input queue is normally created by dpctl on the Python side. Its
     * context and device are reused to construct the internal ``synergy::queue``.
     * This allows Python to keep using standard dpctl device selection while
     * delegating native execution and profiling to SYnergy.
     *
     * @param q SYCL queue created by dpctl.
     */
    explicit SYnergy_Queue_Adapter(sycl::queue q)
        : q_(q.get_context(), q.get_device(), make_synergy_properties()) {}

    /**
     * @brief Destroy the adapter and wait for pending native queue work.
     *
     * The destructor follows an RAII style: before releasing the native queue,
     * it attempts to wait for all pending operations. Exceptions are suppressed
     * because destructors should not throw.
     */
    ~SYnergy_Queue_Adapter() {
        try {
            q_.wait_and_throw();
        } catch (...) {
        }
    }

    /**
     * @brief Wait for all operations submitted to the native SYnergy queue.
     *
     * Unlike the destructor, this method propagates asynchronous SYCL
     * exceptions through ``wait_and_throw``.
     */
    void wait() {
        q_.wait_and_throw();
    }

    /**
     * @brief Return the name of the device associated with the queue.
     *
     * @return Human-readable SYCL device name.
     */
    std::string device_name() const {
        return q_.get_device().get_info<sycl::info::device::name>();
    }

    /**
     * @brief Return the backend name associated with the queue.
     *
     * @return Backend name such as ``"cuda"``, ``"opencl"``,
     * ``"level_zero"`` or ``"unknown"``.
     */
    std::string backend_name() const {
        auto backend = q_.get_backend();

        switch (backend) {
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

    /**
     * @brief Return the current device energy consumption.
     *
     * This method is available only when SYnergy is compiled with
     * ``SYNERGY_DEVICE_PROFILING`` enabled.
     *
     * @return Device energy consumption reported by SYnergy.
     *
     * @throws std::runtime_error If device profiling support was not enabled at
     * compile time.
     */
    double device_energy_consumption() const {
#ifdef SYNERGY_DEVICE_PROFILING
        return q_.device_energy_consumption();
#else
        throw std::runtime_error(
            "SYnergy was not compiled with SYNERGY_DEVICE_PROFILING"
        );
#endif
    }

    /**
     * @brief Return the energy consumption associated with a kernel event.
     *
     * This method is available only when SYnergy is compiled with
     * ``SYNERGY_KERNEL_PROFILING`` enabled.
     *
     * @param event SYCL event associated with a completed kernel execution.
     * @return Kernel energy consumption reported by SYnergy.
     *
     * @throws std::runtime_error If kernel profiling support was not enabled at
     * compile time.
     */
    double kernel_energy_consumption(const sycl::event& event) const {
#ifdef SYNERGY_KERNEL_PROFILING
        return q_.kernel_energy_consumption(event);
#else
        throw std::runtime_error(
            "SYnergy was not compiled with SYNERGY_KERNEL_PROFILING"
        );
#endif
    }

    /**
     * @brief Set target queue frequencies on the native SYnergy queue.
     *
     * This method is kept for native compatibility with SYnergy queue-level
     * frequency APIs. The current Python design manages device frequencies
     * through ``SYnergyDevice`` instead of passing frequency values during
     * kernel submission.
     *
     * @param uncore_frequency Target uncore frequency in MHz.
     * @param core_frequency Target core frequency in MHz.
     */
    void set_target_frequencies(
        unsigned int uncore_frequency,
        unsigned int core_frequency
    ) {
        q_.set_target_frequencies(uncore_frequency, core_frequency);
    }

    /**
     * @brief Return the underlying native SYnergy queue.
     *
     * This method is used by internal native helpers that need direct access to
     * the queue, for example precompiled test-kernel factories.
     *
     * @return Reference to the underlying ``synergy::queue``.
     */
    synergy::queue& native_queue() {
        return q_;
    }

    /**
     * @brief Return an integer handle to this adapter.
     *
     * The returned value is passed through Python/Cython and cast back to
     * ``SYnergy_Queue_Adapter`` by native bridge functions. This method is
     * intended for internal binding use only.
     *
     * @return Integer representation of the adapter pointer.
     */
    std::uintptr_t native_handle() {
        return reinterpret_cast<std::uintptr_t>(this);
    }

    /**
     * @brief Return compile-time SYnergy capability flags.
     *
     * The returned structure reports which backend and profiling features were
     * enabled when SYnergy was compiled.
     *
     * @return ``SYnergyCapabilities`` structure with backend and profiling
     * support flags.
     */
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