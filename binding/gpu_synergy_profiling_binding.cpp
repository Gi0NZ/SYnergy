#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <dpctl4pybind11.hpp>

#include "synergy_queue_adapter.hpp"
#include "synergy_device_adapter.hpp"

namespace py = pybind11;


/**
 * @brief Native pybind11 module exposing SYnergy adapters to Python.
 *
 * This module is imported from Python as ``bindings._synergy_native``.
 * It exposes thin native adapters used by the higher-level Python classes:
 *
 * - ``SYnergy_Queue_Adapter`` is used internally by ``SYnergyQueue`` to access
 *   native queue-level functionality.
 * - ``SYnergy_Device_Adapter`` is used internally by ``SYnergyDevice`` to
 *   access device information, frequency management and low-level energy data.
 *
 * End users are not expected to instantiate these classes directly. They
 * should normally use the Python facades ``SYnergyQueue`` and
 * ``SYnergyDevice``.
 */
PYBIND11_MODULE(_synergy_native, m) {
    m.doc() =
        "Internal native SYnergy bindings used by the Python facade layer.";

    /**
     * Capability flags returned by the native SYnergy queue adapter.
     *
     * The ``as_dict`` method is used by the Python layer to expose these
     * capabilities in a Python-friendly format.
     */
    py::class_<SYnergyCapabilities>(m, "SYnergyCapabilities")
        .def_readonly("cuda_support", &SYnergyCapabilities::cuda_support)
        .def_readonly("rocm_support", &SYnergyCapabilities::rocm_support)
        .def_readonly(
            "level_zero_support",
            &SYnergyCapabilities::level_zero_support
        )
        .def_readonly("geopm_support", &SYnergyCapabilities::geopm_support)
        .def_readonly(
            "device_profiling",
            &SYnergyCapabilities::device_profiling
        )
        .def_readonly(
            "kernel_profiling",
            &SYnergyCapabilities::kernel_profiling
        )
        .def_readonly("host_profiling", &SYnergyCapabilities::host_profiling)
        .def_readonly(
            "use_profiling_energy",
            &SYnergyCapabilities::use_profiling_energy
        )
        .def("as_dict", [](const SYnergyCapabilities& caps) {
            py::dict d;
            d["cuda_support"] = caps.cuda_support;
            d["rocm_support"] = caps.rocm_support;
            d["level_zero_support"] = caps.level_zero_support;
            d["geopm_support"] = caps.geopm_support;
            d["device_profiling"] = caps.device_profiling;
            d["kernel_profiling"] = caps.kernel_profiling;
            d["host_profiling"] = caps.host_profiling;
            d["use_profiling_energy"] = caps.use_profiling_energy;
            return d;
        });

    /**
     * Native queue adapter used by the Python SYnergyQueue facade.
     *
     * The constructor accepts a ``sycl::queue`` converted from a
     * ``dpctl.SyclQueue`` through ``dpctl4pybind11``. The adapter owns the
     * internal ``synergy::queue`` and exposes only the native operations needed
     * by Python.
     */
    py::class_<SYnergy_Queue_Adapter>(m, "SYnergy_Queue_Adapter")
        .def(py::init<sycl::queue>())
        .def("wait", &SYnergy_Queue_Adapter::wait)
        .def("device_name", &SYnergy_Queue_Adapter::device_name)
        .def("backend_name", &SYnergy_Queue_Adapter::backend_name)
        .def("capabilities", &SYnergy_Queue_Adapter::capabilities)
        .def(
            "device_energy_consumption",
            &SYnergy_Queue_Adapter::device_energy_consumption
        )
        .def(
            "kernel_energy_consumption",
            &SYnergy_Queue_Adapter::kernel_energy_consumption
        )
        .def("_native_handle", &SYnergy_Queue_Adapter::native_handle);

    /**
     * Native device adapter used by the Python SYnergyDevice facade.
     *
     * This adapter exposes device-level information and frequency-management
     * methods. Frequency control is intentionally kept at device level and is
     * not exposed through SYnergyQueue.submit.
     */
    py::class_<SYnergy_Device_Adapter>(m, "SYnergy_Device_Adapter")
        .def(py::init<sycl::device>())
        .def("name", &SYnergy_Device_Adapter::name)
        .def("backend_name", &SYnergy_Device_Adapter::backend_name)
        .def("is_gpu", &SYnergy_Device_Adapter::is_gpu)
        .def("is_cpu", &SYnergy_Device_Adapter::is_cpu)
        .def(
            "supported_core_frequencies",
            &SYnergy_Device_Adapter::supported_core_frequencies
        )
        .def(
            "supported_uncore_frequencies",
            &SYnergy_Device_Adapter::supported_uncore_frequencies
        )
        .def(
            "current_core_frequency",
            &SYnergy_Device_Adapter::current_core_frequency,
            py::arg("cached") = true
        )
        .def(
            "current_uncore_frequency",
            &SYnergy_Device_Adapter::current_uncore_frequency,
            py::arg("cached") = true
        )
        .def("set_core_frequency", &SYnergy_Device_Adapter::set_core_frequency)
        .def(
            "set_uncore_frequency",
            &SYnergy_Device_Adapter::set_uncore_frequency
        )
        .def("set_frequencies", &SYnergy_Device_Adapter::set_frequencies)
        .def("power_usage", &SYnergy_Device_Adapter::power_usage)
        .def("energy_usage", &SYnergy_Device_Adapter::energy_usage);
}