#include <pybind11/pybind11.h>
#include <dpctl4pybind11.hpp>
#include "synergy_queue_adapter.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_synergy_native, m) {

    py::class_<SYnergyCapabilities>(m, "SYnergyCapabilities")
    .def_readonly("cuda_support", &SYnergyCapabilities::cuda_support)
    .def_readonly("rocm_support", &SYnergyCapabilities::rocm_support)
    .def_readonly("level_zero_support", &SYnergyCapabilities::level_zero_support)
    .def_readonly("geopm_support", &SYnergyCapabilities::geopm_support)
    .def_readonly("device_profiling", &SYnergyCapabilities::device_profiling)
    .def_readonly("kernel_profiling", &SYnergyCapabilities::kernel_profiling)
    .def_readonly("host_profiling", &SYnergyCapabilities::host_profiling)
    .def_readonly("use_profiling_energy", &SYnergyCapabilities::use_profiling_energy)
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
    py::class_<SYnergy_Queue_Adapter>(m,"SYnergy_Queue_Adapter")
    .def(py::init<sycl::queue>())
    .def("wait", &SYnergy_Queue_Adapter::wait)
    .def("device_name", &SYnergy_Queue_Adapter::device_name)
    .def("backend_name", &SYnergy_Queue_Adapter::backend_name)
    .def("capabilities", &SYnergy_Queue_Adapter::capabilities)

#ifdef SYNERGY_DEVICE_PROFILING
    .def("device_energy_consumption", &SYnergy_Queue_Adapter::device_energy_consumption)
#endif
    ;
}