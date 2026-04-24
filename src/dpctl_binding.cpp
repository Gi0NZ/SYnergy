#include <pybind11/pybind11.h>
#include <sycl/sycl.hpp>
#include <synergy.hpp>
#include <memory>
#include <stdexcept>

namespace py = pybind11;

class SYnergyQueueAdapter {
private:
    std::unique_ptr<synergy::queue> sy_queue;
    double start_energy = 0.0;
    bool hw_supported = false;

public:
    SYnergyQueueAdapter(size_t sycl_queue_addr) {
        if (sycl_queue_addr == 0) return;
        sycl::queue* raw_queue = reinterpret_cast<sycl::queue*>(sycl_queue_addr);
        
        try {
            // Tenta di inizializzare l'integrazione hardware
            sy_queue = std::make_unique<synergy::queue>(*raw_queue);
            hw_supported = true;
        } catch (...) {
            // Se l'hardware respinge la richiesta (NVML Not Supported),
            // il C++ non muore. Rimane vivo ma in modalità "solo calcolo".
            hw_supported = false;
        }
    }

    void apply_frequencies(double core, double uncore) {
        if (!hw_supported) throw std::runtime_error("HW_NOT_SUPPORTED");
        sy_queue->set_target_frequencies(uncore, core);
    }

    void prepare_profiling() {
        if (!hw_supported) throw std::runtime_error("HW_NOT_SUPPORTED");
        start_energy = sy_queue->get_synergy_device().get_energy_usage();
    }

    double finalize_profiling() {
        if (!hw_supported) throw std::runtime_error("HW_NOT_SUPPORTED");
        double end_energy = sy_queue->get_synergy_device().get_energy_usage();
        return end_energy - start_energy;
    }

    double get_current_energy() {
        if (!hw_supported) throw std::runtime_error("HW_NOT_SUPPORTED");
        return sy_queue->get_synergy_device().get_energy_usage();
    }
};

PYBIND11_MODULE(synergy_backend, m) {
    m.doc() = "Adapter C++ per l'integrazione di SYnergy con Python/DPCTL";

    py::class_<SYnergyQueueAdapter>(m, "PySynergyQueue")
        .def(py::init<size_t>(), py::arg("sycl_queue_addr"))
        .def("apply_frequencies", &SYnergyQueueAdapter::apply_frequencies, 
             py::arg("core"), py::arg("uncore"))
        .def("prepare_profiling", &SYnergyQueueAdapter::prepare_profiling)
        .def("finalize_profiling", &SYnergyQueueAdapter::finalize_profiling)
        .def("get_current_energy", &SYnergyQueueAdapter::get_current_energy);
}

