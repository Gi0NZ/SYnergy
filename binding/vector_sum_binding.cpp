#include <pybind11/pybind11.h>
#include <sycl/sycl.hpp>
#include <synergy.hpp>
#include <dpctl4pybind11.hpp>

#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "../samples/vector_sum/vector_sum.hpp"

namespace py = pybind11;

static float* get_float_usm_ptr(const py::object& array, const char* name) {
    if (!py::hasattr(array, "__sycl_usm_array_interface__")) {
        throw std::runtime_error(
            std::string(name) + " does not expose __sycl_usm_array_interface__"
        );
    }

    py::dict iface = array.attr("__sycl_usm_array_interface__").cast<py::dict>();

    if (!iface.contains("data")) {
        throw std::runtime_error(
            std::string(name) + " has no 'data' field in __sycl_usm_array_interface__"
        );
    }

    py::tuple data = iface["data"].cast<py::tuple>();

    if (data.size() < 1) {
        throw std::runtime_error(
            std::string(name) + " has an invalid 'data' field"
        );
    }

    auto ptr_value = data[0].cast<std::uintptr_t>();

    if (ptr_value == 0) {
        throw std::runtime_error(
            std::string(name) + " has a null USM pointer"
        );
    }

    return reinterpret_cast<float*>(ptr_value);
}

double run_vector_add(
    sycl::queue q,
    py::object a,
    py::object b,
    py::object c,
    size_t N
) {
    std::cout << "[PYBIND CODE] Executing Python vector addition call\n";

    float* ptr_a = get_float_usm_ptr(a, "a");
    float* ptr_b = get_float_usm_ptr(b, "b");
    float* ptr_c = get_float_usm_ptr(c, "c");

    Vector_Sum calculator(N);

    return calculator.execute(q, ptr_a, ptr_b, ptr_c);
}

PYBIND11_MODULE(synergy_custom, m) {
    m.def(
        "run_vector_add",
        &run_vector_add,
        "Esegue VectorSum su USM usando una dpctl.SyclQueue",
        py::arg("queue"),
        py::arg("a"),
        py::arg("b"),
        py::arg("c"),
        py::arg("N")
    );
}