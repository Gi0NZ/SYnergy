#pragma once

#include <sycl/sycl.hpp>
#include <synergy.hpp>

#include <chrono>
#include <cstddef>
#include <vector>

struct VectorSumProfileResult {
    double device_energy_before_j = 0.0;
    double device_energy_after_j = 0.0;
    double device_energy_delta_j = 0.0;

    double kernel_energy_j = 0.0;
    double wall_time_ms = 0.0;
    double kernel_time_ms = 0.0;

    float first_value = 0.0f;
};

class Vector_Sum {
private:
    std::size_t size;

public:
    explicit Vector_Sum(std::size_t N) : size(N) {}

    VectorSumProfileResult execute(synergy::queue& q) {
        std::vector<float> a(size, 1.0f);
        std::vector<float> b(size, 1.0f);
        std::vector<float> c(size, 0.0f);

        sycl::buffer<float, 1> a_buf(a.data(), sycl::range<1>(size));
        sycl::buffer<float, 1> b_buf(b.data(), sycl::range<1>(size));
        sycl::buffer<float, 1> c_buf(c.data(), sycl::range<1>(size));

        VectorSumProfileResult result;

#ifdef SYNERGY_DEVICE_PROFILING
        result.device_energy_before_j = q.device_energy_consumption();
#endif

        auto wall_start = std::chrono::high_resolution_clock::now();

        sycl::event e = q.submit([&](sycl::handler& h) {
            sycl::accessor<float, 1, sycl::access_mode::read> a_acc(a_buf, h);
            sycl::accessor<float, 1, sycl::access_mode::read> b_acc(b_buf, h);
            sycl::accessor<float, 1, sycl::access_mode::write> c_acc(c_buf, h);

            h.parallel_for<class VectorSumKernel>(
                sycl::range<1>(size),
                [=](sycl::id<1> idx) {
                    std::size_t i = idx[0];
                    c_acc[i] = a_acc[i] + b_acc[i];
                }
            );
        });

        q.wait_and_throw();

        auto wall_end = std::chrono::high_resolution_clock::now();

        result.wall_time_ms =
            std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

#ifdef SYNERGY_DEVICE_PROFILING
        result.device_energy_after_j = q.device_energy_consumption();
        result.device_energy_delta_j =
            result.device_energy_after_j - result.device_energy_before_j;
#endif

#ifdef SYNERGY_KERNEL_PROFILING
        result.kernel_energy_j = q.kernel_energy_consumption(e);

        auto kernel_start =
            e.get_profiling_info<sycl::info::event_profiling::command_start>();

        auto kernel_end =
            e.get_profiling_info<sycl::info::event_profiling::command_end>();

        result.kernel_time_ms =
            static_cast<double>(kernel_end - kernel_start) / 1'000'000.0;
#endif

        {
            sycl::host_accessor c_host(c_buf, sycl::read_only);
            result.first_value = c_host[0];
        }

        return result;
    }
};