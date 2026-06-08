#include <sycl/sycl.hpp>
#include <synergy.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr std::uint32_t REPEAT = 16384;      
constexpr std::size_t LOCAL_SIZE = 256;
constexpr int WARMUP_RUNS = 3;
constexpr int MEASURED_RUNS = 6;

const std::vector<std::size_t> SIZES = {
    1UL << 14,
    1UL << 16,
    1UL << 18,
    1UL << 20,
    1UL << 22,
    1UL << 24,
    1UL << 26,
    1UL << 30,
    
};

double now_ms_duration(std::chrono::high_resolution_clock::time_point start,
                       std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double kernel_time_ms(const sycl::event& e) {
    const auto start =
        e.get_profiling_info<sycl::info::event_profiling::command_start>();
    const auto end =
        e.get_profiling_info<sycl::info::event_profiling::command_end>();

    return static_cast<double>(end - start) / 1'000'000.0;
}

void initialize_vectors(sycl::queue& q, float* a, float* b, float* c, std::size_t n) {
    q.fill(a, 2.0f, n).wait();
    q.fill(b, 2.0f, n).wait();
    q.fill(c, 0.0f, n).wait();
}

sycl::event submit_vecprod_kernel(
    synergy::queue& q,
    float* a,
    float* b,
    float* c,
    std::size_t n,
    std::uint32_t repeat
) {
    if (n % LOCAL_SIZE != 0) {
        throw std::runtime_error("n must be divisible by LOCAL_SIZE");
    }

    return q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>{
                sycl::range<1>{n},
                sycl::range<1>{LOCAL_SIZE}
            },
            [=](sycl::nd_item<1> item) {
                const std::size_t i = item.get_global_linear_id();

                float acc = 0.0f;
                const float av = a[i];
                const float bv = b[i];

                for (std::uint32_t r = 0; r < repeat; ++r) {
                    acc += av * bv;
                }

                c[i] = acc / static_cast<float>(repeat);
            }
        );
    });
}

struct Metrics {
    double host_time_ms = 0.0;
    double kernel_time_ms = 0.0;
    double overhead_ms = 0.0;
    double overhead_ratio = 0.0;
    double device_energy_delta = 0.0;
    double kernel_energy = 0.0;
};

Metrics run_single_submission(
    synergy::queue& q,
    sycl::queue& base_q,
    float* a,
    float* b,
    float* c,
    std::size_t n
) {
    base_q.fill(c, 0.0f, n).wait();
    q.wait();

    const double energy_before = q.device_energy_consumption();

    const auto t0 = std::chrono::high_resolution_clock::now();

    sycl::event e = submit_vecprod_kernel(q, a, b, c, n, REPEAT);

    e.wait();

    const auto t1 = std::chrono::high_resolution_clock::now();

    const double energy_after = q.device_energy_consumption();

    Metrics m;
    m.host_time_ms = now_ms_duration(t0, t1);
    m.kernel_time_ms = kernel_time_ms(e);
    m.overhead_ms = m.host_time_ms - m.kernel_time_ms;
    m.overhead_ratio =
        m.host_time_ms > 0.0 ? m.overhead_ms / m.host_time_ms : 0.0;
    m.device_energy_delta = energy_after - energy_before;
    m.kernel_energy = q.kernel_energy_consumption(e);

    return m;
}

} // namespace

int main() {
    std::filesystem::create_directories("results");

    const std::string output_path = "results/cpp_vecprod_results.csv";
    std::ofstream csv(output_path);

    if (!csv.is_open()) {
        std::cerr << "Cannot open output CSV: " << output_path << "\n";
        return 1;
    }

    csv << "implementation,device,n,repeat,local_size,run_type,run_index,"
        << "host_time_ms,kernel_time_ms,overhead_ms,overhead_ratio,"
        << "device_energy_delta,kernel_energy\n";

    sycl::queue base_q{
        sycl::gpu_selector_v,
        sycl::property_list{sycl::property::queue::enable_profiling{}}
    };

    synergy::queue q{base_q};

    const std::string device_name =
        base_q.get_device().get_info<sycl::info::device::name>();

    std::cout << "Used device: " << device_name << "\n";

    for (std::size_t n : SIZES) {
        std::cout << "\n===== SIZE n=" << n << " =====\n";

        float* a = sycl::malloc_shared<float>(n, base_q);
        float* b = sycl::malloc_shared<float>(n, base_q);
        float* c = sycl::malloc_shared<float>(n, base_q);

        if (!a || !b || !c) {
            std::cerr << "USM allocation failed for n=" << n << "\n";
            return 1;
        }

        initialize_vectors(base_q, a, b, c, n);

        std::cout << "Warmup Phase\n";
        for (int warmup = 0; warmup < WARMUP_RUNS; ++warmup) {
            (void)run_single_submission(q, base_q, a, b, c, n);
        }

        std::cout << "Measured Phase\n";
        for (int run = 0; run < MEASURED_RUNS; ++run) {
            Metrics m = run_single_submission(q, base_q, a, b, c, n);

            csv << "cpp,"
                << "\"" << device_name << "\"" << ","
                << n << ","
                << REPEAT << ","
                << LOCAL_SIZE << ","
                << "measured,"
                << run << ","
                << m.host_time_ms << ","
                << m.kernel_time_ms << ","
                << m.overhead_ms << ","
                << m.overhead_ratio << ","
                << m.device_energy_delta << ","
                << m.kernel_energy << "\n";

            std::cout
                << "run=" << run
                << " host=" << m.host_time_ms << " ms"
                << " kernel=" << m.kernel_time_ms << " ms"
                << " energy=" << m.device_energy_delta
                << "\n";
        }

        sycl::free(a, base_q);
        sycl::free(b, base_q);
        sycl::free(c, base_q);

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    std::cout << "\nCSV saved to: " << output_path << "\n";
    return 0;
}