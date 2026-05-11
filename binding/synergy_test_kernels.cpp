#include "synergy_test_kernels.hpp"
#include "synergy_queue_adapter.hpp"

#include <sycl/sycl.hpp>
#include <syclinterface/dpctl_sycl_type_casters.hpp>

#include <cstdint>
#include <iostream>
#include <stdexcept>

class SYnergyBusyKernel;

/*
 * Functor esplicito per evitare problemi con l'ordine delle catture
 * della lambda quando il kernel viene recuperato come sycl::kernel
 * e gli argomenti vengono passati manualmente con set_arg.
 *
 * Ordine argomenti atteso:
 *   0. float* a
 *   1. float* b
 *   2. float* c
 *   3. uint32_t n
 *   4. uint32_t repeat
 */
struct SYnergyVectorAddFunctor {
    float* a;
    float* b;
    float* c;
    std::uint32_t n;
    std::uint32_t repeat;

    void operator()(sycl::id<1> idx) const {
        std::uint32_t i = static_cast<std::uint32_t>(idx[0]);

        if (i < n) {
            float x = a[i];
            float y = b[i];

            float acc = 0.0f;

            for (std::uint32_t r = 0; r < repeat; ++r) {
                acc += x + y;
            }

            c[i] = acc / static_cast<float>(repeat);
        }
    }
};

/*
 * Questa funzione serve solo a far compilare il kernel nella device image.
 * La submit vera verrà fatta da Python passando il sycl::kernel a SYnergyQueue.submit().
 */
void __synergy_define_busy_kernel(
    sycl::queue& q,
    float* a,
    float* b,
    float* c,
    std::uint32_t n,
    std::uint32_t repeat
) {
    SYnergyVectorAddFunctor functor{
        a,
        b,
        c,
        n,
        repeat
    };

    q.submit([&](sycl::handler& h) {
        h.parallel_for<SYnergyBusyKernel>(
            sycl::range<1>{n},
            functor
        );
    });
}

namespace {

SYnergy_Queue_Adapter* adapter_from_handle(std::uintptr_t handle) {
    if (handle == 0) {
        throw std::invalid_argument("SYnergy adapter handle is null.");
    }

    return reinterpret_cast<SYnergy_Queue_Adapter*>(handle);
}

} // namespace

extern "C" DPCTLSyclKernelRef SYnergyTest_CreateBusyKernel(
    std::uintptr_t AdapterHandle
) {
    try {
        auto* adapter = adapter_from_handle(AdapterHandle);

        auto& q = adapter->native_queue();
        auto ctx = q.get_context();
        auto dev = q.get_device();

        auto kernel_id = sycl::get_kernel_id<SYnergyBusyKernel>();

        auto bundle =
            sycl::get_kernel_bundle<sycl::bundle_state::executable>(
                ctx,
                {dev}
            );

        sycl::kernel kernel = bundle.get_kernel(kernel_id);

        return ::dpctl::syclinterface::wrap<sycl::kernel>(
            new sycl::kernel(kernel)
        );
    } catch (const std::exception& e) {
        std::cerr << "SYnergyTest_CreateBusyKernel failed: "
                  << e.what()
                  << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "SYnergyTest_CreateBusyKernel failed: unknown exception"
                  << std::endl;
        return nullptr;
    }
}