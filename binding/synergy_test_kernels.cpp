#include "synergy_test_kernels.hpp"
#include "synergy_queue_adapter.hpp"
#include "kernels/vecprod_kernel.hpp"
#include "kernels/vecadd_kernel.hpp"

#include <sycl/sycl.hpp>
#include <syclinterface/dpctl_sycl_type_casters.hpp>

#include <cstdint>
#include <iostream>
#include <stdexcept>


/**
 * @brief Kernel name used to identify the native vector-add kernel.
 *
 * The class is used only as a SYCL kernel name. The actual computation is
 * implemented by SYnergyVectorAddFunctor.
 */
class SYnergyVecAddKernel;

/**
 * @brief Kernel name used to identify the native vector-product kernel.
 *
 * The class is used only as a SYCL kernel name. The actual computation is
 * implemented by SYnergyVecProdFunctor.
 */
class SYnergyVecProdKernel;


/**
 * @brief Define the vector-add kernel so that it is emitted in the device image.
 *
 * This function is not part of the Python-facing execution path. Its purpose
 * is to make the SYCL compiler instantiate the vector-add kernel and include
 * it in the executable kernel bundle.
 *
 * At runtime, Python does not call this function to execute the kernel.
 * Instead, the C factory function ``SYnergyTest_CreateVecAddKernel`` retrieves
 * the precompiled kernel from the bundle and returns it as a DPCTL kernel
 * reference.
 *
 * Expected kernel arguments:
 * - ``float* a``
 * - ``float* b``
 * - ``float* c``
 * - ``uint32_t n``
 * - ``uint32_t repeat``
 *
 * @param q SYCL queue used only for kernel definition/instantiation.
 * @param a First input vector.
 * @param b Second input vector.
 * @param c Output vector.
 * @param n Number of elements.
 * @param repeat Number of repeated operations per work-item.
 */
void __synergy_define_vecadd_kernel(
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
        h.parallel_for<SYnergyVecAddKernel>(
            sycl::range<1>{n},
            functor
        );
    });
}


/**
 * @brief Define the vector-product kernel so that it is emitted in the device image.
 *
 * This function mirrors ``__synergy_define_vecadd_kernel`` for the
 * vector-product workload. It is used to force kernel instantiation at compile
 * time, while actual execution from Python happens through the recovered
 * ``sycl::kernel`` object.
 *
 * Expected kernel arguments:
 * - ``float* a``
 * - ``float* b``
 * - ``float* c``
 * - ``uint32_t n``
 * - ``uint32_t repeat``
 *
 * @param q SYCL queue used only for kernel definition/instantiation.
 * @param a First input vector.
 * @param b Second input vector.
 * @param c Output vector.
 * @param n Number of elements.
 * @param repeat Number of repeated operations per work-item.
 */
void __synergy_define_vecprod_kernel(
    sycl::queue& q,
    float* a,
    float* b,
    float* c,
    std::uint32_t n,
    std::uint32_t repeat
) {
    SYnergyVecProdFunctor functor{
        a,
        b,
        c,
        n,
        repeat
    };

    q.submit([&](sycl::handler& h) {
        h.parallel_for<SYnergyVecProdKernel>(
            sycl::range<1>{n},
            functor
        );
    });
}


namespace {

/**
 * @brief Recover a native queue adapter from an integer handle.
 *
 * The handle is created by ``SYnergy_Queue_Adapter::native_handle`` and passed
 * through Python/Cython as an integer value. This helper validates the handle
 * and casts it back to the native adapter type.
 *
 * @param handle Integer representation of a ``SYnergy_Queue_Adapter`` pointer.
 * @return Pointer to the native queue adapter.
 *
 * @throws std::invalid_argument If the handle is null.
 */
SYnergy_Queue_Adapter* adapter_from_handle(std::uintptr_t handle) {
    if (handle == 0) {
        throw std::invalid_argument("SYnergy adapter handle is null.");
    }

    return reinterpret_cast<SYnergy_Queue_Adapter*>(handle);
}

} // namespace


extern "C" DPCTLSyclKernelRef SYnergyTest_CreateVecAddKernel(
    std::uintptr_t AdapterHandle
) {
    try {
        auto* adapter = adapter_from_handle(AdapterHandle);

        /*
         * Reuse the context and device of the native SYnergy queue so that the
         * recovered kernel is compatible with the queue used by Python.
         */
        auto& q = adapter->native_queue();
        auto ctx = q.get_context();
        auto dev = q.get_device();

        /*
         * Retrieve the precompiled kernel from the executable bundle and wrap
         * it as a DPCTL kernel reference. The Python layer will expose it as a
         * dpctl.program.SyclKernel.
         */
        auto kernel_id = sycl::get_kernel_id<SYnergyVecAddKernel>();

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
        std::cerr << "SYnergyTest_CreateVecAddKernel failed: "
                  << e.what()
                  << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "SYnergyTest_CreateVecAddKernel failed: unknown exception"
                  << std::endl;
        return nullptr;
    }
}


extern "C" DPCTLSyclKernelRef SYnergyTest_CreateVecprodKernel(
    std::uintptr_t AdapterHandle
) {
    try {
        auto* adapter = adapter_from_handle(AdapterHandle);

        /*
         * The kernel must be recovered for the same context/device pair used
         * by the queue adapter, otherwise the returned kernel may not be
         * submit-compatible with the Python queue.
         */
        auto& q = adapter->native_queue();
        auto ctx = q.get_context();
        auto dev = q.get_device();

        /*
         * Retrieve the vector-product kernel from the executable bundle and
         * return it as a DPCTL-compatible kernel reference.
         */
        auto kernel_id = sycl::get_kernel_id<SYnergyVecProdKernel>();

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
        std::cerr << "SYnergyTest_CreateVecprodKernel failed: "
                  << e.what()
                  << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "SYnergyTest_CreateVecprodKernel failed: unknown exception"
                  << std::endl;
        return nullptr;
    }
}