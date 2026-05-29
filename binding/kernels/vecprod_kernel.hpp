#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>


/**
 * @brief Functor implementing a simple vector-product workload.
 *
 * This functor is used to instantiate a native SYCL kernel that computes:
 *
 * ``c[i] = a[i] * b[i]``
 *
 * The operation is repeated ``repeat`` times inside each work-item and then
 * averaged. This keeps the numerical result stable while allowing the workload
 * intensity to be increased for profiling and frequency scaling experiments.
 *
 * Expected kernel arguments:
 * - ``a``: first input vector
 * - ``b``: second input vector
 * - ``c``: output vector
 * - ``n``: number of elements
 * - ``repeat``: number of repeated multiplications per work-item
 */
struct SYnergyVecProdFunctor {
    /**
     * @brief Pointer to the first input vector.
     */
    float* a;

    /**
     * @brief Pointer to the second input vector.
     */
    float* b;

    /**
     * @brief Pointer to the output vector.
     */
    float* c;

    /**
     * @brief Number of elements processed by the kernel.
     */
    std::uint32_t n;

    /**
     * @brief Number of repeated multiplications performed by each work-item.
     */
    std::uint32_t repeat;

    /**
     * @brief Execute the vector-product operation for a single work-item.
     *
     * @param idx One-dimensional SYCL work-item id.
     */
    void operator()(sycl::id<1> idx) const {
        std::uint32_t i = static_cast<std::uint32_t>(idx[0]);

        if (i < n) {
            float x = a[i];
            float y = b[i];

            float acc = 0.0f;

            for (std::uint32_t r = 0; r < repeat; ++r) {
                acc += x * y;
            }

            c[i] = acc / static_cast<float>(repeat);
        }
    }
};