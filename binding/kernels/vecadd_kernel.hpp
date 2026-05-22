#include <sycl/sycl.hpp>
#include <cstdint>


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