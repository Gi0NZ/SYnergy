#include <iostream>
#include <vector>
#include <synergy.hpp>

/*int main() {
    try {
        synergy::queue q{sycl::gpu_selector_v};
        std::cout << "Synergy queue creata correttamente\n";
    } catch (const std::exception& e) {
        std::cerr << "Errore: " << e.what() << "\n";
    }
}*/
int main(int argc, char const *argv[])
{
    std::vector<float> a(2048);
    std::vector<float> b(2048);
    std::vector<float> c(2048);
    

    std::fill(a.begin(), a.end(), 1.0);
    std::fill(b.begin(), b.end(), 1.0);

    sycl::queue q(sycl::gpu_selector_v);
    
    sycl::buffer<float> a_buf(a.data(), 2048);
    sycl::buffer<float> b_buf(b.data(), 2048);
    sycl::buffer<float> c_buf(c.data(), 2048);


    q.submit([&] (sycl::handler& h){
        sycl::accessor<float, 1, sycl::access_mode::read> a_acc(a_buf, h);
        sycl::accessor<float, 1, sycl::access_mode::read> b_acc(b_buf, h);
        sycl::accessor<float, 1, sycl::access_mode::read_write> c_acc(c_buf, h);
        
        h.parallel_for(sycl::range<1>(2048), [=](sycl::id<1> idx){
            size_t i = idx[0];
            c_acc[i] = a_acc[i] + b_acc[i];
        });

    });

    q.wait_and_throw();
    sycl::host_accessor c_host{c_buf, sycl::read_only};
    for (size_t i = 0; i < 2048; i++)
      std::cout << c_host[i] << std::endl;

    return 0;
}
