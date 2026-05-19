/*#include <iostream>
#include <vector>
#include <synergy.hpp>

// questa sarebbe la classe equivalente ad una di quelle di synergy 

#define N 8192*2048
int main() {
    try {
        synergy::queue q{sycl::gpu_selector_v};
        std::cout << "Synergy queue creata correttamente\n";
    } catch (const std::exception& e) {
        std::cerr << "Errore: " << e.what() << "\n";
    }
}
int main(int argc, char const *argv[])
{
    std::vector<float> a(N);
    std::vector<float> b(N);
    std::vector<float> c(N);
    

    std::fill(a.begin(), a.end(), 1.0);
    std::fill(b.begin(), b.end(), 1.0);

    synergy::queue q(sycl::gpu_selector_v);
    std::cout << "Freq. : " << q.get_synergy_device().get_core_frequency() << std::endl;
    sycl::buffer<float> a_buf(a.data(), N);
    sycl::buffer<float> b_buf(b.data(), N);
    sycl::buffer<float> c_buf(c.data(), N);
    
    //q.get_synergy_device().set_core_frequency(2010);
    sycl::event e = q.submit([&] (sycl::handler& h){
        sycl::accessor<float, 1, sycl::access_mode::read> a_acc(a_buf, h);
        sycl::accessor<float, 1, sycl::access_mode::read> b_acc(b_buf, h);
        sycl::accessor<float, 1, sycl::access_mode::read_write> c_acc(c_buf, h);
        
        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx){
            size_t i = idx[0];
            c_acc[i] = a_acc[i] + b_acc[i];
        });

    });
    std::cout << "Freq. : " << q.get_synergy_device().get_core_frequency() << std::endl;
    q.wait_and_throw();
    sycl::host_accessor c_host{c_buf, sycl::read_only};
    for (size_t i = 0; i < 3; i++)
      std::cout << c_host[i] << std::endl;

    std::cout << "Device (q1) Energy consumption: " << q.device_energy_consumption() << " j\n";
    std::cout << "Kernel energy consumption: " << q.kernel_energy_consumption(e) << " j\n";

*/

#include <sycl/sycl.hpp>

int main(int argc, char const *argv[])
{
    sycl::queue q;

    sycl::queue q2(q);

    sycl::device dev = q2.get_device();
    std::cout << "dev" << dev.get_info<sycl::info::device::name>();
    return 0;
}
