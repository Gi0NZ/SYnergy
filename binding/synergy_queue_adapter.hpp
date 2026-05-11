#pragma once

#include <sycl/sycl.hpp>
#include <synergy.hpp>
#include <string>

/**
 * @brief Adapter C++ per esporre synergy::queue a Python tramite pybind11.
 *
 * Questa classe incapsula una synergy::queue e fornisce solo i metodi
 * che vogliamo rendere accessibili lato Python.
 * 
 * 
 */

struct SYnergyCapabilities{
    bool cuda_support = false;
    bool rocm_support = false;
    bool level_zero_support = false;
    bool geopm_support = false;
    
    bool device_profiling = false;
    bool kernel_profiling = false;
    bool host_profiling = false;
    bool use_profiling_energy = false;
};



/**
 * Questa classe è una "sottoclasse" di synergy::queue che permette di esporre a python solo i metodi desiderati
 */
class SYnergy_Queue_Adapter {
private:
    synergy::queue q_;  //Elemento chiave di interazione con SYnergy

public: 

    //setting delle impostazioni di default della synergy queue
    static sycl::property_list make_synergy_properties() {
        return sycl::property_list{
            sycl::property::queue::in_order{},
            sycl::property::queue::enable_profiling{}
        };
    }
    //Costruttore: riceve una sycl::queue dal binding python (dpctl) usata per essere inglobata dalla synergy queue.
    // Ci permette di gestire device selection ed altro con le chiamate standard dpctl, facendo però interagire il backend con SYnergy
    explicit SYnergy_Queue_Adapter(sycl::queue q)  
        : q_(q.get_context(), q.get_device(), make_synergy_properties()) {}         
        
        
    //Sezione RAII: la risorsa viene gestita automaticamente in base al ciclo di vita dell'oggetto: alla distruzione dell'adapter si tenta la wait_and_throw
    ~SYnergy_Queue_Adapter() {
        try{
            q_.wait_and_throw();
        } catch (...){   
        }
    }

    void wait() {
        q_.wait_and_throw();
    }

    std::string device_name() const {
        return q_.get_device().get_info<sycl::info::device::name>();
    }

    std::string backend_name() const {
        auto backend = q_.get_backend();
        switch (backend)
        {
        case sycl::backend::ext_oneapi_cuda:
            return "cuda";
        case sycl::backend::opencl:
            return "opencl";
        case sycl::backend::ext_oneapi_level_zero:
            return "level_zero";
        default:
            return "unknown";
        }
    }

    double device_energy_consumption() const {
 #ifdef SYNERGY_DEVICE_PROFILING
        return q_.device_energy_consumption();
#else
        throw std::runtime_error(
            "SYnergy was not compiled with SYNERGY_DEVICE_PROFILING"
        );
#endif
    }

    double kernel_energy_consumption(const sycl::event& event) const {
#ifdef SYNERGY_KERNEL_PROFILING
        return q_.kernel_energy_consumption(event);
#else
        throw std::runtime_error(
            "SYnergy was not compiled with SYNERGY_KERNEL_PROFILING"
        );
#endif
    }



    void set_target_frequencies(unsigned int uncore_frequency, unsigned int core_frequency) {
        q_.set_target_frequencies(uncore_frequency, core_frequency);
    }

    synergy::queue& native_queue() {
        return q_;
    }

    std::uintptr_t native_handle(){
        return reinterpret_cast<std::uintptr_t>(this);
    }

    SYnergyCapabilities capabilities() const {
    SYnergyCapabilities caps;

#ifdef SYNERGY_CUDA_SUPPORT
    caps.cuda_support = true;
#endif

#ifdef SYNERGY_ROCM_SUPPORT
    caps.rocm_support = true;
#endif

#ifdef SYNERGY_LZ_SUPPORT
    caps.level_zero_support = true;
#endif

#ifdef SYNERGY_GEOPM_SUPPORT
    caps.geopm_support = true;
#endif

#ifdef SYNERGY_DEVICE_PROFILING
    caps.device_profiling = true;
#endif

#ifdef SYNERGY_KERNEL_PROFILING
    caps.kernel_profiling = true;
#endif

#ifdef SYNERGY_HOST_PROFILING
    caps.host_profiling = true;
#endif

#ifdef SYNERGY_USE_PROFILING_ENERGY
    caps.use_profiling_energy = true;
#endif

    return caps;
}
/*
    sycl::event submit(
    const sycl::kernel& kernel,
    const std::vector<KernelArg>& args,
    const std::vector<size_t>& gS,
    const std::optional<std::vector<size_t>>& lS,
    const std::vector<sycl::event>& dep_events,
    unsigned int uncore_frequency,
    unsigned int core_frequency,
    bool use_frequency_scaling
) {
    validate_range(gS, lS);

    auto command_group = [&](sycl::handler& h) {
        for (const auto& ev : dep_events) {
            h.depends_on(ev);
        }

        set_kernel_args(h, args);
        submit_parallel_for(h, kernel, gS, lS);
    };

    if (use_frequency_scaling) {
        return q_.submit(
            uncore_frequency,
            core_frequency,
            command_group
        );
    }

    return q_.submit(command_group);
    }*/
};