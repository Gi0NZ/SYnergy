import dpctl
import importlib
from dataclasses import dataclass

_synergy_native = importlib.import_module("bindings._synergy_native")

#Qui ho messo cuda:gpu:0 - va poi reso generale

#Facade Python per l'adapter Synergy queue.
class SYnergyQueue:
    def __init__(self, device="cuda:gpu:0", queue=None):
        if queue is None:
            self.dpctl_queue = dpctl.SyclQueue(device)
        else:
            self.dpctl_queue = queue

        self._adapter = _synergy_native.SYnergy_Queue_Adapter(self.dpctl_queue)

    @property
    def device_name(self):
        return self._adapter.device_name()
    
    @property
    def backend_name(self):
        return self._adapter.backend_name()
    
    def wait(self):
        self._adapter.wait()

    def device_energy(self):
        return self._adapter.device_energy_consumption()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc, tb):
        self.wait()
        return False
    
    def capabilities(self):
        caps = self._adapter.capabilities()

        return SYnergyCapabilities(
            cuda_support=caps.cuda_support,
            rocm_support=caps.rocm_support,
            level_zero_support=caps.level_zero_support,
            geopm_support=caps.geopm_support,
            device_profiling=caps.device_profiling,
            kernel_profiling=caps.kernel_profiling,
            host_profiling=caps.host_profiling,
            use_profiling_energy=caps.use_profiling_energy,
        )
    
@dataclass
class SYnergyCapabilities:
    cuda_support: bool
    rocm_support: bool
    level_zero_support: bool
    geopm_support: bool

    device_profiling: bool
    kernel_profiling: bool
    host_profiling: bool
    use_profiling_energy: bool

    def as_dict(self):
        return {
            "cuda_support": self.cuda_support,
            "rocm_support": self.rocm_support,
            "level_zero_support": self.level_zero_support,
            "geopm_support": self.geopm_support,
            "device_profiling": self.device_profiling,
            "kernel_profiling": self.kernel_profiling,
            "host_profiling": self.host_profiling,
            "use_profiling_energy": self.use_profiling_energy,
        }
        