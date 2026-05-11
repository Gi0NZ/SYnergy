import dpctl
import importlib
from dataclasses import dataclass

_synergy_native = importlib.import_module("bindings._synergy_native")


#Facade Python per l'adapter Synergy queue.
"""class SYnergyQueue:
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
        """


#Implementazione della classe SYnergyQueue non solo come estensione di SyclQueue ma acnhe come composition, ovvero creazione interna dell'adapter. Questo permette 
#di mantenere le funzionalità di SyclQueue e allo stesso tempo di definire un insieme di funzioni, tra cui la submit che vogliamo, che permette di gestire profiling 
#energetico, frequency scaling e così via.

class SYnergyQueue(dpctl.SyclQueue):

    def __new__(cls, *args, property=None, **kwargs):
        property = cls._ensure_synergy_properties(property)

        return super().__new__(cls, *args, property=property, **kwargs)



    def __init__(self, *args, property=None, **kwargs):

        #adapter C++ contenente la synergy::queue
        self._adapter = _synergy_native.SYnergy_Queue_Adapter(self)
        self._last_event = None
        self._profile_log = []

    @staticmethod
    def _ensure_synergy_properties(prop):

        """
            Garantisce la costruzione della queue con proprietà utili/necessarie al profiling
        """

        if prop is None:
            return("in_order", "enable_profiling")
        
        if isinstance(prop, str):
            props = (prop,)
        else:
            props = tuple(prop)

        if "in_order" not in props:
            props += ("in_order",)

        if "enable_profiling" not in props:
            props += ("enable_profiling",)

        return props


    # L'attuale implementazione del metodo submit fa l'override dell'originale: in particolare aggiunge un flag use_synergy che, se impostato a ver, invia la submit
    # al backend implementato per synergy, altrimenti utilizza super.submit, usando i metodi base di SyclQueue.
    def submit(self, kernel, args, gS, lS=None, dEvents=None, 
               *,
                use_device_profiling=False, 
                use_kernel_profiling=False, 
                uncore_frequency=None, 
                core_frequency=None,):
        

        """
            Semantica:

            - uncore_frequency e core frequency None:
                si usa synergy::queue::submit(cgh)

            - almeno una tra core e uncore settate
                si usa synergy::queue::submit(
                uncore_frequency,
                core_frequency,
                cgh
            )
        """

        """
            Nota: i campi use_device/kernel_profiling non indicano la scelta di backend, si usa sempre synergy, indicano semplicemente quale misura raccogliere e salvare
    
        """
        
        if dEvents is None:
            dEvents = []

        use_frequency_scaling = (
            uncore_frequency is not None or core_frequency is not None    
        )

        normalized_uncore_frequency = (
            0 if uncore_frequency is None else int(uncore_frequency)
        )
        normalized_core_frequency = (
            0 if core_frequency is None else int(core_frequency)
        )

        bridge = self._load_submit_bridge()

        event, profile = bridge.submit( queue=self,
            adapter=self._adapter,
            kernel=kernel,
            args=args,
            gS=gS,
            lS=lS,
            dEvents=dEvents,
            use_device_profiling=bool(use_device_profiling),
            use_kernel_profiling=bool(use_kernel_profiling),
            use_frequency_scaling=bool(use_frequency_scaling),
            uncore_frequency=normalized_uncore_frequency,
            core_frequency=normalized_core_frequency,
        )
        self._last_event = event
        self._profile_log.append(profile)

        return event
    
    def _load_submit_bridge(self):
        """
        Carica il futuro bridge Cython della submit SYnergy.

        Verrà implementato nella fase successiva come modulo:
            bindings._synergy_submit

        Per ora questo metodo serve a fissare chiaramente il punto di
        collegamento tra Python e backend nativo.
        """
        try:
            return importlib.import_module("bindings._synergy_submit")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Il bridge bindings._synergy_submit non è ancora disponibile. "
                "Completa la Fase 2 per implementare la submit SYnergy reale."
            ) from exc
    
    
    def wait(self):
    # Aspetta sia eventuali operazioni dpctl sia quelle SYnergy.
        super().wait()
        self._adapter.wait()

    @property
    def synergy_device_name(self):
        return self._adapter.device_name()

    @property
    def synergy_backend_name(self):
        return self._adapter.backend_name()
    
    @property
    def last_event(self):
        return self._last_event

    @property
    def last_profile(self):
        return self._profile_log[-1] if self._profile_log else None

    @property
    def profile_log(self):
        return list(self._profile_log)

    def capabilities(self):
        return self._adapter.capabilities().as_dict()

    def device_energy_consumption(self):
        return self._adapter.device_energy_consumption()

    def kernel_energy_consumption(self, event):
        return self._adapter.kernel_energy_consumption(event)
    