import dpctl 
import synergy_backend

class SynergyQueue(dpctl.SyclQueue):
    """Implementazione della synergy queue """

    def __init__(self, *args, **kwargs):
        #inizializzazione della coda SYCL
        super().__init__()

        #estrazione del puntatore alla coda SYCL
        sycl_pointer = self.addressof_ref()

        #inizializzazione adapter
        self._synergy_adapter = synergy_backend.PySynergyQueue(sycl_pointer)

        print(f"DEBUG PY: Coda SynergyQueue inizializzata all'indirizzo_ {sycl_pointer}")


    def smart_submit(self, kernel_func, *args, profiling=False, core_freq=0, uncore_freq=0, **kwargs):
        """NOTA: Implementare dopo diversi metodi che vengono chiamati in base agli elementi che vengono inseriti"""
        if core_freq > 0 or uncore_freq > 0:
            try:
                print("f[SYnergy]: impostazione frequenze")
                self._synergy_adapter.set_target_frequencies(core_freq, uncore_freq)
            except RuntimeError as e:
                print("Impossibile fissare le frequenze")

        if profiling:
            try:
                print("Profiling attivato: registrazione energia di partenza")
                self._synergy_adapter.prepare_profiling();
            except RuntimeError as e:
                print("Impossibile accedere ai sensori di sistema")
                profiling = False


        print("Sottomissione kernel al device")
        event = super().submit(kernel_func, *args, **kwargs)

        event.wait()

        if profiling:
            try:
                onsumo_joule = self._synergy_adapter.finalize_profiling()
                print("Il consumo netto è pari a {consumo_joule: .4f} Joule")
            except RuntimeError:
                pass
        
        return event
    
    def get_current_energy(self):
        """lettura energia istantanea del device"""
        return self._synergy_adapter.get_current_energy()