import dpctl
import dpctl.program as dpctl_program
import importlib
from pathlib import Path
from dataclasses import dataclass

_synergy_native = importlib.import_module("bindings._synergy_native")

#Implementazione della classe SYnergyQueue non solo come estensione di SyclQueue ma acnhe come composition, ovvero creazione interna dell'adapter. Questo permette 
#di mantenere le funzionalità di SyclQueue e allo stesso tempo di definire un insieme di funzioni, tra cui la submit che vogliamo, che permette di gestire profiling 
#energetico, frequency scaling e così via.

class SYnergyQueue(dpctl.SyclQueue):

    def __new__(
    cls,
    *args,
    property=None,
    execution_backend="synergy",
    allow_fallback=False,
    **kwargs,
    ):
        property = cls._ensure_synergy_properties(property)
        args, _ = cls._unwrap_synergy_device_args(args)
        return super().__new__(cls, *args, property=property, **kwargs)

    def __init__(
    self,
    *args,
    property=None,
    execution_backend="synergy",
    allow_fallback=False,
    **kwargs,
    ):
        if execution_backend not in ("synergy", "dpctl", "auto"):
            raise ValueError(
                "execution_backend must be one of: 'synergy', 'dpctl', 'auto'."
            )
        args, synergy_device = self._unwrap_synergy_device_args(args)
        self._synergy_device = synergy_device

        self._adapter = None
        self._adapter_error = None
        self._execution_backend = execution_backend
        self._last_event = None
        self._profile_log = []

        

        if execution_backend == "dpctl":
            return

        try:
            self._adapter = _synergy_native.SYnergy_Queue_Adapter(self)
            self._execution_backend = "synergy"

        except Exception as exc:
            self._adapter_error = exc

            if execution_backend == "auto" or allow_fallback:
                self._execution_backend = "dpctl"
            else:
                raise

    @staticmethod
    def _ensure_synergy_properties(prop):

        """
            Garantisce la costruzione della SYnergyQueue con proprietà utili/necessarie al profiling - fondamentali per SYnergy
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
    

    @staticmethod
    def _normalize_range(name, value, required=True):
        if value is None:
            if required:
                raise ValueError(f"{name} is required")
            return None
            
        if not isinstance(value, (list,tuple)):
            raise TypeError(f"{name} must be a list or a tuple with 1, 2 or 3 dimension.")
        
        if len(value) not in (1, 2, 3):
            raise ValueError(f"{name} must have 1, 2 or 3 dimensions.")
        
        normalized = []

        for dim in value:
            dim = int(dim)
            if dim <= 0:
                raise ValueError(f"All dimensions in {name} must be positive")
            normalized.append(dim)

        return normalized
    

    @staticmethod
    def _normalize_frequency(name,value):
        if value is None:
            return 0
        
        value = int(value)

        if value < 0:
            raise ValueError(f"{name} must be grater than or equal to zero.")
            
        return value


    # L'attuale implementazione del metodo submit fa l'override dell'originale: in particolare aggiunge un flag use_synergy che, se impostato a ver, invia la submit
    # al backend implementato per synergy, altrimenti utilizza super.submit, usando i metodi base di SyclQueue.
    def submit(
        self,
        kernel,
        args,
        gS,
        lS=None,
        dEvents=None,
        *,
        use_device_profiling=False,
        use_kernel_profiling=False,
        uncore_frequency=None,
        core_frequency=None,
    ):
        """
        Submit a dpctl.program.SyclKernel through the SYnergy backend.

        This is the direct-kernel execution path: the kernel must already be a
        valid dpctl.program.SyclKernel. This method does not create or compile
        kernels; it only submits an existing kernel through synergy::queue.

        Parameters
        ----------
        kernel:
            Existing dpctl.program.SyclKernel.
        args:
            Kernel arguments, in the same order expected by the kernel.
        gS:
            Global range. Required. Must have 1, 2 or 3 dimensions.
        lS:
            Optional local range. If provided, NDRange submit is used.
            If omitted, simple range submit is used.
        dEvents:
            Optional dependency events.
        use_device_profiling:
            If True, collect device energy before and after the kernel.
        use_kernel_profiling:
            If True, collect kernel energy from the returned event.
        uncore_frequency, core_frequency:
            Optional frequency values. If at least one is provided, the
            frequency-scaling overload of synergy::queue::submit is used.
        """

        if kernel is None:
            raise ValueError("kernel is required.")

        if not isinstance(kernel, dpctl_program.SyclKernel):  #cambio da if not hasattr - ci assicuriamo che l'oggetto che ci arriva sia effettivamente un SyclKernel e non una ref
            raise TypeError(
                "kernel must be a dpctl.program.SyclKernel. "
                "Use submit_spirv(...) or submit_opencl_source(...) if the kernel "
                "has to be created from an external representation."
                )

        if args is None:
            args = []

        if not isinstance(args, (list, tuple)):
            raise TypeError("args must be a list or tuple.")

        args = list(args)

        gS = self._normalize_range("gS", gS, required=True)
        lS = self._normalize_range("lS", lS, required=False)

        if lS is not None and len(lS) != len(gS):
            raise ValueError("lS and gS must have the same number of dimensions.")

        if dEvents is None:
            dEvents = []

        if not isinstance(dEvents, (list, tuple)):
            raise TypeError("dEvents must be a list or tuple of dpctl.SyclEvent objects.")

        dEvents = list(dEvents)

        use_frequency_scaling = (
            uncore_frequency is not None or core_frequency is not None
        )

        normalized_uncore_frequency = self._normalize_frequency(
            "uncore_frequency",
            uncore_frequency,
        )

        normalized_core_frequency = self._normalize_frequency(
            "core_frequency",
            core_frequency,
        )


        if self._execution_backend == "dpctl":
            if use_device_profiling or use_kernel_profiling:
                raise ValueError(
                    "SYnergy energy profiling is not available when "
                    "execution_backend='dpctl'."
                )

            if use_frequency_scaling:
                raise ValueError(
                    "SYnergy frequency scaling is not available when "
                    "execution_backend='dpctl'."
                )

            event = super().submit(
                kernel,
                args=args,
                gS=gS,
                lS=lS,
                dEvents=dEvents,
            )

            self._last_event = event
            self._profile_log.append(
                {
                    "execution_backend": "dpctl",
                    "use_device_profiling": False,
                    "use_kernel_profiling": False,
                    "use_frequency_scaling": False,
                }
            )

            return event

        bridge = self._load_submit_bridge()

        event, profile = bridge.submit(
            queue=self,
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
    

    def _load_spirv_il(self, spirv):
        """
        Load SPIR-V intermediate language 

        Parameters:
        -----------
        spirv:
            Can be either an object containing SPIR-V IL or a path to a .spv file

        Returns:
        --------
        bytes
            SPIR-V binary content
        """

        if isinstance(spirv, (bytes, bytearray, memoryview)):
            il = bytes(spirv)
        else:
            path = Path(spirv)

            if not path.exists():
                raise FileNotFoundError(f"SPIR-V file not found in path {path}")
            
            il = path.read_bytes()

        if not il:
            raise ValueError("SPIR-V IL is empty")
        
        return il
    
    def create_kernel_from_spirv(
            self,
            spirv,
            kernel_name,
            compile_options="",
    ):
        
        """
        This function creates a dpctl.program.SyclKernel form SPIR-V IL.

        Note: this metod ONLY CREATES the kernel, it does not submit it. The returned kernel can than be used as a parameter for the submit(...) method


        Parameters:
        ----------
        spirv:
            Path to a .spv file or SPIR-V IL
        kernel_name:
            Name of the kernel function contained in the spirv module
        compile_options:
            Optional compilation options passed to dpctl

        Returns:
        --------
        dpctl.program.SyclKernel

        """


        if not kernel_name or not isinstance(kernel_name, str):
            raise ValueError("kernel_name must be a non-empty string.")
        
        il = self._load_spirv_il(spirv)

        try:
            program = dpctl_program.create_program_from_spirv(
                self,
                il,
                copts = compile_options or "",
            )

            return program.get_sycl_kernel(kernel_name)
        
        except Exception as exc: #except improvement - check backend 
            backend = getattr(self.sycl_device, "backend", "unknown")
            device = getattr(self.sycl_device, "name", "unknown")

            raise RuntimeError(
                f"Unable to create kernel '{kernel_name}' from SPIR-V"
                f"on device '{device}' with backend '{backend}'."
                "The SPIR-V loading path is correct, but the current backend may not support runtime program from SPIR-V"
            ) from exc
        
    
    def create_kernel_from_opencl_source(
            self,
            source,
            kernel_name,
            compile_options="",
    ):
        """
        Create a dpctl.program.SycKerel from an OpenCL source


        Parameters:
        -----------
        source:
            OpenCL source code, as a string
        kernel_name:
            Name of the kernel function inside the source
        compile_options:
            Optional compilation options for dpctl

        Returns:
        --------
        dpctl.program.SyclKernel
        """

        if not isinstance(source, str) or not source.strip():
            raise ValueError("Source must be a non empty OpenCL C source string")

        if not isinstance(kernel_name, str):
            raise ValueError("kernel_name must be a non-empty string")
        
        try:
            program= dpctl_program.create_program_from_source(
                self,
                source,
                copts=compile_options or ""
            )

            return program.get_sycl_kernel(kernel_name)
        
        except Exception as exc:
            backend = getattr(self.sycl_device, "backend", "unknown")
            device = getattr(self.sycl_device, "name", "unknown")

            raise RuntimeError(
                f"Unable to create kernel '{kernel_name}' from OpenCL source "
                f"on device '{device}' with backend '{backend}'. "
                "The source path is correct, but the current backend may not "
                "support runtime program creation from OpenCL C source."
            ) from exc
        
    def submit_opencl_source(
            self, 
            source,
            kernel_name, 
            args,
            gS,
            lS=None,
            dEvents=None,
            *,
            use_device_profiling=False,
            use_kernel_profiling=False,
            uncore_frequency=None,
            core_frequency=None,
            compile_options="",
    ):
        kernel = self.create_kernel_from_opencl_source(
            source,
            kernel_name,
            compile_options=compile_options
        )

        return self.submit(
            kernel,
            args=args,
            gS=gS,
            lS=lS,
            dEvents=dEvents,
            use_device_profiling=use_device_profiling,
            use_kernel_profiling=use_kernel_profiling,
            uncore_frequency=uncore_frequency,
            core_frequency=core_frequency
        )
        
        
    def submit_spirv(
            self,
            spirv,
            kernel_name,
            args,
            gS,
            lS=None,
            dEvents=None,
            *,
            use_device_profiling=False,
            use_kernel_profiling=False,
            uncore_frequency=None,
            core_frequency=None,
            compile_options="",
    ):
        
        """
        Create a kernel from SPIR-V and submit it through SYnergy backend


        It is a wapper of 
            create_kernel_from_spirv(...)
            submit(...)
        """

        kernel = self.create_kernel_from_spirv(
            spirv,
            kernel_name,
            compile_options=compile_options
            )
        
        return self.submit(
            kernel,
            args=args,
            gS=gS, lS=lS,
            dEvents=dEvents,
            use_device_profiling=use_device_profiling,
            use_kernel_profiling=use_kernel_profiling,
            uncore_frequency=uncore_frequency,
            core_frequency=core_frequency,
            )
    
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
            ) from exc
    
    
    def wait(self):
        super().wait()
        if self._adapter is not None:
            self._adapter.wait()

    @property
    def synergy_device(self):
        if self._synergy_device is not None:
            return self._synergy_device

        try:
            from .synergy_device import SYnergyDevice
        except ImportError:
            from bindings.synergy_device import SYnergyDevice

        self._synergy_device = SYnergyDevice(self.sycl_device)
        return self._synergy_device
    
    @property
    def synergy_device_name(self):
        return self.synergy_device.name


    @property
    def synergy_backend_name(self):
        if self._adapter is not None:
            return self._adapter.backend_name()
        return str(self.sycl_device.backend)
    
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
        if self._adapter is not None:
            return self._adapter.capabilities().as_dict()

        return {
            "cuda_support": False,
            "rocm_support": False,
            "level_zero_support": False,
            "geopm_support": False,
            "device_profiling": False,
            "kernel_profiling": False,
            "host_profiling": False,
            "use_profiling_energy": False,
            "execution_backend": "dpctl",
        }

    def device_energy_consumption(self):
        if self._adapter is None:
            raise RuntimeError(
                "Device energy consumption is available only with "
                "execution_backend='synergy'."
            )
        return self._adapter.device_energy_consumption()


    def kernel_energy_consumption(self, event):
        if self._adapter is None:
            raise RuntimeError(
                "Kernel energy consumption is available only with "
                "execution_backend='synergy'."
            )
        return self._adapter.kernel_energy_consumption(event)
    

    @staticmethod
    def _unwrap_synergy_device_args(args):
        if not args:
            return args, None

        try:
            from .synergy_device import SYnergyDevice
        except ImportError:
            from bindings.synergy_device import SYnergyDevice

        first = args[0]

        if isinstance(first, SYnergyDevice):
            return (first.dpctl_device,) + args[1:], first

        return args, None
        