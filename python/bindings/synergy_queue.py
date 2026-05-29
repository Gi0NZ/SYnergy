"""
SYnergy-aware queue abstraction built on top of dpctl.

This module defines :class:`SYnergyQueue`, a Python facade over
``dpctl.SyclQueue`` that can submit existing ``dpctl.program.SyclKernel``
objects through the native SYnergy backend.

The queue is responsible for kernel execution and optional energy profiling.
Device-level frequency management is intentionally handled by
``SYnergyDevice`` instead, so frequency scaling is configured before kernel
submission and not passed as a submit argument.
"""
import dpctl
import dpctl.program as dpctl_program
import importlib
from pathlib import Path

_synergy_native = importlib.import_module("bindings._synergy_native")

class SYnergyQueue(dpctl.SyclQueue):
    """
    SYnergy-aware queue built on top of ``dpctl.SyclQueue``.

    ``SYnergyQueue`` extends the standard dpctl queue interface with optional
    integration with the native SYnergy backend. When the SYnergy backend is
    available, kernels can be submitted through a native ``synergy::queue`` and
    energy profiling information can be collected after each submission.

    The class also supports a fallback mode based on the standard dpctl
    execution path. In this mode, kernel execution still works through
    ``dpctl.SyclQueue.submit``, but SYnergy-specific profiling features are not
    available.

    Frequency scaling is intentionally not handled by this class. Device
    frequencies should be queried and configured through ``SYnergyDevice``.

    Parameters
    ----------
    *args
        Positional arguments forwarded to ``dpctl.SyclQueue``. The first
        argument may also be a ``SYnergyDevice`` instance; in that case, its
        underlying ``dpctl.SyclDevice`` is used to construct the queue.
    property : str, tuple[str], list[str], or None, optional
        Queue properties passed to ``dpctl.SyclQueue``. The properties
        ``"in_order"`` and ``"enable_profiling"`` are added automatically
        because they are required by the SYnergy profiling workflow.
    execution_backend : {"synergy", "dpctl", "auto"}, optional
        Backend used for kernel submission. ``"synergy"`` requires the native
        SYnergy backend, ``"dpctl"`` forces standard dpctl execution, and
        ``"auto"`` tries SYnergy first and falls back to dpctl if needed.
    allow_fallback : bool, optional
        If True, fall back to the dpctl backend when the SYnergy native adapter
        cannot be initialized.
    **kwargs
        Additional keyword arguments forwarded to ``dpctl.SyclQueue``.

    Attributes
    ----------
    last_event : dpctl.SyclEvent or None
        Last event returned by a kernel submission.
    last_profile : dict or None
        Profiling information collected for the most recent submission.
    profile_log : list[dict]
        Copy of the profiling history collected by this queue.
    """
    def __new__(
    cls,
    *args,
    property=None,
    execution_backend="synergy",
    allow_fallback=False,
    **kwargs,
    ):
        """
        Create the underlying ``dpctl.SyclQueue`` instance.

        This method ensures that profiling-related queue properties are present
        before delegating object creation to ``dpctl.SyclQueue``. If the first
        positional argument is a ``SYnergyDevice``, it is replaced with its
        underlying ``dpctl.SyclDevice`` before queue construction.
        """
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
        """
        Initialize the SYnergy queue facade.

        The constructor stores the requested execution backend and attempts to
        create the native ``SYnergy_Queue_Adapter`` when the SYnergy backend is
        requested. If initialization fails and fallback is allowed, the queue
        remains usable through the standard dpctl backend.

        Raises
        ------
        ValueError
            If ``execution_backend`` is not one of ``"synergy"``, ``"dpctl"``
            or ``"auto"``.
        Exception
            Re-raises the native adapter initialization error when SYnergy is
            required and fallback is not allowed.
        """
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
        Ensure that queue properties required by SYnergy are present.

        SYnergy profiling requires the queue to be created with profiling
        enabled. The queue is also made in-order to simplify profiling and
        dependency handling.

        Parameters
        ----------
        prop : str, tuple[str], list[str], or None
            User-provided queue properties.

        Returns
        -------
        tuple[str, ...]
            Normalized queue properties including ``"in_order"`` and
            ``"enable_profiling"``.
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
        """
        Validate and normalize a SYCL execution range.

        Parameters
        ----------
        name : str
            Name of the range argument, used in error messages.
        value : list[int], tuple[int, ...], or None
            Range value to validate. Valid ranges must have 1, 2, or 3 positive
            dimensions.
        required : bool, optional
            If True, ``None`` is rejected. If False, ``None`` is accepted and
            returned unchanged.

        Returns
        -------
        list[int] or None
            Normalized range as a list of positive integers, or None when the
            range is optional and not provided.

        Raises
        ------
        ValueError
            If the range is required but missing, has an invalid number of
            dimensions, or contains non-positive values.
        TypeError
            If the range is not a list or tuple.
        """
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
    ):
        """
        Submit an existing ``dpctl.program.SyclKernel`` for execution.

        This method is the main kernel execution path exposed by
        ``SYnergyQueue``. The kernel must already exist as a
        ``dpctl.program.SyclKernel``; this method does not compile kernels from
        source code or SPIR-V. For those workflows, use
        :meth:`create_kernel_from_opencl_source`,
        :meth:`create_kernel_from_spirv`, :meth:`submit_opencl_source`, or
        :meth:`submit_spirv`.

        When the execution backend is ``"synergy"``, the kernel is submitted
        through the native SYnergy submit bridge and optional energy profiling
        can be enabled. When the execution backend is ``"dpctl"``, the method
        delegates to ``dpctl.SyclQueue.submit`` and SYnergy profiling is not
        available.

        Frequency scaling is not controlled by this method. Device frequencies
        should be configured through ``SYnergyDevice`` before submitting
        kernels.

        Parameters
        ----------
        kernel : dpctl.program.SyclKernel
            Existing SYCL kernel to submit.
        args : list or tuple
            Kernel arguments, in the same order expected by the kernel.
        gS : list[int] or tuple[int, ...]
            Global execution range. It must contain 1, 2, or 3 positive
            dimensions.
        lS : list[int], tuple[int, ...], or None, optional
            Local execution range. If provided, an NDRange submit is used.
            If None, a simple range submit is used.
        dEvents : list or tuple or None, optional
            Dependency events that must complete before this kernel starts.
        use_device_profiling : bool, optional
            If True, collect device-level energy information before and after
            kernel execution. Available only with the SYnergy backend.
        use_kernel_profiling : bool, optional
            If True, collect kernel-level energy information from the returned
            event. Available only with the SYnergy backend.

        Returns
        -------
        dpctl.SyclEvent
            Event associated with the submitted kernel.

        Raises
        ------
        ValueError
            If the kernel is missing, if the execution ranges are invalid, or
            if SYnergy profiling is requested while using the dpctl backend.
        TypeError
            If ``kernel`` is not a ``dpctl.program.SyclKernel`` or if ``args``
            and ``dEvents`` are not list-like objects.
        RuntimeError
            If the native SYnergy submit bridge cannot be loaded or fails
            during submission.
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



        if self._execution_backend == "dpctl":
            if use_device_profiling or use_kernel_profiling:
                raise ValueError(
                    "SYnergy energy profiling is not available when "
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
        Create a ``dpctl.program.SyclKernel`` from SPIR-V IL.

        This method only creates the kernel object. It does not submit the
        kernel for execution. The returned kernel can be passed to
        :meth:`submit`.

        Parameters
        ----------
        spirv : str, pathlib.Path, bytes, bytearray, or memoryview
            Path to a ``.spv`` file or in-memory SPIR-V binary content.
        kernel_name : str
            Name of the kernel function contained in the SPIR-V module.
        compile_options : str, optional
            Compilation options forwarded to dpctl.

        Returns
        -------
        dpctl.program.SyclKernel
            Kernel object created from the SPIR-V module.

        Raises
        ------
        ValueError
            If ``kernel_name`` is empty or if the SPIR-V content is empty.
        FileNotFoundError
            If ``spirv`` is interpreted as a path and the file does not exist.
        RuntimeError
            If dpctl cannot create the program for the selected backend or
            device.
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
        Create a ``dpctl.program.SyclKernel`` from OpenCL C source code.

        This method compiles OpenCL C source code through dpctl and extracts
        the requested kernel. It only creates the kernel object; execution is
        performed separately through :meth:`submit`.

        Parameters
        ----------
        source : str
            OpenCL C source code.
        kernel_name : str
            Name of the kernel function inside the source.
        compile_options : str, optional
            Compilation options forwarded to dpctl.

        Returns
        -------
        dpctl.program.SyclKernel
            Kernel object created from the OpenCL C source.

        Raises
        ------
        ValueError
            If ``source`` or ``kernel_name`` is empty.
        RuntimeError
            If dpctl cannot compile the source for the selected backend or
            device.
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
            compile_options="",
    ):
        """
        Compile an OpenCL C kernel and submit it for execution.

        This is a convenience method that combines
        :meth:`create_kernel_from_opencl_source` and :meth:`submit`.

        Parameters
        ----------
        source : str
            OpenCL C source code.
        kernel_name : str
            Name of the kernel function to execute.
        args : list or tuple
            Kernel arguments.
        gS : list[int] or tuple[int, ...]
            Global execution range.
        lS : list[int], tuple[int, ...], or None, optional
            Local execution range.
        dEvents : list or tuple or None, optional
            Dependency events.
        use_device_profiling : bool, optional
            Enable device-level energy profiling when using the SYnergy
            backend.
        use_kernel_profiling : bool, optional
            Enable kernel-level energy profiling when using the SYnergy
            backend.
        compile_options : str, optional
            Compilation options forwarded to dpctl.

        Returns
        -------
        dpctl.SyclEvent
            Event associated with the submitted kernel.
        """
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
            compile_options="",
    ):
        
        """
        Create a kernel from SPIR-V IL and submit it for execution.

        This is a convenience method that combines
        :meth:`create_kernel_from_spirv` and :meth:`submit`.

        Parameters
        ----------
        spirv : str, pathlib.Path, bytes, bytearray, or memoryview
            Path to a ``.spv`` file or in-memory SPIR-V binary content.
        kernel_name : str
            Name of the kernel function contained in the SPIR-V module.
        args : list or tuple
            Kernel arguments.
        gS : list[int] or tuple[int, ...]
            Global execution range.
        lS : list[int], tuple[int, ...], or None, optional
            Local execution range.
        dEvents : list or tuple or None, optional
            Dependency events.
        use_device_profiling : bool, optional
            Enable device-level energy profiling when using the SYnergy
            backend.
        use_kernel_profiling : bool, optional
            Enable kernel-level energy profiling when using the SYnergy
            backend.
        compile_options : str, optional
            Compilation options forwarded to dpctl.

        Returns
        -------
        dpctl.SyclEvent
            Event associated with the submitted kernel.
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
            )
    
    def _load_submit_bridge(self):
        """
        Load the Cython bridge used for native SYnergy submission.

        Returns
        -------
        module
            Imported ``bindings._synergy_submit`` module.

        Raises
        ------
        RuntimeError
            If the Cython submit bridge is not available.
        """
        try:
            return importlib.import_module("bindings._synergy_submit")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Il bridge bindings._synergy_submit non è ancora disponibile. "
            ) from exc
    
    
    def wait(self):
        """
        Wait for all queued work to complete.

        This method waits on the underlying ``dpctl.SyclQueue`` and, when the
        SYnergy adapter is available, also waits on the native SYnergy queue.
        """
        super().wait()
        if self._adapter is not None:
            self._adapter.wait()

    @property
    def synergy_device(self):
        """
        Return the ``SYnergyDevice`` associated with this queue.

        Returns
        -------
        SYnergyDevice
            Device wrapper associated with the queue.
        """
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
        """
        Return the name of the SYnergy device associated with this queue.

        Returns
        -------
        str
            Human-readable device name.
        """
        return self.synergy_device.name


    @property
    def synergy_backend_name(self):
        """
        Return the backend name used by this queue.

        Returns
        -------
        str
            Backend name reported by the SYnergy adapter or by dpctl.
        """
        if self._adapter is not None:
            return self._adapter.backend_name()
        return str(self.sycl_device.backend)
    
    @property
    def last_event(self):
        """
        Return the last event generated by this queue.

        Returns
        -------
        dpctl.SyclEvent or None
            Last submitted event, or None if no kernel has been submitted.
        """
        return self._last_event

    @property
    def last_profile(self):
        """
        Return the most recent profiling record.

        Returns
        -------
        dict or None
            Last profiling dictionary, or None if no submission has been
            performed.
        """
        return self._profile_log[-1] if self._profile_log else None

    @property
    def profile_log(self):
        """
        Return a copy of the profiling history.

        Returns
        -------
        list[dict]
            Profiling records collected after kernel submissions.
        """
        return list(self._profile_log)

    def capabilities(self):
        """
        Return the capabilities exposed by the selected execution backend.

        Returns
        -------
        dict
            Dictionary describing backend support and profiling capabilities.
        """
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
        """
        Return the current device energy consumption reported by SYnergy.

        Returns
        -------
        float
            Device energy consumption value reported by the native backend.

        Raises
        ------
        RuntimeError
            If the queue is not using the SYnergy backend.
        """
        if self._adapter is None:
            raise RuntimeError(
                "Device energy consumption is available only with "
                "execution_backend='synergy'."
            )
        return self._adapter.device_energy_consumption()


    def kernel_energy_consumption(self, event):
        """
        Return the energy consumption associated with a kernel event.

        Parameters
        ----------
        event : dpctl.SyclEvent
            Event associated with a submitted kernel.

        Returns
        -------
        float
            Kernel energy consumption value reported by the native backend.

        Raises
        ------
        RuntimeError
            If the queue is not using the SYnergy backend.
        """
        if self._adapter is None:
            raise RuntimeError(
                "Kernel energy consumption is available only with "
                "execution_backend='synergy'."
            )
        return self._adapter.kernel_energy_consumption(event)
    

    @staticmethod
    def _unwrap_synergy_device_args(args):
        """
        Replace a leading ``SYnergyDevice`` argument with its dpctl device.

        Parameters
        ----------
        args : tuple
            Positional arguments passed to ``SYnergyQueue``.

        Returns
        -------
        tuple
            Pair ``(normalized_args, synergy_device)`` where
            ``normalized_args`` can be forwarded to ``dpctl.SyclQueue`` and
            ``synergy_device`` is the original ``SYnergyDevice`` instance, if
            one was provided.
        """
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
        