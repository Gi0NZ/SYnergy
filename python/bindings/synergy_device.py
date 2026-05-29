import dpctl
import importlib

_synergy_native = importlib.import_module("bindings._synergy_native")


class SYnergyDevice:
    """
    High-level representation of a SYnergy-compatible SYCL device.

    ``SYnergyDevice`` wraps a ``dpctl.SyclDevice`` and, when available,
    connects it to the native SYnergy backend. The class is responsible for
    device-level operations such as querying supported frequencies, reading
    current frequencies, and applying frequency scaling policies.

    This class intentionally separates frequency management from kernel
    execution. Device frequencies are configured through ``SYnergyDevice``,
    while kernel submission and profiling are handled by ``SYnergyQueue``.

    Parameters
    ----------
    selector : str or dpctl.SyclDevice, optional
        SYCL device selector or existing ``dpctl.SyclDevice`` instance.
        The default value is ``"cuda:gpu:0"``.
    require_synergy : bool, optional
        If True, raise an exception when the selected device is visible to
        dpctl but not supported by the SYnergy native backend. If False, the
        object is still created and can expose dpctl-level information, but
        SYnergy-specific operations are unavailable.

    Attributes
    ----------
    selector : str or None
        Original selector string used to create the device. It is set to
        None when an existing ``dpctl.SyclDevice`` is passed.
    is_synergy_supported : bool
        True if the native SYnergy backend successfully initialized for this
        device.
    adapter_error : RuntimeError or None
        Error raised while initializing the SYnergy native adapter, if any.

    Examples
    --------
    Create a SYnergy device and inspect its supported frequencies:

    ```python
    dev = SYnergyDevice("cuda:gpu:0")

    if dev.is_synergy_supported:
        print(dev.supported_core_frequencies())
        print(dev.current_core_frequency(cached=False))
    ```

    Set a GPU core frequency before creating or using a SYnergy queue:

    ```python
    dev = SYnergyDevice("cuda:gpu:0")
    dev.set_core_frequency(1207)
    ```
    """

    def __init__(self, selector="cuda:gpu:0", require_synergy=False):
        """
        Initialize a SYnergy device wrapper.

        The constructor first creates or stores a ``dpctl.SyclDevice`` and then
        attempts to initialize the native SYnergy device adapter. If the native
        backend is not available, the object can still expose basic dpctl
        information unless ``require_synergy`` is set to True.

        Parameters
        ----------
        selector : str or dpctl.SyclDevice, optional
            SYCL device selector string, for example ``"cuda:gpu:0"``, or an
            already created ``dpctl.SyclDevice`` instance.
        require_synergy : bool, optional
            If True, fail immediately when the SYnergy native backend does not
            support the selected device.

        Raises
        ------
        RuntimeError
            If ``require_synergy`` is True and the selected device cannot be
            initialized by the SYnergy native backend.
        """
        self.selector = selector
        self._adapter = None
        self._adapter_error = None

        if isinstance(selector, dpctl.SyclDevice):
            self._dpctl_device = selector
            self.selector = None
        else:
            self._dpctl_device = dpctl.SyclDevice(selector)

        try:
            self._adapter = _synergy_native.SYnergy_Device_Adapter(
                self._dpctl_device
            )
        except RuntimeError as exc:
            self._adapter_error = exc

            if require_synergy:
                raise RuntimeError(
                    f"Device {self.name!r} is visible to dpctl, "
                    "but is not supported by the SYnergy native backend."
                ) from exc

    @property
    def dpctl_device(self):
        """
        Return the underlying ``dpctl.SyclDevice``.

        Returns
        -------
        dpctl.SyclDevice
            Device object managed by dpctl.
        """
        return self._dpctl_device

    @property
    def is_synergy_supported(self):
        """
        Check whether the device is supported by the SYnergy backend.

        Returns
        -------
        bool
            True if the native SYnergy adapter was successfully initialized,
            False otherwise.
        """
        return self._adapter is not None

    @property
    def adapter_error(self):
        """
        Return the error raised during SYnergy adapter initialization.

        Returns
        -------
        RuntimeError or None
            Initialization error raised by the native SYnergy adapter, or None
            if initialization succeeded.
        """
        return self._adapter_error

    @property
    def name(self):
        """
        Return the device name.

        If the SYnergy native adapter is available, the name is retrieved from
        the SYnergy backend. Otherwise, the name exposed by dpctl is returned.

        Returns
        -------
        str
            Human-readable device name.
        """
        if self._adapter is not None:
            return self._adapter.name()
        return self._dpctl_device.name

    @property
    def backend(self):
        """
        Return the backend associated with the selected device.

        When the SYnergy adapter is available, the backend name is obtained
        from the native backend. Otherwise, the method falls back to dpctl
        information.

        Returns
        -------
        str
            Backend name, for example ``"cuda"``, ``"level_zero"``,
            ``"opencl"``, or another dpctl-supported backend.
        """
        if self._adapter is not None:
            return self._adapter.backend_name()

        try:
            return self._dpctl_device.get_filter_string().split(":")[0]
        except Exception:
            return str(self._dpctl_device.backend)

    @property
    def is_gpu(self):
        """
        Check whether the selected device is a GPU.

        Returns
        -------
        bool
            True if the selected device is a GPU, False otherwise.
        """
        if self._adapter is not None:
            return self._adapter.is_gpu()
        return bool(self._dpctl_device.is_gpu)

    @property
    def is_cpu(self):
        """
        Check whether the selected device is a CPU.

        Returns
        -------
        bool
            True if the selected device is a CPU, False otherwise.
        """
        if self._adapter is not None:
            return self._adapter.is_cpu()
        return bool(self._dpctl_device.is_cpu)

    def require_synergy_backend(self):
        """
        Ensure that the selected device supports the SYnergy native backend.

        This method is used internally before executing SYnergy-specific
        operations such as querying supported frequencies or applying frequency
        scaling.

        Raises
        ------
        RuntimeError
            If the device is visible to dpctl but not supported by the SYnergy
            native backend.
        """
        if self._adapter is None:
            raise RuntimeError(
                f"Device {self.name!r} is a valid dpctl device, "
                "but it is not supported by the SYnergy native backend. "
                "Frequency scaling and SYnergy energy profiling are unavailable."
            )

    def supported_core_frequencies(self):
        """
        Return the supported GPU core frequencies.

        The returned values represent the core frequency levels supported by
        the selected device and exposed by the SYnergy native backend.

        Returns
        -------
        list[int]
            Supported GPU core frequencies in MHz.

        Raises
        ------
        RuntimeError
            If the device is not supported by the SYnergy native backend.
        """
        self.require_synergy_backend()
        return self._adapter.supported_core_frequencies()

    def supported_uncore_frequencies(self):
        """
        Return the supported GPU uncore frequencies.

        On NVIDIA devices, the uncore frequency usually corresponds to the
        memory clock frequency exposed by the backend.

        Returns
        -------
        list[int]
            Supported GPU uncore frequencies in MHz.

        Raises
        ------
        RuntimeError
            If the device is not supported by the SYnergy native backend.
        """
        self.require_synergy_backend()
        return self._adapter.supported_uncore_frequencies()

    def current_core_frequency(self, cached=True):
        """
        Return the current GPU core frequency.

        Parameters
        ----------
        cached : bool, optional
            If True, return the cached value stored by the native adapter.
            If False, query the backend directly for the current device value.

        Returns
        -------
        int
            Current GPU core frequency in MHz.

        Raises
        ------
        RuntimeError
            If the device is not supported by the SYnergy native backend.
        """
        self.require_synergy_backend()
        return self._adapter.current_core_frequency(cached)

    def current_uncore_frequency(self, cached=True):
        """
        Return the current GPU uncore frequency.

        On NVIDIA devices, this value usually maps to the memory clock
        frequency.

        Parameters
        ----------
        cached : bool, optional
            If True, return the cached value stored by the native adapter.
            If False, query the backend directly for the current device value.

        Returns
        -------
        int
            Current GPU uncore frequency in MHz.

        Raises
        ------
        RuntimeError
            If the device is not supported by the SYnergy native backend.
        """
        self.require_synergy_backend()
        return self._adapter.current_uncore_frequency(cached)

    def set_core_frequency(self, freq):
        """
        Set the GPU core frequency.

        The requested frequency should belong to the list returned by
        ``supported_core_frequencies``.

        Parameters
        ----------
        freq : int
            Target GPU core frequency in MHz.

        Returns
        -------
        object
            Return value produced by the native SYnergy adapter.

        Raises
        ------
        RuntimeError
            If the device is not supported by the SYnergy native backend.
        ValueError
            If the native backend rejects the requested frequency.
        """
        self.require_synergy_backend()
        return self._adapter.set_core_frequency(int(freq))

    def set_uncore_frequency(self, freq):
        """
        Set the GPU uncore frequency.

        On NVIDIA devices, the uncore frequency usually corresponds to the
        memory clock frequency. The requested value should belong to the list
        returned by ``supported_uncore_frequencies``.

        Parameters
        ----------
        freq : int
            Target GPU uncore frequency in MHz.

        Returns
        -------
        object
            Return value produced by the native SYnergy adapter.

        Raises
        ------
        RuntimeError
            If the device is not supported by the SYnergy native backend.
        ValueError
            If the native backend rejects the requested frequency.
        """
        self.require_synergy_backend()
        return self._adapter.set_uncore_frequency(int(freq))

    def set_frequencies(self, core, uncore):
        """
        Set both GPU core and uncore frequencies.

        This method is a convenience wrapper around the native backend function
        that applies both frequency values together.

        Parameters
        ----------
        core : int
            Target GPU core frequency in MHz.
        uncore : int
            Target GPU uncore frequency in MHz.

        Returns
        -------
        object
            Return value produced by the native SYnergy adapter.

        Raises
        ------
        RuntimeError
            If the device is not supported by the SYnergy native backend.
        ValueError
            If one of the requested frequencies is rejected by the native
            backend.
        """
        self.require_synergy_backend()
        return self._adapter.set_frequencies(int(core), int(uncore))

    def __repr__(self):
        """
        Return a compact string representation of the device.

        Returns
        -------
        str
            String containing device name, backend, device type, and SYnergy
            support status.
        """
        return (
            f"SYnergyDevice("
            f"name={self.name!r}, "
            f"backend={self.backend!r}, "
            f"is_gpu={self.is_gpu}, "
            f"is_cpu={self.is_cpu}, "
            f"synergy_supported={self.is_synergy_supported})"
        )