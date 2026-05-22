import dpctl
import importlib

_synergy_native = importlib.import_module("bindings._synergy_native")


class SYnergyDevice:
    def __init__(self, selector="cuda:gpu:0", require_synergy=False):
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
        return self._dpctl_device

    @property
    def is_synergy_supported(self):
        return self._adapter is not None

    @property
    def adapter_error(self):
        return self._adapter_error

    @property
    def name(self):
        if self._adapter is not None:
            return self._adapter.name()
        return self._dpctl_device.name

    @property
    def backend(self):
        if self._adapter is not None:
            return self._adapter.backend_name()

        try:
            return self._dpctl_device.get_filter_string().split(":")[0]
        except Exception:
            return str(self._dpctl_device.backend)

    @property
    def is_gpu(self):
        if self._adapter is not None:
            return self._adapter.is_gpu()
        return bool(self._dpctl_device.is_gpu)

    @property
    def is_cpu(self):
        if self._adapter is not None:
            return self._adapter.is_cpu()
        return bool(self._dpctl_device.is_cpu)

    def require_synergy_backend(self):
        if self._adapter is None:
            raise RuntimeError(
                f"Device {self.name!r} is a valid dpctl device, "
                "but it is not supported by the SYnergy native backend. "
                "Frequency scaling and SYnergy energy profiling are unavailable."
            )

    def supported_core_frequencies(self):
        self.require_synergy_backend()
        return self._adapter.supported_core_frequencies()

    def supported_uncore_frequencies(self):
        self.require_synergy_backend()
        return self._adapter.supported_uncore_frequencies()

    def current_core_frequency(self, cached=True):
        self.require_synergy_backend()
        return self._adapter.current_core_frequency(cached)

    def current_uncore_frequency(self, cached=True):
        self.require_synergy_backend()
        return self._adapter.current_uncore_frequency(cached)

    def set_core_frequency(self, freq):
        self.require_synergy_backend()
        return self._adapter.set_core_frequency(int(freq))

    def set_uncore_frequency(self, freq):
        self.require_synergy_backend()
        return self._adapter.set_uncore_frequency(int(freq))

    def set_frequencies(self, core, uncore):
        self.require_synergy_backend()
        return self._adapter.set_frequencies(int(core), int(uncore))

    def __repr__(self):
        return (
            f"SYnergyDevice("
            f"name={self.name!r}, "
            f"backend={self.backend!r}, "
            f"is_gpu={self.is_gpu}, "
            f"is_cpu={self.is_cpu}, "
            f"synergy_supported={self.is_synergy_supported})"
        )