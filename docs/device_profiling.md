# Device and kernel profiling

This page describes how device-level and kernel-level profiling are exposed through the SYnergy Python binding.

Profiling is one of the main reasons for using `SYnergyQueue` instead of the plain `dpctl.SyclQueue`.
When the native SYnergy backend is available, kernel submission can collect energy-related information from the selected device and from the submitted kernel.

At a high level, profiling is controlled through two flags:

```python
use_device_profiling=True
use_kernel_profiling=True
```
These flags can be passed to `queue.submit()` and to the helper submission methods, such as `submit_opencl_source()` and `submit_spirv()`.

## Profiling levels
The Python binding exposes two profiling levels:

- **device profiling**, which measures device-level energy information before and after kernel execution
- **kernel profiling**, which retrieves energy information associated with the submitted kernel event. 

These two levels are independent from each other. 

## Device Profiling
Device profiling collects energy information at the device level. 

When the flag `use_device_profiling=True`, the native bridge reads the device energy value before and after the kernel execution. The difference between the two values is then stored in the profiling result and can be accessed via

```python
profile = queue.last_profile
print(profile)
```

## Kernel Profiling

Kernel profiling collects energy information related to the submitted kernel event.

When `use_kernel_profiling=True` the profile dictionary contains `kernel_energy`.

Kernel profiling is dependent on the backend support. A device may be visible to `dpctl` but kernel-level energy profiling may still be unavailable. 

## Profiling capabilities

Before relying on profiling information, it can be useful to inspect the capabilities exposed by the queue:

```python
capabilities = queue.capabilities()

for key, value in capabilities.items():
    print(f"{key}: {value}")
```

If the queue is using the native SYnergy backend, the capabilities are retrieved from the native adapter.

If the queue is using the standard dpctl backend, SYnergy-specific profiling capabilities are not available.