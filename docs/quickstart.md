# Quickstart

This page is meant to give a quick indication on how to use the components of the SYnergy Python bindings. 

The goal of the Python layer is to provide a small facade over the native SYnergy/SYCL backend while keeping the programming model close to `dpctl`.

At a high level, the user interacts with two main Python objects:

- `SYnergyDevice`, used to inspect the selected SYCL device;
- `SYnergyQueue`, used to submit work and access SYnergy-specific profiling or frequency-scaling features.

The native implementation is still provided by SYnergy and SYCL. The Python layer only exposes a simplified interface for experimentation and integration with the custom `dpctl` build.

## Importing the package

After following the [installation steps](installation.md), the Python bindings can be imported via

```python
from bindings import SYnergyDevice, SYnergyQueue
```

In case the import fails, make sure that the SYnergy/python directory has been added to the PYTHONPATH.

## Creating a SYnergyDevice and SYnergyQueue
### SYnergyDevice
A `SYnergyDevice` wraps a `dpctl.SyclDevice`. It can be created via a SYCL device selector string:

```python
from bindings import SYnergyDevice

device = SYnergyDevice("cuda:gpu")
```

The selected device can be inspected via high-level properties, exposed by the facade:

```python
print("Device name:", device.name)
print("Backend:", device.backend)
print("Is GPU:", device.is_gpu)
print("Is CPU:", device.is_cpu)
print("SYnergy supported:", device.is_synergy_supported)
```

For more information refer to [API](api.md)

### SYnergyQueue

A `SYnergyQueue` is the main object used to submit kernels. It emulates the `dpctl.SyclQueue` behaviour while communicating with the SYnergy implementation. 

A `SYnergyQueue` can be created in 2 main ways: 
```python
from bindings import SYnergyQueue

queue = SYnergyQueue("cuda:gpu:0")
```

```python
from bindings import SYnergyDevice, SYnergyQueue

device = SYnergyDevice("cuda:gpu:0")
queue = SYnergyQueue(device)
```

Note that if you need to execute frequency scaling operations, you need to use the 2 mode, since the frequency manipulation elements are contained within the SYnergyDevice. 

Internally, the `SYnergyQueue` extends the `dpctl.SyclQueue`, meaning that it keeps all of its standard behaviour, provided by dpctl, while also adding the native SYnergy backend for energy related tasks. 

By default, when defining a `SYnergyQueue` without specifying an `execution_backend`, it attempts to use the SYnergy execution backend. If it fails it falls back to `dpctl`. This behaviour can be replicated by specifying `execution_backend="auto"` when creating a SYnergyQueue. 

One can also specify the precise backend wanted; 2 are the possible options:

1. `execution_backend="synergy"`
2. `execution_backend="dpctl"`

A `SYnergyQueue` offers a variety of different interactions. To explore them refer to [API](api.md)

Among all, one of the most important is `queue.wait()` methods, which allows one to wait for all queued work to complete. 

When the SYnergy adapter is available, this method waits both on the underlying `dpctl.SyclQueue` and on the native SYnergy queue. 

A minimal example of queue and device use is shown:

```python
    from bindings import SYnergyDevice, SYnergyQueue

    device = SYnergyDevice("cuda:gpu:0")
    queue = SYnergyQueue(device, execution_backend="auto")

    print("Device name:", device.name)
    print("Device backend:", device.backend)
    print("SYnergy supported:", device.is_synergy_supported)

    print("Queue device:", queue.synergy_device_name)
    print("Queue backend:", queue.synergy_backend_name)

    print("Capabilities:")
    for key, value in queue.capabilities().items():
        print(f"  {key}: {value}")

    queue.wait()
```

## Submitting a kernel
`SYnergyQueue` can submit kernels through the native SYnergy backend or through the standard `dpctl` execution path.

The main submission method is:

```python
queue.submit(
    kernel,
    args,
    gS,
    lS=None,
    dEvents=None,
    use_device_profiling=False,
    use_kernel_profiling=False,
)
```

The `kernel` argument must be an existing `dpctl.program.SyclKernel`

The `args` argument contains the kernel arguments in the same order expected by the defined kernel

The `gS` argument represents the global execution range: it can contain one, two or three dimensions

The optional `lS` argument represents the local execution range. When provided, an _NDRange_ submission is used. 

The flags `use_device_profiling` and `use_kernel_profiling` enable SYnergy profiling information when the queue is using the native SYnergy backend. 

As stated above, if you want to use directly the `.submit(...)` method, you need a `dpctl.program.SyclKernel`. 

The way of obtaining such object varies based on the type of kernel:

- **OpenCL Kernel**: if you have an OpenCL kernel, you can obtain a `dpctl.program.SyclKernel` by invoking the `create_program_from_source()` dpctl method. 
- **SPIR-V Kernel**: if you have an SPIR-V kernel, you can obtain a `dpctl.program.SyclKernel` by invoking the `create_program_from_spirv()` dpctl method.
- **Direct Kernel**: if you have a direct kernel, consult [Direct Kernel](direct_kernel.md) to understand the creation process. 

For OpenCL and SPIR-V kernel submission, there also are available two other functions:

- `submit_opencl_source()`: the method creates the kernel internally; the user need to indicate _source_ and _kernel\_name_ 

- `submit_spirv()`: the method creates the kernel internally; the user need to indicate the path to a `.spv` file and _kernel\_name_ 

To understand how to structure kernel submission, refer to the `SYnergy/python/tests` section for coded examples. 

## Waiting for completion

Once a kernel has been submitted, the returned event can be used to wait for that specific operation:

```python
event.wait()
```

If the user wants to wait for all the operations submitted to the queue, the `wait()` method can be used:

```python
queue.wait()
```

In general, event.wait() is useful when the application needs to synchronize only with a specific submitted kernel, while queue.wait() is useful when all queued operations must be completed before continuing.
