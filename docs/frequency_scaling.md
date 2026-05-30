# Frequency Scaling
This page describes how frequency scaling is exposed through the SYnergy Python binding.

Frequency scaling is handled at device level through `SYnergyDevice`. Kernel execution and profiling are handled by `SYnergyQueue`.

This separation is intentional:

```text
SYnergyDevice
    -> device information
    -> supported frequencies
    -> current frequencies
    -> frequency scaling

SYnergyQueue
    -> kernel submission
    -> queue synchronization
    -> device/kernel profiling
```

To utilize frequency scaling features you must use the `SYnergyDevice`

## Creating a SYnergyDevice

Frequency scaling operations require a `SYnergyDevice`.

Refer to [SYnergyDevice](quickstart.md) to understand how to create one.

One important information t must be exposed is that most devices cannot accept random frequencies. most of them have different accepted configuration. It is strongly advised to check the correct ones for your device before starting tinkering. 

On NVIDIA devices, the memory frequency is usually referred to as `uncore frequency` and corresponds to the memory clock frequency that is exposed by the backend. 

## Reading frequencies 
The SYnergy Python binding exposes different ways to read frequencies. 

The first considered is `device.current_core_frequency` (\_uncore\_ for memory). This method allows to obtain the current frequency the core or memory is at. It can be very useful when actually changing frequencies as a double check, to make sure that the selected frequencies have been set correctly.  

```python
current_core = device.current_core_frequency()


current_uncore = device.current_uncore_frequency()
```
Both methods accept the `cached` parameter, which is set to `True` by default.


## Setting frequencies 

The user can set the desired frequencies, as long as it is supported, via 

```python
device.set_core_frequency(desired_frequency)

device.set_uncore_frequency(desired_frequency)
```

Always make sure that the `desired_frequency`, especially for the core, is supported and in range with the desrerd uncore frequency selected. 

These two functions are completely independent between each other. 

If the user wants to change both, the function `device.set_frequencies(core=core_freq, uncore=uncore_freq)` can be used. 

A small example to better comprehend is shown:

```python

from bindings import SYnergyDevice

device = SYnergyDevice("cuda:gpu:0", require_synergy=True)

core_values = device.supported_core_frequencies()
uncore_values = device.supported_uncore_frequencies()

target_core = core_values[0]
target_uncore = uncore_values[0]

device.set_frequencies(
    core=target_core,
    uncore=target_uncore,
)

print("Current core frequency:", device.current_core_frequency(cached=False))
print("Current uncore frequency:", device.current_uncore_frequency(cached=False))
```

