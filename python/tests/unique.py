#!/usr/bin/env python3
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np
import dpctl.memory as dpm


from bindings import SYnergyQueue

import bindings._synergy_submit as synergy_submit



N = 4 * 1024 * 1024
REPEAT = 256
LOCAL_SIZE = 256
DEVICE_SELECTOR = "cuda:gpu:0"

def allocate_usm_vectors(q: SYnergyQueue, n: int):
    """
    Creates 3 vectors:
        a = [1, 1, 1, ...]
        b = [2, 2, 2, ...]
        c = [0, 0, 0, ...]

    Data allocated as USM Shared Memory so they can be sent as args to SYCL kernel
    """

    a_host = np.ones(n, dtype=np.float32)
    b_host = np.full(n, 2.0, dtype=np.float32)
    c_host = np.empty_like(a_host)

    a_dev = dpm.MemoryUSMShared(a_host.nbytes, queue=q)
    b_dev = dpm.MemoryUSMShared(b_host.nbytes, queue=q)
    c_dev = dpm.MemoryUSMShared(c_host.nbytes, queue=q)

    q.memcpy(a_dev, a_host, a_host.nbytes)
    q.memcpy(b_dev, b_host, b_host.nbytes)
    c_dev.memset(0)

    return a_dev, b_dev, c_dev, c_host


def check_result(q: SYnergyQueue, c_dev, c_host, expected_value: float = 3.0) -> bool:
    """
    Copy USM -> device and print the first 10 elements
    """

    q.memcpy(c_host, c_dev, c_host.nbytes)

    print("Values check:", c_host[:10])

    

def main() -> int:
    print("=== Creating SYnergyQueue ===")

    q = dpctl.SyclQueue(
        DEVICE_SELECTOR
    )

    print("Device:", q.synergy_device_name)
    print("Backend:", q.synergy_backend_name)
    print("Capabilities:", q.capabilities())
    print()

   
    if not hasattr(synergy_submit, "create_busy_kernel"):
        raise RuntimeError(
            "bindings._synergy_submit does not expose create_busy_kernel(). "
            "Make sure _synergy_submit has been compiled correctly."
        )
    
    #Creation of SyclKernel - needed for submit
    kernel = synergy_submit.create_busy_kernel(q._adapter)

    print("Kernel:", kernel)

    try:
        print("Kernel name:", kernel.get_function_name())
    except Exception as exc:
        print("Kernel name unavailable:", repr(exc))

    a_dev, b_dev, c_dev, c_host = allocate_usm_vectors(q, N)

    print()

    event = q.submit(
        kernel,
        args=[
            a_dev,
            b_dev,
            c_dev,
            ctypes.c_uint32(N),
            ctypes.c_uint32(REPEAT),
        ],
        gS=[N],
        lS=[LOCAL_SIZE],
        use_device_profiling=True,
        use_kernel_profiling=True,
    )

    event.wait()

    print("Event:", event)
    print("Last profile:", q.last_profile)
    print()

    print("=== Checking result ===")

    check_result(q, c_dev, c_host)
    
    print()
    print("=== Event profiling info ===")

    try:
        print("submit ns:", event.profiling_info_submit)
        print("start ns:", event.profiling_info_start)
        print("end ns:", event.profiling_info_end)
        print("duration ns:", event.profiling_info_end - event.profiling_info_start)
    except Exception as exc:
        print("Event profiling info unavailable:", repr(exc))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())