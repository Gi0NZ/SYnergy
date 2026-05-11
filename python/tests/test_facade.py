from bindings import SYnergyQueue
import bindings._synergy_submit as synergy_submit

import ctypes
import numpy as np
import dpctl.memory as dpm


def main():
    q = SYnergyQueue("cuda:gpu:0")

    print("=== Queue info ===")
    print("Device:", q.synergy_device_name)
    print("Backend:", q.synergy_backend_name)
    print("Capabilities:", q.capabilities())
    print("Native handle:", hex(q._adapter._native_handle()))
    print()

    print("=== Creating native SYCL vector-add kernel ===")
    kernel = synergy_submit.create_busy_kernel(q._adapter)

    print("Kernel:", kernel)

    try:
        print("Kernel name:", kernel.get_function_name())
    except Exception as exc:
        print("Kernel name non disponibile:", repr(exc))

    print()

    n = 4 * 1024 * 1024
    repeat = 256

    print("=== Allocating host data ===")
    a_host = np.ones(n, dtype=np.float32)
    b_host = np.full(n, 2.0, dtype=np.float32)
    c_host = np.empty_like(a_host)

    print("=== Allocating USM shared data ===")
    a_dev = dpm.MemoryUSMShared(a_host.nbytes, queue=q)
    b_dev = dpm.MemoryUSMShared(b_host.nbytes, queue=q)
    c_dev = dpm.MemoryUSMShared(c_host.nbytes, queue=q)

    print("=== Copy host -> USM ===")
    q.memcpy(a_dev, a_host, a_host.nbytes)
    q.memcpy(b_dev, b_host, b_host.nbytes)
    c_dev.memset(0)

    print("=== Submit tramite SYnergyQueue.submit ===")
    event = q.submit(
        kernel,
        args=[
            a_dev,
            b_dev,
            c_dev,
            ctypes.c_uint32(n),
            ctypes.c_uint32(repeat),
        ],
        gS=[n],
        lS=[256],
        use_device_profiling=True,
        use_kernel_profiling=True,
    )

    print("Event:", event)
    print("Last profile:", q.last_profile)
    print()

    print("=== Copy USM -> host ===")
    q.memcpy(c_host, c_dev, c_host.nbytes)

    print("=== Correctness check ===")
    expected = np.full(n, 3.0, dtype=np.float32)

    max_abs_err = float(np.max(np.abs(c_host - expected)))
    print("Max abs error:", max_abs_err)
    print("First values:", c_host[:10])

    if max_abs_err < 1e-4:
        print("Vector add OK")
    else:
        print("Vector add FAILED")

    print()
    print("=== Event timing ===")
    try:
        print("submit ns:", event.profiling_info_submit)
        print("start ns:", event.profiling_info_start)
        print("end ns:", event.profiling_info_end)
        print(
            "duration ns:",
            event.profiling_info_end - event.profiling_info_start,
        )
    except Exception as exc:
        print("Event profiling info unavailable:", repr(exc))


if __name__ == "__main__":
    main()