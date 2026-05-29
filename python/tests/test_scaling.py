import dpctl
import time
import numpy as np
import dpctl.memory as dpm
import ctypes
import csv

from bindings import SYnergyDevice, SYnergyQueue
import bindings._synergy_submit as synergy_submit

"""
UNCORE_CLOCK = 1107

dev = SYnergyDevice("cuda:gpu:0")

core_freq = dev.current_core_frequency()
uncore_freq = dev.current_uncore_frequency(cached=False)

print("Current uncore cached:", core_freq , "MHz")
print("Current uncore direct:", uncore_freq, "MHz")

print()

print("======================")
print("Sezione Scaling")
print("======================")

#dev.set_core_frequency(1207)
print("Scaled core frequency", dev.current_core_frequency())
"""
N = 16 * 8192 * 8192
REPEAT = 4092
LOCAL_SIZE = 256


def vec_add(q: SYnergyQueue):
    kernel = synergy_submit.create_vecadd_kernel(q._adapter)

    print("Kernel:", kernel.get_function_name())

    a_host = np.full(N, 2.0, dtype=np.float32)
    b_host = np.full(N, 3.0, dtype=np.float32)
    c_host = np.zeros(N, dtype=np.float32)

    a_dev = dpm.MemoryUSMShared(a_host.nbytes, queue=q)
    b_dev = dpm.MemoryUSMShared(b_host.nbytes, queue=q)
    c_dev = dpm.MemoryUSMShared(c_host.nbytes, queue=q)

    q.memcpy(a_dev, a_host, a_host.nbytes)
    q.memcpy(b_dev, b_host, b_host.nbytes)
    q.wait()
    host_start = time.perf_counter()
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
    host_end = time.perf_counter()
    q.memcpy(c_host, c_dev, c_host.nbytes)
    q.wait()

    print("First values:", c_host[:10])
    print("Tempo impiegato: ", (host_end - host_start) * 1000)

    try:
        kernel_start = event.profiling_info_start
        kernel_end = event.profiling_info_end
        kernel_time_ms = (kernel_end - kernel_start) / 1_000_000.0
    except Exception as exc:
        kernel_time_ms = None
        print("Event profiling non disponibile:", repr(exc))

    print("Kernel time:", kernel_time_ms)
    print(q.last_profile)


if __name__ == "__main__":

    print("TEST BASE")
    dev = SYnergyDevice("cuda:gpu:0")
    print("Frequenza attuale: ", dev.current_core_frequency())
    q_auto = SYnergyQueue(dev)
    
    print("Frequenza attuale post coda: ", dev.current_core_frequency())
    

    vec_add(q_auto)
    print("============================================")
    
    print("TEST 1207")
    dev.set_core_frequency(1207)
    print("Frequenza attuale: ", dev.current_core_frequency())

    q_1 = SYnergyQueue(dev)

    vec_add(q_1)
    print("============================================")
    print("TEST 360")

    dev.set_core_frequency(360)
    print("Frequenza attuale: ", dev.current_core_frequency())

    q_2 = SYnergyQueue(dev)

    vec_add(q_2)
    print("============================================")

    
    


