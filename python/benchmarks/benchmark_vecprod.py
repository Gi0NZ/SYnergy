import numpy as np
import dpctl
import dpctl.memory as dpm
import ctypes
import time
import csv
import gc


import bindings._synergy_submit as synergy_submit

from bindings import SYnergyQueue
from pathlib import Path


SIZES = [
    2**10,
    2**12,
    2**14,
    2**16,
    2**18,
    2**20,
    2**22,
    2**24,
    2**26,
    2**28,
    2**30
]
REPEAT = 16384
LOCAL_SIZE = 256
DEVICE_SELECTOR = "cuda:gpu:0"

WARMUP_RUNS = 3
MEASURED_RUNS = 10

OUTPUT_CSV = Path("results/python_vecprod_results.csv")



def allocate_usm_vectors(q: SYnergyQueue, n: int):
    """
    Creates and allocates 3 vectors
    
    Data is allocated as USM Shared so they can be sent as args to SYCL Kernel 
    """

    a_host = np.full(n, 2.0, dtype=np.float32)
    b_host = np.full(n, 2.0, dtype=np.float32)
    c_host = np.empty_like (a_host)

    a_dev = dpm.MemoryUSMShared(a_host.nbytes, queue= q)
    b_dev = dpm.MemoryUSMShared(b_host.nbytes, queue= q)
    c_dev = dpm.MemoryUSMShared(c_host.nbytes, queue= q)
    
    q.memcpy(a_dev, a_host, a_host.nbytes)
    q.memcpy(b_dev, b_host, b_host.nbytes)
    c_dev.memset(0)

    return a_dev, b_dev, c_dev, c_host


def print_results(q: SYnergyQueue, c_dev, c_host) -> bool:
    q.memcpy(c_host, c_dev, c_host.nbytes)

    print("Vettore risultante:", c_host[:10])

def get_kernel_time(event):
    """
    Uses the event profiling information to extract kernel execution informations.
    Return None if profiling informations are not available. 
    """

    try:
        return ((event.profiling_info_end - event.profiling_info_start) / 1_000_000.0)
    except Exception:
        return None

def run_single_submission(q, kernel, a_dev, b_dev, c_dev, n):
    """
    Measure informations on one kernel submission

    The variable "host_time_ms" include the Python-level submit call, the native bridge, the actual kernel execution and the final wait. 
    """
    c_dev.memset(0)
    q.wait()
    t0 = time.perf_counter()

    event = q.submit(
        kernel,
        args=[
            a_dev,
            b_dev,
            c_dev,
            ctypes.c_uint32(n),
            ctypes.c_uint32(REPEAT),
        ],
        gS=[n],
        lS=[LOCAL_SIZE],
        use_device_profiling=True,
        use_kernel_profiling=True,
    )

    event.wait()
    t1 = time.perf_counter()

    host_time_ms = (t1 - t0) * 1000.0
    kernel_time_ms = get_kernel_time(event)

    profile = q.last_profile
    device_energy_delta = profile.get("device_energy_delta")
    kernel_energy = profile.get("kernel_energy")

    overhead_ms = None
    overhead_ratio = None

    if kernel_time_ms is not None:
        overhead_ms = host_time_ms - kernel_time_ms
        overhead_ratio = overhead_ms / host_time_ms if host_time_ms > 0 else None

    return {
        "host_time_ms": host_time_ms,
        "kernel_time_ms": kernel_time_ms,
        "overhead_ms": overhead_ms,
        "overhead_ratio": overhead_ratio,
        "device_energy_delta": device_energy_delta,
        "kernel_energy": kernel_energy,
    }

def main() -> int:
    print("Creating Queue")

    q = SYnergyQueue(
        DEVICE_SELECTOR,
        execution_backend="synergy"
        )
    
    print("Used device: ", q.synergy_device_name)

    if not hasattr(synergy_submit, "create_vecprod_kernel"):
        raise RuntimeError(
            "bindings._synergy_submit does not expose create_vecprod_kernel()\n"
            "Make sure the name is correct and that _synergy_submit has been compiled correctly"
        )
    
    #Creazione SyclKernel
    kernel = synergy_submit.create_vecprod_kernel(q._adapter)

    try:
        print("Kernel name:", kernel.get_function_name())
    except Exception as exc:
        print("Kernel name unavailable", repr(exc))

    fieldnames = [
        "implementation",
        "device",
        "n",
        "repeat",
        "local_size",
        "run_type",
        "run_index",
        "host_time_ms",
        "kernel_time_ms",
        "overhead_ms",
        "overhead_ratio",
        "device_energy_delta",
        "kernel_energy",
    ]
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as f: 

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for n in SIZES:
            print(f"\n===== SIZE n={n} ======")
            a_dev, b_dev, c_dev, c_host = allocate_usm_vectors(q, n)

            print("\nWarmup Phase")
            for warmup_idx in range(WARMUP_RUNS):
                _ = run_single_submission(q, kernel, a_dev, b_dev, c_dev, n)
            print("\nMeasured Phase")
            for run_idx in range(MEASURED_RUNS):
                metrics = run_single_submission(q, kernel, a_dev, b_dev, c_dev, n)

                row = {
                    "implementation": "python",
                    "device": q.synergy_device_name,
                    "n": n,
                    "repeat": REPEAT,
                    "local_size": LOCAL_SIZE,
                    "run_type": "measured",
                    "run_index": run_idx,
                    **metrics,
                }

                writer.writerow(row)

                print(
                    f"run={run_idx:02d} "
                    f"host={metrics['host_time_ms']:.4f}ms "
                    f"kernel={metrics['kernel_time_ms']}ms "
                    f"energy={metrics['device_energy_delta']}"
                )

            del a_dev, b_dev, c_dev, c_host
            gc.collect()

            # Small pause to reduce thermal/queue carry-over effects between sizes.
            time.sleep(3.0)
        
if __name__ == "__main__":
    raise SystemExit(main())
