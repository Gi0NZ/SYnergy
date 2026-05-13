from __future__ import annotations

import ctypes
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import dpctl
import dpctl.program as dpctl_program
import dpctl.memory as dpm

from bindings import SYnergyQueue



VECTOR_ADD_OPENCL_SOURCE = r"""
__kernel void vector_add(
    __global const float *a,
    __global const float *b,
    __global float *c,
    const unsigned int n,
    const unsigned int repeat)
{
    unsigned int i = get_global_id(0);

    if (i < n) {
        float value = a[i] + b[i];

        for (unsigned int r = 0; r < repeat; ++r) {
            value = value + 0.0f;
        }

        c[i] = value;
    }
}
"""


def find_oneapi_tool(name: str) -> str | None:
    path = shutil.which(name)
    if path:
        return path

    fallback = Path(f"/opt/intel/oneapi/compiler/2025.3/bin/compiler/{name}")
    if fallback.exists():
        return str(fallback)

    return None


def generate_spirv_kernel(tmpdir: Path) -> Path | None:
    clang = find_oneapi_tool("clang")
    llvm_spirv = find_oneapi_tool("llvm-spirv")

    if clang is None or llvm_spirv is None:
        print("[SKIP] SPIR-V generation: clang or llvm-spirv not found.")
        return None

    cl_path = tmpdir / "vector_add.cl"
    bc_path = tmpdir / "vector_add.bc"
    spv_path = tmpdir / "vector_add.spv"

    cl_path.write_text(VECTOR_ADD_OPENCL_SOURCE)

    subprocess.run(
        [
            clang,
            "-cc1",
            "-triple",
            "spir64-unknown-unknown",
            "-cl-std=CL2.0",
            "-finclude-default-header",
            "-emit-llvm-bc",
            str(cl_path),
            "-o",
            str(bc_path),
        ],
        check=True,
    )

    subprocess.run(
        [
            llvm_spirv,
            str(bc_path),
            "-o",
            str(spv_path),
        ],
        check=True,
    )

    if not spv_path.exists():
        print("[FAIL] SPIR-V generation: .spv file was not created.")
        return None

    return spv_path


def allocate_vector_data(q, n: int):
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


def check_vector_result(q, c_dev, c_host, expected_value=3.0) -> bool:
    q.memcpy(c_host, c_dev, c_host.nbytes)

    expected = np.full_like(c_host, expected_value)
    max_abs_err = float(np.max(np.abs(c_host - expected)))

    print("Max abs error:", max_abs_err)
    print("First values:", c_host[:10])

    return max_abs_err < 1e-4


def run_vector_add_dpctl_submit(label: str, q: dpctl.SyclQueue, kernel) -> str:
    print()
    print(f"=== {label} ===")

    n = 1024 * 1024
    repeat = 64

    a_dev, b_dev, c_dev, c_host = allocate_vector_data(q, n)

    try:
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
        )

        event.wait()

        print("Event:", event)

        if check_vector_result(q, c_dev, c_host):
            print(f"[PASS] {label}")
            return "PASS"

        print(f"[FAIL] {label}: wrong result.")
        return "FAIL"

    except Exception as exc:
        print(f"[FAIL] {label}: unexpected exception.")
        print(type(exc).__name__, exc)
        return "FAIL"


def run_vector_add_synergy_submit(label: str, q, submitter) -> str:
    print()
    print(f"=== {label} ===")

    n = 1024 * 1024
    repeat = 64

    a_dev, b_dev, c_dev, c_host = allocate_vector_data(q, n)

    try:
        event = submitter(
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

        event.wait()

        print("Event:", event)
        print("Last profile:", q.last_profile)

        if check_vector_result(q, c_dev, c_host):
            print(f"[PASS] {label}")
            return "PASS"

        print(f"[FAIL] {label}: wrong result.")
        return "FAIL"

    except RuntimeError as exc:
        message = str(exc)

        if "may not support runtime program creation" in message:
            print(f"[SKIP] {label}: backend does not support this kernel creation path.")
            print(message)
            return "SKIP"

        print(f"[FAIL] {label}: unexpected RuntimeError.")
        print(message)
        return "FAIL"

    except Exception as exc:
        print(f"[FAIL] {label}: unexpected exception.")
        print(type(exc).__name__, exc)
        return "FAIL"


def test_submit_validation(q) -> str:
    print()
    print("=== submit validation ===")

    try:
        q.submit(None, args=[], gS=[1])
    except (ValueError, TypeError) as exc:
        print("[PASS] submit rejects invalid kernel.")
        print(type(exc).__name__, exc)
        return "PASS"

    print("[FAIL] submit accepted an invalid kernel.")
    return "FAIL"


def test_direct_kernel_if_available(q) -> str:
    """
    Temporary direct-kernel check.

    This uses the existing create_busy_kernel helper only to verify the direct
    SyclKernel submit path while SPIR-V/OpenCL runtime creation is unsupported
    on the current CUDA-only setup.
    """

    print()
    print("=== direct SyclKernel submit ===")

    try:
        import bindings._synergy_submit as synergy_submit

        if not hasattr(synergy_submit, "create_busy_kernel"):
            print("[SKIP] create_busy_kernel not available.")
            return "SKIP"

        kernel = synergy_submit.create_busy_kernel(q._adapter)

        return run_vector_add_synergy_submit(
            "direct SyclKernel submit",
            q,
            lambda **kwargs: q.submit(kernel, **kwargs),
        )

    except Exception as exc:
        print("[FAIL] direct SyclKernel setup failed.")
        print(type(exc).__name__, exc)
        return "FAIL"


def test_spirv_submit(q: dpctl.SyclQueue) -> str:
    with tempfile.TemporaryDirectory(prefix="synergy_spirv_") as tmp:
        spv_path = generate_spirv_kernel(Path(tmp))

        if spv_path is None:
            return "SKIP"

        print()
        print("=== SPIR-V kernel creation through dpctl ===")

        try:
            il = spv_path.read_bytes()

            program = dpctl_program.create_program_from_spirv(
                q,
                il,
                copts="",
            )

            kernel = program.get_sycl_kernel("vector_add")

        except Exception as exc:
            print("[FAIL] SPIR-V kernel creation failed.")
            print(type(exc).__name__, exc)
            return "FAIL"

        return run_vector_add_dpctl_submit(
            "SPIR-V submit through dpctl.SyclQueue",
            q,
            kernel,
        )


def test_opencl_source_submit(q: dpctl.SyclQueue) -> str:
    print()
    print("=== OpenCL source kernel creation through dpctl ===")

    try:
        program = dpctl_program.create_program_from_source(
            q,
            VECTOR_ADD_OPENCL_SOURCE,
            copts="",
        )

        kernel = program.get_sycl_kernel("vector_add")

    except Exception as exc:
        print("[FAIL] OpenCL source kernel creation failed.")
        print(type(exc).__name__, exc)
        return "FAIL"

    return run_vector_add_dpctl_submit(
        "OpenCL source submit through dpctl.SyclQueue",
        q,
        kernel,
    )


def main():
    qg = SYnergyQueue("cuda:gpu:0")
    qc = dpctl.SyclQueue("opencl:cpu:0")

    print("=== Queue GPU info ===")
    print("Device:", qg.synergy_device_name)
    print("Backend:", qg.synergy_backend_name)
    print("Capabilities:", qg.capabilities())

    print()
    print("=== Queue CPU info ===")
    print("Device:", qc.sycl_device.name)
    print("Backend:", qc.sycl_device.backend)

    results = [
        test_submit_validation(qg),
        test_direct_kernel_if_available(qg),
        test_spirv_submit(qc),
        test_opencl_source_submit(qc),
    ]

    print()
    print("=== Summary ===")
    print("PASS:", results.count("PASS"))
    print("SKIP:", results.count("SKIP"))
    print("FAIL:", results.count("FAIL"))

    if "FAIL" in results:
        raise SystemExit(1)

    raise SystemExit(0)


if __name__ == "__main__":
    main()



    