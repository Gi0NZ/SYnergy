import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
bindings_dir = os.path.abspath(os.path.join(current_dir, "bindings"))

if bindings_dir not in sys.path:
    sys.path.append(bindings_dir)

import dpctl as d
import dpctl.tensor as dpt
import synergy_custom

N = 8192 * 2048
#Poiché tramite comando funziona specificando la gpu, in questo modo:
# ONEAPI_DEVICE_SELECTOR=cuda:gpu python python/vector_sum.py
#Vado a specificare direttamente di prendere la gpu tramite "cuda:gpu:0"

q = d.SyclQueue("cuda:gpu:0")

print("Using device:")
q.sycl_device.print_device_info()
a = dpt.ones(N, dtype=dpt.float32, sycl_queue=q)
b = dpt.ones(N, dtype=dpt.float32, sycl_queue=q)
c = dpt.zeros(N, dtype=dpt.float32, sycl_queue=q)


# Passiamo direttamente queue e array al binding, non i loro indirizzi.
energia = synergy_custom.run_vector_add(q, a, b, c, N)

print(f"Calcolo terminato - Energia consumata: {energia:.4f} J")
print(f"Verifica C[0]: {float(c[0])}")


