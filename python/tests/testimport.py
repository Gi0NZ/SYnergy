import sys
import os
import dpctl

# 1. Troviamo la cartella "bindings" in modo dinamico e la aggiungiamo al PATH
# Questo permette a Python di trovare SIA synergy_queue.py SIA synergy_backend.so
current_dir = os.path.dirname(os.path.abspath(__file__))
bindings_dir = os.path.abspath(os.path.join(current_dir, '../bindings'))
sys.path.append(bindings_dir)

# 2. Ora possiamo importare in modo pulito e diretto!
from synergy_queue import SynergyQueue

# --- INIZIO TEST ---
print("Cerco un device compatibile...")
device = dpctl.SyclDevice("gpu")
print(f"Trovato! Usando il device: {device.name}")

# Inizializziamo la nostra coda speciale
sq = SynergyQueue(device)

# Proviamo a leggere l'energia hardware in tempo reale dal C++!
try:
    energia = sq.get_current_energy()
    print(f"🎉 SUCCESSO! Energia attuale del device: {energia} Joule")
except Exception as e:
    print(f"❌ Errore durante la lettura dell'energia: {e}")

