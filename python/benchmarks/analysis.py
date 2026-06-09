from pathlib import Path
import csv
import pandas as pd

CPP_CSV_PATH = Path("../../results/cpp_vecprod_results.csv")
PYTHON_CSV_PATH = Path("../../results/python_vecprod_results.csv")
OUTPUT_CSV = Path("../../results/mean_values.csv")

df = pd.read_csv(CPP_CSV_PATH)
sizes = df["n"].unique()

df1 = df.groupby(["implementation","n"])[["implementation", "host_time_ms", "kernel_time_ms", "device_energy_delta", "kernel_energy"]].agg(
    {"host_time_ms": "mean",
    "kernel_time_ms": "mean",
    "device_energy_delta": "mean",
    "kernel_energy": "mean",}
)

print(df1)
dfp = pd.read_csv(PYTHON_CSV_PATH)

df2 = dfp.groupby(["implementation","n"])[["implementation", "host_time_ms", "kernel_time_ms", "device_energy_delta", "kernel_energy"]].agg(
    {"host_time_ms": "mean",
    "kernel_time_ms": "mean",
    "device_energy_delta": "mean",
    "kernel_energy": "mean",}
)

dfc = pd.concat([df1, df2])

dfc.to_csv(OUTPUT_CSV)

