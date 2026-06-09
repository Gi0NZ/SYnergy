from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

PYTHON_CSV = ROOT / "results" / "python_vecprod_results.csv"
CPP_CSV = ROOT / "results" / "cpp_vecprod_results.csv"

PLOTS_DIR = ROOT / "results" / "plots"
SUMMARY_CSV = ROOT / "results" / "vecprod_summary.csv"


def load_results() -> pd.DataFrame:
    python_df = pd.read_csv(PYTHON_CSV)
    cpp_df = pd.read_csv(CPP_CSV)

    df = pd.concat([python_df, cpp_df], ignore_index=True)

    # Ensure numeric columns are correctly parsed.
    numeric_cols = [
        "n",
        "repeat",
        "local_size",
        "host_time_ms",
        "kernel_time_ms",
        "overhead_ms",
        "overhead_ratio",
        "device_energy_delta",
        "kernel_energy",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Recompute overhead metrics to be safe.
    df["overhead_ms"] = df["host_time_ms"] - df["kernel_time_ms"]
    df["overhead_ratio"] = df["overhead_ms"] / df["host_time_ms"]

    # Useful for x-axis labels.
    df["log2_n"] = df["n"].apply(lambda x: int(x).bit_length() - 1)
    df["n_label"] = df["log2_n"].apply(lambda x: rf"$2^{{{x}}}$")

    return df


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["implementation", "n", "log2_n"], as_index=False)
        .agg(
            host_time_mean=("host_time_ms", "mean"),
            host_time_std=("host_time_ms", "std"),
            kernel_time_mean=("kernel_time_ms", "mean"),
            kernel_time_std=("kernel_time_ms", "std"),
            overhead_mean=("overhead_ms", "mean"),
            overhead_std=("overhead_ms", "std"),
            overhead_ratio_mean=("overhead_ratio", "mean"),
            overhead_ratio_std=("overhead_ratio", "std"),
            device_energy_mean=("device_energy_delta", "mean"),
            device_energy_std=("device_energy_delta", "std"),
            kernel_energy_mean=("kernel_energy", "mean"),
            kernel_energy_std=("kernel_energy", "std"),
        )
        .sort_values(["implementation", "n"])
    )

    return summary


def plot_metric(
    summary: pd.DataFrame,
    metric_mean: str,
    metric_std: str,
    ylabel: str,
    title: str,
    output_name: str,
    log_y: bool = False,
):
    plt.figure(figsize=(7.2, 4.6))

    for impl in ["cpp", "python"]:
        data = summary[summary["implementation"] == impl]

        plt.errorbar(
            data["log2_n"],
            data[metric_mean],
            yerr=data[metric_std],
            marker="o",
            capsize=3,
            linewidth=1.5,
            label=impl.upper(),
        )

    plt.xlabel(r"Input size $n$ ($2^x$ elements)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()

    x_ticks = sorted(summary["log2_n"].unique())
    plt.xticks(x_ticks, [rf"$2^{{{x}}}$" for x in x_ticks], rotation=45)

    if log_y:
        plt.yscale("log")

    plt.tight_layout()

    out_pdf = PLOTS_DIR / f"{output_name}.pdf"
    out_png = PLOTS_DIR / f"{output_name}.png"

    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


def plot_execution_times(summary: pd.DataFrame):
    plot_metric(
        summary=summary,
        metric_mean="host_time_mean",
        metric_std="host_time_std",
        ylabel="Host-side time (ms)",
        title="Host-side execution time across input sizes",
        output_name="host_time_vs_size",
        log_y=True,
    )

    plot_metric(
        summary=summary,
        metric_mean="kernel_time_mean",
        metric_std="kernel_time_std",
        ylabel="Kernel time (ms)",
        title="Kernel execution time across input sizes",
        output_name="kernel_time_vs_size",
        log_y=True,
    )


def plot_overhead(summary: pd.DataFrame):
    plot_metric(
        summary=summary,
        metric_mean="overhead_ratio_mean",
        metric_std="overhead_ratio_std",
        ylabel="Overhead ratio",
        title="Relative overhead across input sizes",
        output_name="overhead_ratio_vs_size",
        log_y=False,
    )


def plot_energy(summary: pd.DataFrame):
    plot_metric(
        summary=summary,
        metric_mean="kernel_energy_mean",
        metric_std="kernel_energy_std",
        ylabel="Kernel energy (J)",
        title="Kernel energy consumption across input sizes",
        output_name="kernel_energy_vs_size",
        log_y=True,
    )


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_results()
    summary = compute_summary(df)

    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved summary: {SUMMARY_CSV}")

    plot_execution_times(summary)
    plot_overhead(summary)
    plot_energy(summary)


if __name__ == "__main__":
    main()