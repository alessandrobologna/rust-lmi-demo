#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3>=1.42.1", "matplotlib>=3.8", "click>=8.1", "seaborn>=0.13", "pandas>=2.2"]
# ///
"""
Load test benchmark + plotting script for comparing Lambda endpoints.

This uses uv + PEP 723 inline metadata, so you can run it without
pre-installing dependencies:

  uv run benchmark.py --stack rust-lmi-demo

The script:
  1. Fetches API endpoints from CloudFormation stack outputs.
  2. Runs k6 load test comparing Default vs LMI endpoints.
  3. Parses the CSV results and generates comparison charts.
"""
import os
import subprocess
from datetime import datetime
from pathlib import Path

import boto3
import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_stack_outputs(stack_name: str, region: str | None) -> dict[str, str]:
    """Fetch CloudFormation stack outputs."""
    if region:
        client = boto3.client("cloudformation", region_name=region)
    else:
        client = boto3.client("cloudformation")

    response = client.describe_stacks(StackName=stack_name)
    if not response["Stacks"]:
        raise RuntimeError(f"Stack {stack_name} not found")

    outputs = {}
    for output in response["Stacks"][0].get("Outputs", []):
        outputs[output["OutputKey"]] = output["OutputValue"]

    return outputs


def run_k6_test(
    default_url: str,
    lmi_url: str,
    csv_path: Path,
    duration: str,
    vus: int,
) -> None:
    """Run k6 load test with CSV output using environment variables."""
    env = os.environ.copy()
    env["DEFAULT_URL"] = default_url
    env["LMI_URL"] = lmi_url
    env["DURATION"] = duration
    env["VUS"] = str(vus)

    cmd = [
        "k6", "run",
        "--out", f"csv={csv_path}",
        "loadtest.js",
    ]

    print(f"Running: {' '.join(cmd)}")
    print(f"  DURATION={duration}, VUS={vus}")

    # Don't capture output - let k6 print to terminal
    result = subprocess.run(cmd, env=env)
    # Exit code 99 means thresholds crossed - still continue to generate report
    if result.returncode != 0 and result.returncode != 99:
        raise RuntimeError(f"k6 failed with return code {result.returncode}")
    if result.returncode == 99:
        print("Warning: k6 thresholds were crossed (this is expected under heavy load)")


def parse_k6_csv(csv_path: Path) -> pd.DataFrame:
    """Parse k6 CSV output into a DataFrame."""
    df = pd.read_csv(csv_path)

    # Filter for http_req_duration metric only
    df = df[df["metric_name"] == "http_req_duration"].copy()

    # Parse the extra_tags column to extract endpoint tag
    def extract_endpoint(extra_tags: str) -> str:
        if pd.isna(extra_tags):
            return "unknown"
        extra_str = str(extra_tags)
        if "endpoint=default" in extra_str:
            return "default"
        elif "endpoint=lmi" in extra_str:
            return "lmi"
        return "unknown"

    df["endpoint"] = df["extra_tags"].apply(extract_endpoint)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df["latency_ms"] = df["metric_value"]

    return df[["timestamp", "endpoint", "latency_ms"]]


def calculate_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate statistics per endpoint."""
    stats = df.groupby("endpoint")["latency_ms"].agg([
        ("count", "count"),
        ("avg", "mean"),
        ("min", "min"),
        ("max", "max"),
        ("p50", lambda x: x.quantile(0.50)),
        ("p90", lambda x: x.quantile(0.90)),
        ("p95", lambda x: x.quantile(0.95)),
        ("p99", lambda x: x.quantile(0.99)),
    ]).round(2)
    return stats


def plot_results(
    df: pd.DataFrame,
    stats: pd.DataFrame,
    output_path: Path,
    duration: str,
) -> None:
    """Generate charts for the test run."""
    sns.set_theme(style="darkgrid", context="notebook", rc={
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "axes.edgecolor": ".5",
        "axes.labelcolor": ".5",
        "xtick.color": ".5",
        "ytick.color": ".5",
        "grid.color": ".5",
        "grid.alpha": 0.3,
        "text.color": ".5",
        "axes.titlecolor": ".5",
        "legend.facecolor": "none",
        "legend.edgecolor": ".5",
        "legend.framealpha": 0.0,
        "legend.labelcolor": ".5",
    })

    palette = {"default": "#4C78A8", "lmi": "#E45756"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    min_ts = df["timestamp"].min()
    df["elapsed_seconds"] = (df["timestamp"] - min_ts).dt.total_seconds()

    # Calculate phase boundaries
    if duration.endswith("s"):
        duration_seconds = int(duration[:-1])
    elif duration.endswith("m"):
        duration_seconds = int(duration[:-1]) * 60
    else:
        duration_seconds = 60

    phase1_end = duration_seconds
    phase2_end = duration_seconds * 2

    # Plot 1: Latency over time
    ax1 = axes[0, 0]
    for endpoint in ["default", "lmi"]:
        data = df[df["endpoint"] == endpoint]
        ax1.scatter(
            data["elapsed_seconds"],
            data["latency_ms"],
            alpha=0.3,
            s=10,
            label=endpoint,
            color=palette[endpoint],
        )

    ax1.axvline(x=phase1_end, color="gray", linestyle="--", alpha=0.5)
    ax1.axvline(x=phase2_end, color="gray", linestyle="--", alpha=0.5)
    ymax = ax1.get_ylim()[1]
    ax1.text(phase1_end / 2, ymax * 0.95, "Ramp Up", ha="center", fontsize=9, color=".5")
    ax1.text((phase1_end + phase2_end) / 2, ymax * 0.95, "Sustained", ha="center", fontsize=9, color=".5")
    ax1.text((phase2_end + duration_seconds * 3) / 2, ymax * 0.95, "Ramp Down", ha="center", fontsize=9, color=".5")
    ax1.set_xlabel("Elapsed Time (seconds)")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Response Latency Over Time")
    ax1.legend(title="Endpoint")

    # Plot 2: Rolling average
    ax2 = axes[0, 1]
    for endpoint in ["default", "lmi"]:
        data = df[df["endpoint"] == endpoint].sort_values("elapsed_seconds")
        data_resampled = data.set_index("timestamp").resample("1s")["latency_ms"].mean().reset_index()
        data_resampled["elapsed_seconds"] = (data_resampled["timestamp"] - min_ts).dt.total_seconds()
        ax2.plot(
            data_resampled["elapsed_seconds"],
            data_resampled["latency_ms"],
            label=endpoint,
            color=palette[endpoint],
            linewidth=2,
        )

    ax2.axvline(x=phase1_end, color="gray", linestyle="--", alpha=0.5)
    ax2.axvline(x=phase2_end, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Elapsed Time (seconds)")
    ax2.set_ylabel("Avg Latency (ms)")
    ax2.set_title("Average Latency (1s buckets)")
    ax2.legend(title="Endpoint")

    # Plot 3: Box plot
    ax3 = axes[1, 0]
    sns.boxplot(data=df, x="endpoint", y="latency_ms", hue="endpoint", palette=palette, legend=False, ax=ax3)
    ax3.set_xlabel("Endpoint")
    ax3.set_ylabel("Latency (ms)")
    ax3.set_title("Latency Distribution")

    # Plot 4: Stats comparison
    ax4 = axes[1, 1]
    stats_melted = stats.reset_index().melt(
        id_vars=["endpoint"],
        value_vars=["avg", "p50", "p95", "p99"],
        var_name="statistic",
        value_name="latency_ms",
    )
    sns.barplot(data=stats_melted, x="statistic", y="latency_ms", hue="endpoint", palette=palette, ax=ax4)
    ax4.set_xlabel("Statistic")
    ax4.set_ylabel("Latency (ms)")
    ax4.set_title("Latency Statistics Comparison")
    ax4.legend(title="Endpoint")

    fig.suptitle("Lambda Load Test: Default vs LMI", fontsize=14, y=1.02, color=".5")
    fig.tight_layout()
    fig.savefig(output_path, transparent=True, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote chart: {output_path}")


def print_comparison_table(stats: pd.DataFrame) -> None:
    """Print a formatted comparison table."""
    print(f"\n{'=' * 70}")
    print("LOAD TEST RESULTS")
    print("=" * 70)

    if "default" in stats.index and "lmi" in stats.index:
        default = stats.loc["default"]
        lmi = stats.loc["lmi"]

        def diff_pct(d: float, l: float) -> str:
            if d == 0:
                return "N/A"
            pct = ((l - d) / d) * 100
            sign = "+" if pct > 0 else ""
            return f"{sign}{pct:.1f}%"

        print(f"\n{'Metric':<20} {'Default':>12} {'LMI':>12} {'Diff':>10}")
        print("-" * 56)
        print(f"{'Requests':<20} {int(default['count']):>12} {int(lmi['count']):>12} {'':>10}")
        print(f"{'Avg Latency (ms)':<20} {default['avg']:>12.1f} {lmi['avg']:>12.1f} {diff_pct(default['avg'], lmi['avg']):>10}")
        print(f"{'Min Latency (ms)':<20} {default['min']:>12.1f} {lmi['min']:>12.1f} {diff_pct(default['min'], lmi['min']):>10}")
        print(f"{'p50 Latency (ms)':<20} {default['p50']:>12.1f} {lmi['p50']:>12.1f} {diff_pct(default['p50'], lmi['p50']):>10}")
        print(f"{'p90 Latency (ms)':<20} {default['p90']:>12.1f} {lmi['p90']:>12.1f} {diff_pct(default['p90'], lmi['p90']):>10}")
        print(f"{'p95 Latency (ms)':<20} {default['p95']:>12.1f} {lmi['p95']:>12.1f} {diff_pct(default['p95'], lmi['p95']):>10}")
        print(f"{'p99 Latency (ms)':<20} {default['p99']:>12.1f} {lmi['p99']:>12.1f} {diff_pct(default['p99'], lmi['p99']):>10}")
        print(f"{'Max Latency (ms)':<20} {default['max']:>12.1f} {lmi['max']:>12.1f} {diff_pct(default['max'], lmi['max']):>10}")
        print("-" * 56)
    else:
        print(stats.to_string())

    print("=" * 70 + "\n")


@click.command()
@click.option("--stack", "stack_name", required=True, help="CloudFormation stack name")
@click.option("--region", default=None, help="AWS region (overrides AWS_REGION)")
@click.option("--duration", default="30s", help="Duration per phase (ramp up, sustain, ramp down)")
@click.option("--vus", default=50, type=int, help="Number of virtual users at peak")
@click.option(
    "--output-dir",
    "output_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path.cwd() / "benchmark-results",
    help="Directory to write results",
)
@click.option(
    "--skip-test",
    is_flag=True,
    default=False,
    help="Skip running k6 and regenerate charts from existing CSV",
)
def main(
    stack_name: str,
    region: str | None,
    duration: str,
    vus: int,
    output_dir: Path,
    skip_test: bool,
) -> None:
    """Run load test benchmark comparing Default vs LMI Lambda endpoints."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Get endpoints from CloudFormation
    print(f"Fetching endpoints from stack: {stack_name}")
    outputs = get_stack_outputs(stack_name, region)

    default_url = outputs.get("HelloWorldApiDefaultEndpoint")
    lmi_url = outputs.get("HelloWorldApiLMIEndpoint")

    if not default_url or not lmi_url:
        raise SystemExit(f"Could not find endpoint outputs. Found: {list(outputs.keys())}")

    print(f"Default URL: {default_url}")
    print(f"LMI URL: {lmi_url}")

    csv_path = output_dir / f"k6-{timestamp}.csv"
    chart_path = output_dir / f"benchmark-{timestamp}.png"

    if skip_test:
        # Find most recent CSV
        csv_files = sorted(output_dir.glob("k6-*.csv"), reverse=True)
        if not csv_files:
            raise SystemExit("No CSV found. Run without --skip-test first.")
        csv_path = csv_files[0]
        print(f"Using existing CSV: {csv_path}")
    else:
        print(f"\n{'=' * 70}")
        print(f"RUNNING TEST: {vus} VUs, {duration} per phase")
        print("=" * 70)
        run_k6_test(default_url, lmi_url, csv_path, duration, vus)

    # Parse and analyze
    print(f"\nParsing results from: {csv_path}")
    df = parse_k6_csv(csv_path)

    if df.empty:
        raise SystemExit("No data found in CSV.")

    stats = calculate_stats(df)

    # Print stats
    print_comparison_table(stats)

    # Generate chart
    plot_results(df, stats, chart_path, duration)

    print(f"\n{'=' * 70}")
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
