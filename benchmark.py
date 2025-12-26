#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3>=1.42.1", "matplotlib>=3.8", "click>=8.1", "seaborn>=0.13", "pandas>=2.2", "scipy>=1.11"]
# ///
"""
Load test benchmark + plotting script for comparing Lambda endpoints.

This uses uv + PEP 723 inline metadata, so you can run it without
pre-installing dependencies:

  uv run benchmark.py --stack rust-lmi-demo

The script:
  1. Fetches API endpoints from CloudFormation stack outputs.
  2. Runs k6 load tests for CPU and I/O(wait) scenarios.
  3. Parses the CSV results and generates comparison charts.
"""
import os
import subprocess
import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cbook import boxplot_stats
from matplotlib.patches import Patch
from scipy.interpolate import make_interp_spline


MAX_DELAY_MS = 15_000
MAX_HASH_LOOPS = 1_000_000


def extract_endpoint(extra_tags: str) -> str:
    if pd.isna(extra_tags):
        return "unknown"
    for part in str(extra_tags).split(","):
        part = part.strip()
        if part.startswith("endpoint="):
            value = part.removeprefix("endpoint=").strip()
            if value.lower() == "lmi":
                return "LMI"
            return value
    return "unknown"


_DURATION_PART_RE = re.compile(r"(\d+)([smh])")


def parse_duration_to_seconds(value: str) -> int:
    """Parse a k6-style duration string like '30s', '5m', or '1h30m' to seconds."""
    if not value:
        raise ValueError("duration is empty")

    matches = list(_DURATION_PART_RE.finditer(value))
    if not matches or "".join(m.group(0) for m in matches) != value:
        raise ValueError(f"unsupported duration format: {value!r}")

    total = 0
    for match in matches:
        amount = int(match.group(1))
        unit = match.group(2)
        if unit == "s":
            total += amount
        elif unit == "m":
            total += amount * 60
        elif unit == "h":
            total += amount * 3600
        else:
            raise ValueError(f"unsupported duration unit: {unit}")
    return total


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


def list_capacity_provider_instance_ids(
    *,
    region: str | None,
    capacity_provider_arn: str,
) -> list[str]:
    """List EC2 instance IDs tagged for a Lambda managed instances capacity provider."""
    client = boto3.client("ec2", region_name=region) if region else boto3.client("ec2")
    paginator = client.get_paginator("describe_instances")
    instance_ids: set[str] = set()
    for page in paginator.paginate(
        Filters=[
            {"Name": "tag:aws:lambda:capacity-provider", "Values": [capacity_provider_arn]},
            {"Name": "instance-state-name", "Values": ["pending", "running"]},
        ],
    ):
        for reservation in page.get("Reservations", []):
            for instance in reservation.get("Instances", []):
                instance_id = instance.get("InstanceId")
                if instance_id:
                    instance_ids.add(instance_id)
    return sorted(instance_ids)


def resolve_capacity_provider_arn(*, region: str | None, stack_name: str, capacity_provider_arn: str | None) -> str | None:
    """Resolve a Lambda capacity provider ARN used for EC2 scale-in waiting.

    Returns None if region can't be determined and no ARN was provided.
    """
    if capacity_provider_arn:
        return capacity_provider_arn

    effective_region = region or boto3.session.Session().region_name
    if not effective_region:
        return None

    sts = boto3.client("sts", region_name=effective_region)
    account_id = sts.get_caller_identity()["Account"]
    capacity_provider_name = f"{stack_name}-cp-2-c8xlarge"
    return f"arn:aws:lambda:{effective_region}:{account_id}:capacity-provider:{capacity_provider_name}"


def wait_for_capacity_provider_scale_in(
    *,
    region: str | None,
    capacity_provider_arn: str,
    baseline_count: int,
    check_interval_seconds: int,
    max_wait_minutes: int,
) -> bool:
    """Wait until capacity provider instance count returns to baseline_count."""
    if check_interval_seconds <= 0:
        raise ValueError("check_interval_seconds must be > 0")
    if max_wait_minutes <= 0:
        raise ValueError("max_wait_minutes must be > 0")

    deadline = time.monotonic() + (max_wait_minutes * 60)
    while True:
        try:
            instance_ids = list_capacity_provider_instance_ids(region=region, capacity_provider_arn=capacity_provider_arn)
        except Exception as e:  # noqa: BLE001 - best-effort orchestration
            print(f"Warning: failed to query EC2 for capacity provider instances ({e}); skipping scale-in wait.")
            return False

        current = len(instance_ids)
        if current == baseline_count:
            print(f"Capacity provider scaled in to baseline ({current} instances).")
            return True
        if time.monotonic() >= deadline:
            print(
                "Warning: timed out waiting for capacity provider scale-in "
                f"(current={current}, baseline={baseline_count})."
            )
            return False

        print(
            "Waiting for capacity provider scale-in: "
            f"current={current}, baseline={baseline_count} (check every {check_interval_seconds}s)"
        )
        time.sleep(check_interval_seconds)


def run_k6_test(
    targets: list[dict],
    csv_path: Path,
    *,
    stages: list[dict],
    mode: str | None,
    delay_ms: int,
    hash_loops: int,
) -> None:
    """Run k6 load test with CSV output using environment variables."""
    env = os.environ.copy()
    env["TARGETS"] = json.dumps(targets)
    env["STAGES"] = json.dumps(stages)
    if mode is not None:
        env["MODE"] = mode
    env.setdefault("EXECUTOR", "ramping-arrival-rate")
    env["DELAY_MS"] = str(delay_ms)
    env["HASH_LOOPS"] = str(hash_loops)

    cmd = [
        "k6", "run",
        "--out", f"csv={csv_path}",
        "loadtest.js",
    ]

    print(f"Running: {' '.join(cmd)}")
    print(f"  STAGES={env['STAGES']}")
    if mode is not None:
        print(f"  MODE={mode}")
        if env.get("EXECUTOR"):
            print(f"  EXECUTOR={env['EXECUTOR']}")
        if mode == "per_endpoint":
            try:
                peak_target = max(int(s["target"]) for s in stages)
                if env.get("EXECUTOR") == "ramping-arrival-rate":
                    print(
                        "  NOTE: per_endpoint runs one k6 scenario per endpoint "
                        f"(peak total rps ≈ {peak_target * len(targets)})"
                    )
                else:
                    print(
                        "  NOTE: per_endpoint runs one k6 scenario per endpoint "
                        f"(peak total VUs ≈ {peak_target * len(targets)})"
                    )
            except Exception:
                pass
    for t in targets:
        print(f"  TARGET={t['name']} ({t['url']})")
    print(f"  WORKLOAD: delay_ms={delay_ms}, hash_loops={hash_loops}")

    # Don't capture output - let k6 print to terminal
    result = subprocess.run(cmd, env=env)
    # Exit code 99 means thresholds crossed - still continue to generate report
    if result.returncode != 0 and result.returncode != 99:
        raise RuntimeError(f"k6 failed with return code {result.returncode}")
    if result.returncode == 99:
        print("Warning: k6 thresholds were crossed (this is expected under heavy load)")


def load_k6_csv(csv_path: Path) -> pd.DataFrame:
    """Load k6 CSV output with only the columns we use."""
    df = pd.read_csv(
        csv_path,
        usecols=["metric_name", "timestamp", "metric_value", "status", "extra_tags"],
    )
    df["endpoint"] = df["extra_tags"].apply(extract_endpoint)
    return df


def parse_k6_latencies(df: pd.DataFrame) -> pd.DataFrame:
    """Extract per-request latency samples from a loaded k6 CSV DataFrame."""
    lat = df[df["metric_name"] == "http_req_duration"].copy()
    lat["timestamp"] = pd.to_datetime(lat["timestamp"], unit="s", utc=True)
    lat["latency_ms"] = lat["metric_value"]
    return lat[["timestamp", "endpoint", "latency_ms"]]


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


def calculate_error_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate HTTP error counts/rates per endpoint from http_reqs rows."""
    reqs = df[df["metric_name"] == "http_reqs"].copy()
    reqs["status"] = pd.to_numeric(reqs["status"], errors="coerce")
    reqs["is_error"] = reqs["status"] >= 400
    reqs["is_4xx"] = (reqs["status"] >= 400) & (reqs["status"] < 500)
    reqs["is_5xx"] = (reqs["status"] >= 500) & (reqs["status"] < 600)
    reqs["is_429"] = reqs["status"] == 429

    errors = reqs.groupby("endpoint").agg(
        requests=("status", "count"),
        errors=("is_error", "sum"),
        errors_4xx=("is_4xx", "sum"),
        errors_5xx=("is_5xx", "sum"),
        errors_429=("is_429", "sum"),
    )
    errors["error_rate"] = (errors["errors"] / errors["requests"]).fillna(0.0).round(6)
    return errors


def fetch_execution_environment_concurrency(
    *,
    region: str | None,
    function_name: str,
    resource: str,
    capacity_provider_name: str,
    start_time: datetime,
    end_time: datetime,
    period_seconds: int,
) -> pd.DataFrame:
    client = boto3.client("cloudwatch", region_name=region) if region else boto3.client("cloudwatch")
    metric = {
        "Namespace": "AWS/Lambda",
        "MetricName": "ExecutionEnvironmentConcurrency",
        "Dimensions": [
            {"Name": "FunctionName", "Value": function_name},
            {"Name": "Resource", "Value": resource},
            {"Name": "CapacityProviderName", "Value": capacity_provider_name},
        ],
    }
    queries = [
        {
            "Id": "avg",
            "MetricStat": {"Metric": metric, "Period": period_seconds, "Stat": "Average"},
            "ReturnData": True,
        },
        {
            "Id": "count",
            "MetricStat": {"Metric": metric, "Period": period_seconds, "Stat": "SampleCount"},
            "ReturnData": True,
        },
    ]
    results: dict[str, list[dict]] = {}
    next_token = None
    while True:
        kwargs = {
            "StartTime": start_time,
            "EndTime": end_time,
            "MetricDataQueries": queries,
            "ScanBy": "TimestampAscending",
        }
        if next_token:
            kwargs["NextToken"] = next_token
        response = client.get_metric_data(**kwargs)
        for result in response.get("MetricDataResults", []):
            results.setdefault(result["Id"], []).append(result)
        next_token = response.get("NextToken")
        if not next_token:
            break

    def merge_series(entries: list[dict]) -> pd.Series:
        timestamps: list[datetime] = []
        values: list[float] = []
        for entry in entries:
            timestamps.extend(entry.get("Timestamps", []))
            values.extend(entry.get("Values", []))
        if not timestamps:
            return pd.Series(dtype=float)
        series = pd.Series(values, index=pd.to_datetime(timestamps, utc=True)).sort_index()
        return series

    avg_series = merge_series(results.get("avg", []))
    count_series = merge_series(results.get("count", []))
    df = pd.DataFrame({"average": avg_series, "sample_count": count_series}).sort_index()
    return df


def save_execution_environment_concurrency_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    out = df.copy()
    out = out.reset_index().rename(columns={"index": "timestamp"})
    out.to_csv(path, index=False)


def load_execution_environment_concurrency_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    return df


def _smooth_concurrency_series(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    series = series.dropna()
    if len(series) < 2:
        return series.index.to_numpy(dtype=float), series.to_numpy(dtype=float)
    x = series.index.to_numpy(dtype=float)
    y = series.to_numpy(dtype=float)
    x_new = np.linspace(x.min(), x.max(), int(x.max() - x.min()) + 1)
    k = min(3, len(x) - 1)
    if k < 2:
        return x_new, np.interp(x_new, x, y)
    spline = make_interp_spline(x, y, k=k)
    return x_new, spline(x_new)


def _compute_stage_markers(stages: list[dict], data_max_elapsed: float) -> tuple[bool, list[float]]:
    if not stages:
        return False, []
    stage_durations = [parse_duration_to_seconds(s["duration"]) for s in stages]
    stage_ends = []
    total = 0.0
    for duration in stage_durations:
        total += duration
        stage_ends.append(total)
    if not stage_ends:
        return False, []
    show = abs(stage_ends[-1] - data_max_elapsed) <= max(30.0, data_max_elapsed * 0.10)
    return show, stage_ends


def draw_execution_environment_concurrency(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    title: str,
    show_legend: bool,
    stages: list[dict] | None = None,
) -> None:
    min_ts = df.index.min()
    elapsed_seconds = (df.index - min_ts).total_seconds()
    df_plot = df.copy()
    df_plot.index = elapsed_seconds

    x_avg, y_avg = _smooth_concurrency_series(df_plot["average"])
    x_cnt, y_cnt = _smooth_concurrency_series(df_plot["sample_count"])

    line_avg = ax.plot(
        x_avg,
        y_avg,
        label="Concurrency per execution environment",
        color="#1f77b4",
        linewidth=2,
    )[0]
    ax_count = ax.twinx()
    line_cnt = ax_count.plot(
        x_cnt,
        y_cnt,
        label="Execution environment count",
        color="#ff7f0e",
        linewidth=2,
    )[0]
    ax.set_title(title)
    ax.set_ylabel("Concurrency per execution environment")
    ax_count.set_ylabel("Execution environment count")
    ax_count.tick_params(axis="y", colors=".5")
    ax.set_xlabel("Elapsed Time (seconds)")
    data_max_elapsed = float(df_plot.index.max()) if len(df_plot.index) else 0.0
    if stages:
        show_markers, stage_ends = _compute_stage_markers(stages, data_max_elapsed)
        if show_markers:
            for x in stage_ends[:-1]:
                ax.axvline(x=x, color="gray", linestyle="--", alpha=0.5)
            ymax = ax.get_ylim()[1] if ax.get_ylim()[1] else 1
            starts = [0] + stage_ends[:-1]
            for i, (start, end) in enumerate(zip(starts, stage_ends, strict=False)):
                start_target = 0 if i == 0 else int(stages[i - 1]["target"])
                end_target = int(stages[i]["target"])
                ax.text(
                    (start + end) / 2,
                    ymax * 0.95,
                    f"{start_target}→{end_target} rps",
                    ha="center",
                    fontsize=9,
                    color=".5",
                )
    if show_legend:
        ax.legend(
            [line_avg, line_cnt],
            [line_avg.get_label(), line_cnt.get_label()],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=2,
            frameon=False,
        )
    return line_avg, line_cnt


def plot_execution_environment_concurrency(
    df: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
    stages: list[dict] | None = None,
) -> None:
    if df.empty:
        print("Warning: no CloudWatch concurrency data to plot.")
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    draw_execution_environment_concurrency(ax, df, title=title, show_legend=True, stages=stages)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(output_path, transparent=True, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote CloudWatch chart: {output_path}")


def plot_execution_environment_concurrency_grid(
    charts: list[tuple[str, pd.DataFrame]],
    *,
    output_path: Path,
    ncols: int = 2,
    stages: list[dict] | None = None,
) -> None:
    if not charts:
        return
    nrows = (len(charts) + ncols - 1) // ncols
    # Match the main benchmark chart canvas size/aspect.
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 10))
    axes = np.array(axes).reshape(nrows, ncols)
    handles = None
    labels = None
    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            if idx >= len(charts):
                ax.axis("off")
                continue
            title, df = charts[idx]
            line_avg, line_cnt = draw_execution_environment_concurrency(
                ax,
                df,
                title=title,
                show_legend=False,
                stages=stages,
            )
            if handles is None:
                handles = [line_avg, line_cnt]
                labels = [line_avg.get_label(), line_cnt.get_label()]
            idx += 1
    fig.suptitle("Lambda Managed Instances - Concurrency scaling", fontsize=14, y=0.995, color=".5")
    if handles and labels:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=2,
            frameon=False,
        )
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    fig.savefig(output_path, transparent=True, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote combined CloudWatch chart: {output_path}")


def plot_results(
    df: pd.DataFrame,
    stats: pd.DataFrame,
    output_path: Path,
    stages: list[dict],
    endpoints: list[str],
    title: str,
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

    default_colors = {"128": "#4C78A8", "512": "#F58518", "LMI": "#E45756", "lmi": "#E45756"}
    fallback_colors = sns.color_palette("tab10", n_colors=max(len(endpoints), 3)).as_hex()
    fallback_iter = iter(fallback_colors)
    palette = {}
    for endpoint in endpoints:
        if endpoint in default_colors:
            palette[endpoint] = default_colors[endpoint]
        else:
            palette[endpoint] = next(fallback_iter)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    min_ts = df["timestamp"].min()
    df["elapsed_seconds"] = (df["timestamp"] - min_ts).dt.total_seconds()
    data_max_elapsed = float(df["elapsed_seconds"].max())

    stage_durations = [parse_duration_to_seconds(s["duration"]) for s in stages]
    stage_ends = []
    total = 0
    for d in stage_durations:
        total += d
        stage_ends.append(total)
    stage_total_seconds = stage_ends[-1] if stage_ends else None
    show_stage_markers = (
        stage_total_seconds is not None and abs(stage_total_seconds - data_max_elapsed) <= max(30.0, data_max_elapsed * 0.10)
    )

    # Plot 1: Latency over time (scatter)
    ax1 = axes[0, 0]
    max_scatter_points_per_endpoint = 50_000
    scatter_parts = []
    for endpoint in endpoints:
        data = df[df["endpoint"] == endpoint][["elapsed_seconds", "latency_ms", "endpoint"]]
        if len(data) > max_scatter_points_per_endpoint:
            data = data.sample(n=max_scatter_points_per_endpoint, random_state=42)
        scatter_parts.append(data)

    scatter_df = pd.concat(scatter_parts, ignore_index=True)
    scatter_df = scatter_df.sample(frac=1, random_state=42)  # avoid one series hiding the other

    ax1.scatter(
        scatter_df["elapsed_seconds"],
        scatter_df["latency_ms"],
        alpha=0.25,
        s=8,
        c=scatter_df["endpoint"].map(palette),
        edgecolors="none",
    )

    if show_stage_markers:
        for x in stage_ends[:-1]:
            ax1.axvline(x=x, color="gray", linestyle="--", alpha=0.5)
    ymax = ax1.get_ylim()[1]
    if show_stage_markers:
        starts = [0] + stage_ends[:-1]
        for i, (start, end) in enumerate(zip(starts, stage_ends, strict=False)):
            start_target = 0 if i == 0 else int(stages[i - 1]["target"])
            end_target = int(stages[i]["target"])
            ax1.text(
                (start + end) / 2,
                ymax * 0.95,
                f"{start_target}→{end_target} rps",
                ha="center",
                fontsize=9,
                color=".5",
            )
    ax1.set_xlabel("Elapsed Time (seconds)")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Response Latency Over Time")

    # Plot 2: Rolling average
    ax2 = axes[0, 1]
    for endpoint in endpoints:
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

    if show_stage_markers:
        for x in stage_ends[:-1]:
            ax2.axvline(x=x, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Elapsed Time (seconds)")
    ax2.set_ylabel("Avg Latency (ms)")
    ax2.set_title("Average Latency (1s buckets)")
    # legend is shown at the figure level

    # Plot 3: Box plot
    ax3 = axes[1, 0]
    sns.boxplot(
        data=df,
        x="endpoint",
        y="latency_ms",
        hue="endpoint",
        order=list(endpoints),
        hue_order=list(endpoints),
        palette=palette,
        dodge=False,
        ax=ax3,
    )
    for patch in ax3.patches:
        patch.set_edgecolor(patch.get_facecolor())
        patch.set_linewidth(2)
    legend = ax3.get_legend()
    if legend:
        legend.remove()
    ax3.set_xlabel("Endpoint")
    ax3.set_ylabel("Latency (ms)")
    ax3.set_title("Latency Distribution")
    # Keep the boxplot readable by using whisker bounds instead of extreme outliers
    stats_by_endpoint = []
    for endpoint in endpoints:
        vals = df[df["endpoint"] == endpoint]["latency_ms"].to_numpy()
        if len(vals) > 0:
            stats_by_endpoint.append(boxplot_stats(vals, whis=1.5)[0])
    if stats_by_endpoint:
        y_min = min(s["whislo"] for s in stats_by_endpoint)
        y_max = max(s["whishi"] for s in stats_by_endpoint)
        if y_max > y_min:
            ax3.set_ylim(y_min * 0.98, y_max * 1.02)
    # legend is shown at the figure level

    # Plot 4: Stats comparison
    ax4 = axes[1, 1]
    stats_melted = stats.reset_index().melt(
        id_vars=["endpoint"],
        value_vars=["avg", "p50", "p95", "p99"],
        var_name="statistic",
        value_name="latency_ms",
    )
    sns.barplot(
        data=stats_melted,
        x="statistic",
        y="latency_ms",
        hue="endpoint",
        hue_order=list(endpoints),
        palette=palette,
        ax=ax4,
    )
    ax4.set_xlabel("Statistic")
    ax4.set_ylabel("Latency (ms)")
    ax4.set_title("Latency Statistics Comparison")
    legend = ax4.get_legend()
    if legend:
        legend.remove()

    fig.suptitle(title, fontsize=14, y=1.02, color=".5")
    fig.legend(
        handles=[Patch(facecolor=palette[e], edgecolor=palette[e], label=e) for e in endpoints],
        title="Endpoint",
        loc="lower center",
        ncol=len(endpoints),
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.98])
    fig.savefig(output_path, transparent=True, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote chart: {output_path}")


def print_comparison_table(stats: pd.DataFrame, *, baseline_name: str, variant_name: str) -> None:
    """Print a formatted comparison table."""
    print(f"\n{'=' * 70}")
    print("LOAD TEST RESULTS")
    print("=" * 70)

    if baseline_name in stats.index and variant_name in stats.index:
        baseline = stats.loc[baseline_name]
        variant = stats.loc[variant_name]
        value_width = max(len(baseline_name), len(variant_name), 7) + 2

        def diff_pct(d: float, l: float) -> str:
            if d == 0:
                return "N/A"
            pct = ((l - d) / d) * 100
            sign = "+" if pct > 0 else ""
            return f"{sign}{pct:.1f}%"

        def diff_pp(d: float, l: float) -> str:
            return f"{(l - d):+.1f}pp"

        print(f"\n{'Metric':<20} {baseline_name:>{value_width}} {variant_name:>{value_width}} {'Diff':>10}")
        print("-" * 56)
        print(f"{'Requests':<20} {int(baseline['count']):>{value_width}} {int(variant['count']):>{value_width}} {'':>10}")
        if "errors_4xx" in stats.columns:
            print(
                f"{'HTTP 4xx':<20} {int(baseline.get('errors_4xx', 0)):>{value_width}} {int(variant.get('errors_4xx', 0)):>{value_width}} {'':>10}"
            )
        if "errors_429" in stats.columns:
            print(
                f"{'HTTP 429':<20} {int(baseline.get('errors_429', 0)):>{value_width}} {int(variant.get('errors_429', 0)):>{value_width}} {'':>10}"
            )
        if "errors_5xx" in stats.columns:
            print(
                f"{'HTTP 5xx':<20} {int(baseline.get('errors_5xx', 0)):>{value_width}} {int(variant.get('errors_5xx', 0)):>{value_width}} {'':>10}"
            )
        if "error_rate" in stats.columns:
            print(
                f"{'Error rate':<20} {baseline.get('error_rate', 0.0) * 100:>{value_width}.1f}% {variant.get('error_rate', 0.0) * 100:>{value_width}.1f}% {diff_pp(baseline.get('error_rate', 0.0) * 100, variant.get('error_rate', 0.0) * 100):>10}"
            )
        print(f"{'Avg Latency (ms)':<20} {baseline['avg']:>{value_width}.1f} {variant['avg']:>{value_width}.1f} {diff_pct(baseline['avg'], variant['avg']):>10}")
        print(f"{'Min Latency (ms)':<20} {baseline['min']:>{value_width}.1f} {variant['min']:>{value_width}.1f} {diff_pct(baseline['min'], variant['min']):>10}")
        print(f"{'p50 Latency (ms)':<20} {baseline['p50']:>{value_width}.1f} {variant['p50']:>{value_width}.1f} {diff_pct(baseline['p50'], variant['p50']):>10}")
        print(f"{'p90 Latency (ms)':<20} {baseline['p90']:>{value_width}.1f} {variant['p90']:>{value_width}.1f} {diff_pct(baseline['p90'], variant['p90']):>10}")
        print(f"{'p95 Latency (ms)':<20} {baseline['p95']:>{value_width}.1f} {variant['p95']:>{value_width}.1f} {diff_pct(baseline['p95'], variant['p95']):>10}")
        print(f"{'p99 Latency (ms)':<20} {baseline['p99']:>{value_width}.1f} {variant['p99']:>{value_width}.1f} {diff_pct(baseline['p99'], variant['p99']):>10}")
        print(f"{'Max Latency (ms)':<20} {baseline['max']:>{value_width}.1f} {variant['max']:>{value_width}.1f} {diff_pct(baseline['max'], variant['max']):>10}")
        print("-" * 56)
    else:
        print(stats.to_string())

        print("=" * 70 + "\n")


def print_endpoint_table(stats: pd.DataFrame, *, endpoints: list[str]) -> None:
    """Print a formatted table of per-endpoint stats."""
    print(f"\n{'=' * 70}")
    print("LOAD TEST RESULTS")
    print("=" * 70)

    cols = ["count", "errors_4xx", "errors_429", "errors_5xx", "error_rate", "avg", "p95", "p99", "max"]
    missing = [c for c in cols if c not in stats.columns]
    if missing:
        print(stats.to_string())
        return

    name_width = max(max(len(e) for e in endpoints), len("Endpoint")) + 2
    print(
        f"\n{'Endpoint':<{name_width}} {'Reqs':>10} {'4xx':>6} {'429':>6} {'5xx':>6} {'Err%':>7} {'Avg(ms)':>10} {'p95':>10} {'p99':>10} {'Max':>10}"
    )
    print("-" * (name_width + 10 + 6 + 6 + 6 + 7 + 10 + 10 + 10 + 10 + 9))

    for endpoint in endpoints:
        if endpoint not in stats.index:
            continue
        row = stats.loc[endpoint]
        print(
            f"{endpoint:<{name_width}} "
            f"{int(row['count']):>10} "
            f"{int(row.get('errors_4xx', 0)):>6} "
            f"{int(row.get('errors_429', 0)):>6} "
            f"{int(row.get('errors_5xx', 0)):>6} "
            f"{row.get('error_rate', 0.0) * 100:>6.1f}% "
            f"{row['avg']:>10.1f} "
            f"{row['p95']:>10.1f} "
            f"{row['p99']:>10.1f} "
            f"{row['max']:>10.1f}"
        )


@click.command()
@click.option("--stack", "stack_name", required=True, help="CloudFormation stack name")
@click.option("--region", default=None, help="AWS region (overrides AWS_REGION)")
@click.option("--duration", default="30s", help="Duration per k6 stage (only used with --stage-targets)")
@click.option(
    "--stage-targets",
    default=None,
    help="Comma-separated arrival-rate targets (rps). Each target uses --duration. Overrides scenario defaults.",
)
@click.option(
    "--stages-json",
    default=None,
    help="Full k6 stages JSON (advanced). Overrides --stage-targets.",
)
@click.option(
    "--scenario",
    type=click.Choice(["bursty-io", "steady-io", "cpu-break", "mixed", "all"]),
    default="all",
    show_default=True,
    help="Which scenario(s) to run: bursty-io, steady-io, cpu-break, mixed, or all.",
)
@click.option(
    "--cpu-break-hash-loops",
    "cpu_break_hash_loops",
    type=int,
    default=200_000,
    show_default=True,
    help="hash_loops used for the CPU break-point scenario (delay_ms=0).",
)
@click.option(
    "--steady-io-delay-ms",
    "steady_io_delay_ms",
    type=int,
    default=100,
    show_default=True,
    help="delay_ms used for the Steady I/O scenario (hash_loops=0).",
)
@click.option(
    "--bursty-io-delay-ms",
    "bursty_io_delay_ms",
    type=int,
    default=500,
    show_default=True,
    help="delay_ms used for the Bursty I/O scenario (hash_loops=0).",
)
@click.option(
    "--mixed-delay-ms",
    "mixed_delay_ms",
    type=int,
    default=50,
    show_default=True,
    help="delay_ms used for the Mixed scenario.",
)
@click.option(
    "--mixed-hash-loops",
    "mixed_hash_loops",
    type=int,
    default=25_000,
    show_default=True,
    help="hash_loops used for the Mixed scenario.",
)
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
@click.option(
    "--wait-for-scale-in/--no-wait-for-scale-in",
    default=None,
    help="When running multiple scenarios, wait for the LMI capacity provider to scale back to its pre-run EC2 instance count before starting the next scenario.",
)
@click.option(
    "--capacity-provider-arn",
    default=None,
    help="Capacity provider ARN used to detect scale-out/scale-in EC2 instances (defaults to arn:aws:lambda:<region>:<account>:capacity-provider:<stack>-cp-2-c8xlarge).",
)
@click.option(
    "--scale-in-check-seconds",
    type=int,
    default=60,
    show_default=True,
    help="How often to check EC2 instance count while waiting for scale-in.",
)
@click.option(
    "--scale-in-max-minutes",
    type=int,
    default=90,
    show_default=True,
    help="Maximum time to wait for scale-in between scenarios.",
)
@click.option(
    "--cloudwatch-concurrency/--no-cloudwatch-concurrency",
    default=False,
    help="Fetch and plot ExecutionEnvironmentConcurrency (Average + SampleCount) for LMI during each scenario window.",
)
@click.option(
    "--cloudwatch-period-seconds",
    type=int,
    default=300,
    show_default=True,
    help="CloudWatch period (seconds) for ExecutionEnvironmentConcurrency queries.",
)
@click.option(
    "--cloudwatch-function-name",
    default=None,
    help="Override LMI FunctionName dimension (default: <stack>-lmi).",
)
@click.option(
    "--cloudwatch-resource",
    default=None,
    help="Override LMI Resource dimension (default: <function>:$LATEST.PUBLISHED).",
)
@click.option(
    "--cloudwatch-capacity-provider-name",
    default=None,
    help="Override CapacityProviderName dimension (default: <stack>-cp-2-c8xlarge).",
)
@click.option(
    "--refresh-cloudwatch",
    is_flag=True,
    default=False,
    help="Re-fetch CloudWatch concurrency data even if cached CSVs exist (useful with --skip-test).",
)
def main(
    stack_name: str,
    region: str | None,
    duration: str,
    stage_targets: str | None,
    stages_json: str | None,
    scenario: str,
    cpu_break_hash_loops: int,
    steady_io_delay_ms: int,
    bursty_io_delay_ms: int,
    mixed_delay_ms: int,
    mixed_hash_loops: int,
    output_dir: Path,
    skip_test: bool,
    wait_for_scale_in: bool | None,
    capacity_provider_arn: str | None,
    scale_in_check_seconds: int,
    scale_in_max_minutes: int,
    cloudwatch_concurrency: bool,
    cloudwatch_period_seconds: int,
    cloudwatch_function_name: str | None,
    cloudwatch_resource: str | None,
    cloudwatch_capacity_provider_name: str | None,
    refresh_cloudwatch: bool,
) -> None:
    """Run load test benchmarks comparing standard vs LMI Lambda endpoints."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if steady_io_delay_ms < 0 or steady_io_delay_ms > MAX_DELAY_MS:
        raise SystemExit(f"--steady-io-delay-ms must be between 0 and {MAX_DELAY_MS}")
    if bursty_io_delay_ms < 0 or bursty_io_delay_ms > MAX_DELAY_MS:
        raise SystemExit(f"--bursty-io-delay-ms must be between 0 and {MAX_DELAY_MS}")
    if mixed_delay_ms < 0 or mixed_delay_ms > MAX_DELAY_MS:
        raise SystemExit(f"--mixed-delay-ms must be between 0 and {MAX_DELAY_MS}")
    if cpu_break_hash_loops < 0 or cpu_break_hash_loops > MAX_HASH_LOOPS:
        raise SystemExit(f"--cpu-break-hash-loops must be between 0 and {MAX_HASH_LOOPS}")
    if mixed_hash_loops < 0 or mixed_hash_loops > MAX_HASH_LOOPS:
        raise SystemExit(f"--mixed-hash-loops must be between 0 and {MAX_HASH_LOOPS}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Get endpoints from CloudFormation (optional when --skip-test)
    url_512 = url_128 = url_lmi = None
    if skip_test:
        try:
            print(f"Fetching endpoints from stack: {stack_name}")
            outputs = get_stack_outputs(stack_name, region)
            url_512 = outputs.get("HelloWorldApi512Endpoint") or outputs.get("HelloWorldApiDefaultEndpoint")
            url_128 = outputs.get("HelloWorldApi128Endpoint")
            url_lmi = outputs.get("HelloWorldApiLMIEndpoint")
            if url_512 and url_128 and url_lmi:
                print(f"512  URL: {url_512}")
                print(f"128  URL: {url_128}")
                print(f"LMI  URL: {url_lmi}")
            else:
                print(
                    "Warning: incomplete stack outputs; continuing with CSV endpoint names only. "
                    f"Found: {list(outputs.keys())}"
                )
        except Exception as e:  # noqa: BLE001 - best-effort for chart regeneration
            print(f"Warning: failed to fetch stack outputs ({e}); continuing with CSV endpoint names only.")
    else:
        print(f"Fetching endpoints from stack: {stack_name}")
        outputs = get_stack_outputs(stack_name, region)
        url_512 = outputs.get("HelloWorldApi512Endpoint") or outputs.get("HelloWorldApiDefaultEndpoint")
        url_128 = outputs.get("HelloWorldApi128Endpoint")
        url_lmi = outputs.get("HelloWorldApiLMIEndpoint")

        if not url_512 or not url_128 or not url_lmi:
            raise SystemExit(f"Could not find endpoint outputs. Found: {list(outputs.keys())}")

        print(f"512  URL: {url_512}")
        if url_128:
            print(f"128  URL: {url_128}")
        print(f"LMI  URL: {url_lmi}")

    if cloudwatch_concurrency:
        cw_function_name = cloudwatch_function_name or f"{stack_name}-lmi"
        cw_resource = cloudwatch_resource or f"{cw_function_name}:$LATEST.PUBLISHED"
        cw_capacity_provider_name = cloudwatch_capacity_provider_name or f"{stack_name}-cp-2-c8xlarge"
    else:
        cw_function_name = cw_resource = cw_capacity_provider_name = None

    def timestamp_from_csv_path(path: Path, *, scenario_name: str) -> str | None:
        prefix = f"k6-{scenario_name}-"
        if path.stem.startswith(prefix):
            return path.stem.removeprefix(prefix)
        return None

    if stages_json:
        try:
            global_stages = json.loads(stages_json)
        except json.JSONDecodeError as e:
            raise SystemExit(f"Invalid --stages-json: {e}") from e
        if not isinstance(global_stages, list) or not global_stages:
            raise SystemExit("--stages-json must be a non-empty JSON array")
    elif stage_targets:
        try:
            stage_target_values = [int(x.strip()) for x in stage_targets.split(",") if x.strip()]
        except ValueError as e:
            raise SystemExit("--stage-targets must be a comma-separated list of integers") from e
        if not stage_target_values:
            raise SystemExit("--stage-targets must contain at least one target")
        global_stages = [{"duration": duration, "target": t} for t in stage_target_values]
    else:
        global_stages = None

    def validate_stages(stage_list: list[dict]) -> None:
        try:
            for stage in stage_list:
                parse_duration_to_seconds(stage["duration"])
                if int(stage["target"]) < 0:
                    raise ValueError("target must be non-negative")
        except (KeyError, TypeError, ValueError) as e:
            raise SystemExit(f"Invalid stages configuration: {e}") from e

    if global_stages is not None:
        validate_stages(global_stages)

    def latest_csv_for(scenario_name: str) -> Path:
        csv_files = sorted(output_dir.glob(f"k6-{scenario_name}-*.csv"), reverse=True)
        if not csv_files:
            raise SystemExit(f"No CSV found for scenario '{scenario_name}'. Run without --skip-test first.")
        return csv_files[0]

    scenarios = ["bursty-io", "steady-io", "cpu-break", "mixed"] if scenario == "all" else [scenario]
    wait_between_scenarios = (wait_for_scale_in if wait_for_scale_in is not None else scenario == "all") and not skip_test
    concurrency_data: dict[str, tuple[str, pd.DataFrame]] = {}
    combined_timestamp: str | None = None
    capacity_provider_arn_resolved = None
    baseline_instance_count: int | None = None
    if wait_between_scenarios:
        capacity_provider_arn_resolved = resolve_capacity_provider_arn(
            region=region,
            stack_name=stack_name,
            capacity_provider_arn=capacity_provider_arn,
        )
        if capacity_provider_arn_resolved is None:
            print(
                "Warning: --wait-for-scale-in is enabled but region is not set and capacity provider ARN can't be derived. "
                "Pass --region or --capacity-provider-arn to enable scale-in waiting."
            )
            wait_between_scenarios = False
        else:
            try:
                baseline_instance_count = len(
                    list_capacity_provider_instance_ids(region=region, capacity_provider_arn=capacity_provider_arn_resolved)
                )
                print(
                    f"Capacity provider baseline: {baseline_instance_count} instances "
                    f"(tag aws:lambda:capacity-provider={capacity_provider_arn_resolved})"
                )
            except Exception as e:  # noqa: BLE001 - best-effort orchestration
                print(f"Warning: failed to query EC2 for capacity provider instances ({e}); disabling scale-in waiting.")
                wait_between_scenarios = False
                capacity_provider_arn_resolved = None
                baseline_instance_count = None
    for scenario_name in scenarios:
        scenario_display = {
            "bursty-io": "Bursty I/O",
            "steady-io": "Steady I/O",
            "cpu-break": "CPU Break-point",
            "mixed": "Mixed I/O + CPU",
        }.get(scenario_name, scenario_name)
        endpoint_targets = [
            {"name": "128", "url": url_128},
            {"name": "512", "url": url_512},
            {"name": "LMI", "url": url_lmi},
        ]

        if scenario_name == "bursty-io":
            delay_ms, hash_loops = bursty_io_delay_ms, 0
            title = f"Lambda Load Test (Bursty I/O): 128 vs 512 vs LMI (delay_ms={delay_ms})"
            default_stages = [
                {"duration": "3m", "target": 200},
                {"duration": "6m", "target": 200},
                {"duration": "3m", "target": 500},
            ]
        elif scenario_name == "steady-io":
            delay_ms, hash_loops = steady_io_delay_ms, 0
            title = f"Lambda Load Test (Steady I/O): 128 vs 512 vs LMI (delay_ms={delay_ms})"
            default_stages = [
                {"duration": "3m", "target": 250},
                {"duration": "6m", "target": 250},
                {"duration": "3m", "target": 250},
            ]
        elif scenario_name == "cpu-break":
            delay_ms, hash_loops = 0, cpu_break_hash_loops
            title = f"Lambda Load Test (CPU Break-point): 128 vs 512 vs LMI (hash_loops={hash_loops})"
            default_stages = [
                {"duration": "3m", "target": 30},
                {"duration": "3m", "target": 60},
                {"duration": "3m", "target": 90},
                {"duration": "3m", "target": 120},
            ]
        elif scenario_name == "mixed":
            delay_ms, hash_loops = mixed_delay_ms, mixed_hash_loops
            title = (
                "Lambda Load Test (Mixed I/O + CPU): 128 vs 512 vs LMI "
                f"(delay_ms={delay_ms}, hash_loops={hash_loops})"
            )
            default_stages = [
                {"duration": "3m", "target": 150},
                {"duration": "6m", "target": 150},
                {"duration": "3m", "target": 300},
            ]
        else:
            raise SystemExit(f"Unknown scenario: {scenario_name}")

        mode = "per_endpoint"
        stages = global_stages if global_stages is not None else default_stages
        validate_stages(stages)

        endpoints = [t["name"] for t in endpoint_targets]
        csv_path = output_dir / f"k6-{scenario_name}-{timestamp}.csv"
        chart_path = output_dir / f"benchmark-{scenario_name}-{timestamp}.png"
        concurrency_chart_path = output_dir / f"cloudwatch-concurrency-{scenario_name}-{timestamp}.png"
        concurrency_csv_path = output_dir / f"cloudwatch-concurrency-{scenario_name}-{timestamp}.csv"

        print(f"\n{'=' * 70}")
        print(f"SCENARIO: {scenario_display} ({scenario_name})")
        print("=" * 70)

        if skip_test:
            csv_path = latest_csv_for(scenario_name)
            existing_timestamp = timestamp_from_csv_path(csv_path, scenario_name=scenario_name)
            if existing_timestamp:
                chart_path = output_dir / f"benchmark-{scenario_name}-{existing_timestamp}.png"
                concurrency_chart_path = output_dir / f"cloudwatch-concurrency-{scenario_name}-{existing_timestamp}.png"
                concurrency_csv_path = output_dir / f"cloudwatch-concurrency-{scenario_name}-{existing_timestamp}.csv"
                if combined_timestamp is None:
                    combined_timestamp = existing_timestamp
            print(f"Using existing CSV: {csv_path}")
        else:
            if combined_timestamp is None:
                combined_timestamp = timestamp
            print(f"RUNNING TEST: stages={stages}")
            run_k6_test(
                endpoint_targets,
                csv_path,
                stages=stages,
                mode=mode,
                delay_ms=delay_ms,
                hash_loops=hash_loops,
            )

        print(f"\nParsing results from: {csv_path}")
        raw_df = load_k6_csv(csv_path)
        latency_df = parse_k6_latencies(raw_df)
        if latency_df.empty:
            raise SystemExit("No data found in CSV.")

        if skip_test:
            stage_total_seconds = sum(parse_duration_to_seconds(s["duration"]) for s in stages)
            data_total_seconds = (latency_df["timestamp"].max() - latency_df["timestamp"].min()).total_seconds()
            if abs(stage_total_seconds - data_total_seconds) > max(30.0, data_total_seconds * 0.10):
                print(
                    "Warning: stage configuration does not match CSV duration "
                    f"(stages={stage_total_seconds:.0f}s vs csv≈{data_total_seconds:.0f}s). "
                    "Pass the original --duration/--stage-targets (or --stages-json) to get accurate phase markers."
                )

        stats = calculate_stats(latency_df)
        error_stats = calculate_error_stats(raw_df)
        if not error_stats.empty:
            stats = stats.join(
                error_stats[["errors_4xx", "errors_5xx", "errors_429", "error_rate"]],
                how="left",
            ).fillna(0)
        if len(endpoints) == 2:
            print_comparison_table(stats, baseline_name=endpoints[0], variant_name=endpoints[1])
        else:
            print_endpoint_table(stats, endpoints=endpoints)
        plot_results(
            latency_df,
            stats,
            chart_path,
            stages,
            endpoints,
            title,
        )

        if cloudwatch_concurrency and cw_function_name and cw_resource and cw_capacity_provider_name:
            try:
                cw_df = None
                if skip_test and concurrency_csv_path.exists() and not refresh_cloudwatch:
                    cw_df = load_execution_environment_concurrency_csv(concurrency_csv_path)
                if cw_df is None or cw_df.empty:
                    start_time = latency_df["timestamp"].min().to_pydatetime().replace(tzinfo=timezone.utc)
                    end_time = latency_df["timestamp"].max().to_pydatetime().replace(tzinfo=timezone.utc) + timedelta(
                        seconds=cloudwatch_period_seconds
                    )
                    cw_df = fetch_execution_environment_concurrency(
                        region=region,
                        function_name=cw_function_name,
                        resource=cw_resource,
                        capacity_provider_name=cw_capacity_provider_name,
                        start_time=start_time,
                        end_time=end_time,
                        period_seconds=cloudwatch_period_seconds,
                    )
                    save_execution_environment_concurrency_csv(cw_df, concurrency_csv_path)
                if scenario == "all":
                    concurrency_data[scenario_name] = (f"{scenario_display} – LMI Concurrency", cw_df)
                else:
                    plot_execution_environment_concurrency(
                        cw_df,
                        output_path=concurrency_chart_path,
                        title=f"{scenario_display} – LMI Concurrency",
                        stages=stages,
                    )
            except Exception as e:  # noqa: BLE001 - best-effort visualization
                print(f"Warning: failed to fetch/plot CloudWatch concurrency ({e}).")

        if wait_between_scenarios and scenario_name != scenarios[-1]:
            assert capacity_provider_arn_resolved is not None
            assert baseline_instance_count is not None
            wait_for_capacity_provider_scale_in(
                region=region,
                capacity_provider_arn=capacity_provider_arn_resolved,
                baseline_count=baseline_instance_count,
                check_interval_seconds=scale_in_check_seconds,
                max_wait_minutes=scale_in_max_minutes,
            )

    if cloudwatch_concurrency and scenario == "all":
        ordered = ["bursty-io", "steady-io", "cpu-break", "mixed"]
        chart_entries = [concurrency_data.get(name) for name in ordered]
        if all(chart_entries):
            combined_name = combined_timestamp or timestamp
            combined_path = output_dir / f"cloudwatch-concurrency-all-{combined_name}.png"
            plot_execution_environment_concurrency_grid(
                chart_entries,
                output_path=combined_path,
                stages=stages,
            )
        else:
            missing = [name for name, entry in zip(ordered, chart_entries, strict=False) if not entry]
            if missing:
                print(f"Warning: missing CloudWatch charts for {', '.join(missing)}; skipping combined output.")

    print(f"\n{'=' * 70}")
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
