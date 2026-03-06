#!/usr/bin/env python3
"""
analyze_focus_results.py

Read a focus-score CSV produced by focus_scores.py and generate summary tables,
condition plots, and focus-tick analyses for comparing imaging conditions.

This script is intentionally tolerant of different CSV schemas. It will:
- use manual_aperture / manual_flash_power when present
- use manual_focus when present for focus-tick analysis
- derive polygon summary stats from poly1_score..poly4_score if needed
- derive a center-region focus score from user-selected center polygons
- skip plots for metrics whose source columns are missing

Example:
    python analyze_focus_results.py \
        --csv focus_scores.csv \
        --outdir focus_analysis \
        --metric tenengrad
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METRICS = [
    "avg_score",
    "polygon_score_min",
    "polygon_score_std",
    "polygon_score_range",
    "full_image_score",
    "full_mean_intensity",
    "full_median_intensity",
    "full_std_intensity",
    "full_dynamic_range",
    "full_shadow_clip_pct",
    "full_highlight_clip_pct",
    "full_rms_contrast",
]

FOCUS_PRIORITY_METRICS = [
    "center_focus_score",
    "avg_score",
    "polygon_score_min",
    "polygon_score_std",
    "polygon_score_max",
    "polygon_score_range",
]

POLY_SCORE_COLS = ["poly1_score", "poly2_score", "poly3_score", "poly4_score"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate focus/lighting comparison plots from a focus score CSV."
    )
    parser.add_argument("--csv", required=True, type=Path, help="Input focus score CSV.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("focus_analysis"),
        help="Directory for plots and summary CSVs.",
    )
    parser.add_argument(
        "--metric",
        dest="focus_metric",
        default=None,
        help="Optional filter on the CSV 'metric' column, e.g. tenengrad.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Optional explicit list of value columns to plot. Defaults to common columns.",
    )
    parser.add_argument(
        "--min-replicates",
        type=int,
        default=1,
        help="Minimum rows required in an aperture x flash or focus group to include it in summaries.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved figures.",
    )
    parser.add_argument(
        "--center-polygons",
        nargs="+",
        type=int,
        default=[2, 3],
        help=(
            "Polygon numbers to treat as the plant/center region for focus selection. "
            "Default: 2 3"
        ),
    )
    return parser.parse_args()


_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+")
_FLASH_FRACTION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$")


def parse_aperture_value(value: object) -> tuple[float, str]:
    """Return sortable numeric aperture plus display label."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return (math.nan, "")
    text = str(value).strip()
    if not text:
        return (math.nan, "")
    text = text.lower().replace("f/", "").replace("f", "")
    match = _FLOAT_RE.search(text)
    if not match:
        return (math.nan, str(value).strip())
    num = float(match.group())
    label = f"f/{num:g}"
    return (num, label)


def parse_flash_value(value: object) -> tuple[float, str]:
    """Return sortable numeric flash value plus display label."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return (math.nan, "")
    text = str(value).strip()
    if not text:
        return (math.nan, "")

    frac = _FLASH_FRACTION_RE.match(text)
    if frac:
        num = float(frac.group(1)) / float(frac.group(2))
        return (num, f"{frac.group(1)}/{frac.group(2)}")

    match = _FLOAT_RE.search(text)
    if not match:
        return (math.nan, text)
    num = float(match.group())
    if 0 < num <= 1:
        label = f"{num:g}"
    else:
        label = str(value).strip()
    return (num, label)


def parse_focus_value(value: object) -> tuple[float, str]:
    """Return sortable numeric focus tick plus display label."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return (math.nan, "")
    text = str(value).strip()
    if not text:
        return (math.nan, "")
    match = _FLOAT_RE.search(text)
    if not match:
        return (math.nan, text)
    num = float(match.group())
    if num.is_integer():
        label = str(int(num))
    else:
        label = f"{num:g}"
    return (num, label)


NUMERICISH_COLUMNS = {
    "avg_score",
    "center_focus_score",
    "full_image_score",
    "polygon_score_min",
    "polygon_score_max",
    "polygon_score_std",
    "polygon_score_range",
    "full_mean_intensity",
    "full_median_intensity",
    "full_std_intensity",
    "full_min_intensity",
    "full_max_intensity",
    "full_p1_intensity",
    "full_p99_intensity",
    "full_dynamic_range",
    "full_shadow_clip_pct",
    "full_highlight_clip_pct",
    "full_rms_contrast",
    "camera_height_cm",
    "focus_card_height_agl",
    "manual_focus",
    *POLY_SCORE_COLS,
}


def load_data(csv_path: Path, center_polygons: Optional[Iterable[int]] = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    for col in df.columns:
        if col in NUMERICISH_COLUMNS or col.startswith("poly") or col.endswith("_pct"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "manual_aperture" not in df.columns:
        df["manual_aperture"] = np.nan
    if "manual_flash_power" not in df.columns:
        df["manual_flash_power"] = np.nan
    if "manual_focus" not in df.columns:
        df["manual_focus"] = np.nan

    existing_poly_cols = [c for c in POLY_SCORE_COLS if c in df.columns]
    if existing_poly_cols:
        poly_frame = df[existing_poly_cols]
        if "polygon_score_min" not in df.columns:
            df["polygon_score_min"] = poly_frame.min(axis=1, skipna=True)
        if "polygon_score_max" not in df.columns:
            df["polygon_score_max"] = poly_frame.max(axis=1, skipna=True)
        if "polygon_score_std" not in df.columns:
            df["polygon_score_std"] = poly_frame.std(axis=1, skipna=True)
        if "polygon_score_range" not in df.columns:
            df["polygon_score_range"] = df["polygon_score_max"] - df["polygon_score_min"]

    center_polygons = list(center_polygons or [2, 3])
    center_cols = [f"poly{i}_score" for i in center_polygons if f"poly{i}_score" in df.columns]
    if center_cols:
        df["center_focus_score"] = df[center_cols].mean(axis=1, skipna=True)
    df.attrs["center_cols"] = center_cols

    ap_parsed = df["manual_aperture"].apply(parse_aperture_value)
    df["aperture_value"] = ap_parsed.apply(lambda x: x[0])
    df["aperture_label"] = ap_parsed.apply(lambda x: x[1])

    fl_parsed = df["manual_flash_power"].apply(parse_flash_value)
    df["flash_value"] = fl_parsed.apply(lambda x: x[0])
    df["flash_label"] = fl_parsed.apply(lambda x: x[1])

    focus_parsed = df["manual_focus"].apply(parse_focus_value)
    df["focus_value"] = focus_parsed.apply(lambda x: x[0])
    df["focus_label"] = focus_parsed.apply(lambda x: x[1])

    return df


def ensure_output_dirs(base: Path) -> dict[str, Path]:
    paths = {
        "base": base,
        "summary": base / "summary_csv",
        "line_aperture": base / "plots_by_aperture",
        "line_flash": base / "plots_by_flash",
        "heatmaps": base / "heatmaps",
        "focus_summary": base / "focus_tick_summary_csv",
        "focus_plots": base / "plots_by_focus",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def available_metric_columns(df: pd.DataFrame, requested: Optional[Iterable[str]]) -> list[str]:
    cols = list(requested) if requested else DEFAULT_METRICS
    return [c for c in cols if c in df.columns]


def build_condition_summary(df: pd.DataFrame, metric_col: str, min_replicates: int) -> pd.DataFrame:
    keep = df[["aperture_label", "aperture_value", "flash_label", "flash_value", metric_col]].copy()
    keep = keep.dropna(subset=[metric_col])
    keep = keep[(keep["aperture_label"] != "") & (keep["flash_label"] != "")]

    if keep.empty:
        return keep

    grouped = (
        keep.groupby(["aperture_label", "aperture_value", "flash_label", "flash_value"], dropna=False)[metric_col]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "count": "n",
                "mean": metric_col + "_mean",
                "std": metric_col + "_std",
                "min": metric_col + "_min",
                "max": metric_col + "_max",
            }
        )
    )
    grouped = grouped[grouped["n"] >= min_replicates].copy()
    grouped = grouped.sort_values(["aperture_value", "flash_value", "aperture_label", "flash_label"])
    return grouped


def build_focus_summary(df: pd.DataFrame, min_replicates: int) -> pd.DataFrame:
    if "focus_label" not in df.columns or "focus_value" not in df.columns:
        return pd.DataFrame()

    summary_metrics = [col for col in FOCUS_PRIORITY_METRICS if col in df.columns]
    if not summary_metrics:
        return pd.DataFrame()

    keep_cols = ["focus_label", "focus_value"] + summary_metrics
    poly_cols = [c for c in POLY_SCORE_COLS if c in df.columns]
    keep_cols.extend(poly_cols)
    keep = df[keep_cols].copy()
    keep = keep[keep["focus_label"] != ""]
    if keep.empty:
        return keep

    grouped = keep.groupby(["focus_label", "focus_value"], dropna=False)
    pieces: list[pd.DataFrame] = []
    for metric in summary_metrics + poly_cols:
        agg = (
            grouped[metric]
            .agg(["count", "mean", "std", "min", "max"])
            .reset_index()
            .rename(
                columns={
                    "count": f"{metric}_n",
                    "mean": f"{metric}_mean",
                    "std": f"{metric}_std",
                    "min": f"{metric}_min",
                    "max": f"{metric}_max",
                }
            )
        )
        pieces.append(agg)

    summary = pieces[0]
    for piece in pieces[1:]:
        summary = summary.merge(piece, on=["focus_label", "focus_value"], how="outer")

    summary["n"] = summary[[c for c in summary.columns if c.endswith("_n")]].max(axis=1)
    summary = summary[summary["n"] >= min_replicates].copy()

    if {"center_focus_score_mean", "center_focus_score_std"}.issubset(summary.columns):
        summary["center_focus_stability_score"] = (
            summary["center_focus_score_mean"] - summary["center_focus_score_std"].fillna(0)
        )

    summary = summary.sort_values(["focus_value", "focus_label"])
    return summary


def save_summary_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def plot_metric_vs_aperture(summary: pd.DataFrame, metric_col: str, outpath: Path, dpi: int) -> None:
    if summary.empty:
        return

    value_col = metric_col + "_mean"
    plt.figure(figsize=(8, 5.5))
    for flash_label, group in summary.groupby("flash_label", dropna=False):
        group = group.sort_values("aperture_value")
        plt.plot(group["aperture_value"], group[value_col], marker="o", label=f"Flash {flash_label}")

    plt.xlabel("Aperture (f-number)")
    plt.ylabel(metric_col)
    plt.title(f"{metric_col} vs aperture")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Manual flash", fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_metric_vs_flash(summary: pd.DataFrame, metric_col: str, outpath: Path, dpi: int) -> None:
    if summary.empty:
        return

    value_col = metric_col + "_mean"
    plt.figure(figsize=(8, 5.5))
    for ap_label, group in summary.groupby("aperture_label", dropna=False):
        group = group.sort_values("flash_value")
        x = np.arange(len(group))
        plt.plot(x, group[value_col], marker="o", label=ap_label)
        plt.xticks(x, group["flash_label"], rotation=45, ha="right")

    plt.xlabel("Flash power")
    plt.ylabel(metric_col)
    plt.title(f"{metric_col} vs flash power")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Manual aperture", fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_heatmap(summary: pd.DataFrame, metric_col: str, outpath: Path, dpi: int) -> None:
    if summary.empty:
        return

    value_col = metric_col + "_mean"
    heat = summary.pivot_table(
        index="aperture_label",
        columns="flash_label",
        values=value_col,
        aggfunc="mean",
    )
    if heat.empty:
        return

    ap_order = (
        summary[["aperture_label", "aperture_value"]]
        .drop_duplicates()
        .sort_values(["aperture_value", "aperture_label"])["aperture_label"]
        .tolist()
    )
    fl_order = (
        summary[["flash_label", "flash_value"]]
        .drop_duplicates()
        .sort_values(["flash_value", "flash_label"])["flash_label"]
        .tolist()
    )
    heat = heat.reindex(index=ap_order, columns=fl_order)

    fig, ax = plt.subplots(figsize=(max(6, len(fl_order) * 1.0), max(4, len(ap_order) * 0.7)))
    im = ax.imshow(heat.values, aspect="auto")
    ax.set_xticks(np.arange(len(fl_order)))
    ax.set_xticklabels(fl_order, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(ap_order)))
    ax.set_yticklabels(ap_order)
    ax.set_xlabel("Flash power")
    ax.set_ylabel("Aperture")
    ax.set_title(f"{metric_col} heatmap")

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.iat[i, j]
            text = "" if pd.isna(val) else f"{val:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label=metric_col)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_focus_metric(summary: pd.DataFrame, mean_col: str, ylabel: str, title: str, outpath: Path, dpi: int) -> None:
    if summary.empty or mean_col not in summary.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(summary["focus_value"], summary[mean_col], marker="o")
    ax.set_xlabel("Manual focus tick")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_focus_polygons(summary: pd.DataFrame, outpath: Path, dpi: int) -> None:
    poly_mean_cols = [f"{col}_mean" for col in POLY_SCORE_COLS if f"{col}_mean" in summary.columns]
    if summary.empty or not poly_mean_cols:
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    for mean_col in poly_mean_cols:
        label = mean_col.replace("_mean", "")
        ax.plot(summary["focus_value"], summary[mean_col], marker="o", label=label)

    ax.set_xlabel("Manual focus tick")
    ax.set_ylabel("Polygon focus score")
    ax.set_title("Manual focus vs each polygon score")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_focus_recommendations(focus_summary: pd.DataFrame) -> pd.DataFrame:
    if focus_summary.empty:
        return pd.DataFrame()

    rows = []
    for metric_col, goal in [
        ("center_focus_score_mean", "max"),
        ("center_focus_stability_score", "max"),
        ("avg_score_mean", "max"),
        ("polygon_score_min_mean", "max"),
        ("polygon_score_std_mean", "min"),
    ]:
        if metric_col not in focus_summary.columns:
            continue
        ordered = focus_summary.sort_values(metric_col, ascending=(goal == "min"))
        best = ordered.iloc[0]
        rows.append(
            {
                "selection_rule": metric_col,
                "optimization": goal,
                "recommended_manual_focus": best["focus_label"],
                "focus_value": best["focus_value"],
                "metric_value": best[metric_col],
                "n": int(best["n"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    df = load_data(args.csv, center_polygons=args.center_polygons)

    if args.focus_metric and "metric" in df.columns:
        df = df[df["metric"].astype(str).str.lower() == args.focus_metric.lower()].copy()

    if df.empty:
        raise SystemExit("No rows left after filtering. Check --metric or input CSV.")

    outdirs = ensure_output_dirs(args.outdir)

    metric_columns = available_metric_columns(df, args.metrics)
    if not metric_columns:
        raise SystemExit("No requested metric columns were found in the CSV.")

    filtered_csv = outdirs["base"] / "filtered_rows.csv"
    df.to_csv(filtered_csv, index=False)

    overall_summary_rows = []
    for metric_col in metric_columns:
        summary = build_condition_summary(df, metric_col, args.min_replicates)
        if summary.empty:
            continue

        summary_path = outdirs["summary"] / f"{metric_col}_by_aperture_flash.csv"
        save_summary_csv(summary, summary_path)

        plot_metric_vs_aperture(
            summary,
            metric_col,
            outdirs["line_aperture"] / f"{metric_col}_vs_aperture.png",
            args.dpi,
        )
        plot_metric_vs_flash(
            summary,
            metric_col,
            outdirs["line_flash"] / f"{metric_col}_vs_flash.png",
            args.dpi,
        )
        plot_heatmap(
            summary,
            metric_col,
            outdirs["heatmaps"] / f"{metric_col}_heatmap.png",
            args.dpi,
        )

        value_col = metric_col + "_mean"
        best_row = summary.sort_values(value_col, ascending=False).iloc[0]
        overall_summary_rows.append(
            {
                "metric_column": metric_col,
                "best_aperture": best_row["aperture_label"],
                "best_flash": best_row["flash_label"],
                "best_mean_value": best_row[value_col],
                "n": int(best_row["n"]),
                "std": best_row.get(metric_col + "_std", np.nan),
            }
        )

    if overall_summary_rows:
        pd.DataFrame(overall_summary_rows).to_csv(
            outdirs["base"] / "best_conditions_overview.csv", index=False
        )

    focus_summary = build_focus_summary(df, args.min_replicates)
    if not focus_summary.empty:
        focus_summary_path = outdirs["focus_summary"] / "manual_focus_summary.csv"
        save_summary_csv(focus_summary, focus_summary_path)

        center_cols = df.attrs.get("center_cols", [])
        center_label = " + ".join(center_cols) if center_cols else "selected center polygon(s)"
        focus_plot_specs = [
            (
                "center_focus_score_mean",
                "Center-region focus score",
                f"Manual focus vs center-region sharpness ({center_label})",
                "focus_vs_center_score.png",
            ),
            (
                "center_focus_stability_score",
                "Center mean minus std",
                f"Manual focus vs center-region stability ({center_label})",
                "focus_vs_center_stability.png",
            ),
            (
                "avg_score_mean",
                "Average focus score",
                "Manual focus vs average sharpness",
                "focus_vs_avg_score.png",
            ),
            (
                "polygon_score_min_mean",
                "Weakest polygon score",
                "Manual focus vs weakest polygon",
                "focus_vs_min_polygon.png",
            ),
            (
                "polygon_score_std_mean",
                "Polygon score std dev",
                "Manual focus vs focus uniformity",
                "focus_vs_std.png",
            ),
            (
                "polygon_score_max_mean",
                "Best polygon score",
                "Manual focus vs strongest polygon",
                "focus_vs_max_polygon.png",
            ),
        ]
        for col, ylabel, title, filename in focus_plot_specs:
            plot_focus_metric(
                focus_summary,
                col,
                ylabel,
                title,
                outdirs["focus_plots"] / filename,
                args.dpi,
            )

        plot_focus_polygons(
            focus_summary,
            outdirs["focus_plots"] / "focus_vs_each_polygon.png",
            args.dpi,
        )

        focus_recommendations = build_focus_recommendations(focus_summary)
        if not focus_recommendations.empty:
            focus_recommendations.to_csv(
                outdirs["base"] / "best_manual_focus_overview.csv",
                index=False,
            )

    readme = outdirs["base"] / "README.txt"
    center_polys_text = " ".join(str(x) for x in args.center_polygons)
    readme.write_text(
        "Generated files:\n"
        "- filtered_rows.csv: row-level data after optional metric filtering\n"
        "- summary_csv/: grouped means/std/counts by manual aperture and flash\n"
        "- plots_by_aperture/: line plots of metric vs aperture, one line per flash setting\n"
        "- plots_by_flash/: line plots of metric vs flash, one line per aperture setting\n"
        "- heatmaps/: aperture x flash heatmaps\n"
        "- best_conditions_overview.csv: top aperture/flash condition per metric column\n"
        "- focus_tick_summary_csv/manual_focus_summary.csv: grouped results by manual focus tick\n"
        f"- center focus selection uses polygon(s): {center_polys_text}\n"
        "- plots_by_focus/: manual focus plots including center-region, avg, min, std, and each polygon\n"
        "- best_manual_focus_overview.csv: recommended manual focus ticks using several rules\n"
    )

    print(f"Saved analysis outputs to: {args.outdir}")


if __name__ == "__main__":
    main()
