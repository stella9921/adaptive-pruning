import argparse
import csv
from pathlib import Path

from src.utils.visualization import _draw_bars


METRICS = ("abs_mean", "std", "zero_fraction", "q95", "q99")


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def find_pair(run_dir, epoch):
    metrics_dir = run_dir / "metrics"
    suffix = f"epoch-{epoch:03d}.csv"
    before = sorted(metrics_dir.glob(f"*activation_distribution_before*{suffix}"))
    after = sorted(metrics_dir.glob(f"*activation_distribution_after*{suffix}"))
    if not before or not after:
        raise FileNotFoundError(
            f"activation_distribution before/after CSV not found for epoch {epoch} "
            f"under {metrics_dir}"
        )
    return before[-1], after[-1]


def build_delta_rows(before_rows, after_rows):
    after_by_layer = {row["layer"]: row for row in after_rows}
    rows = []
    for before in before_rows:
        layer = before["layer"]
        after = after_by_layer.get(layer)
        if after is None:
            continue
        row = {
            "layer": layer,
            "stage": before.get("stage", ""),
        }
        for metric in METRICS:
            before_value = float(before[metric])
            after_value = float(after[metric])
            row[f"{metric}_before"] = before_value
            row[f"{metric}_after"] = after_value
            row[f"{metric}_delta"] = after_value - before_value
        rows.append(row)
    return rows


def save_delta_plots(rows, run_dir, run_id, epoch):
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    specs = [
        ("abs_mean_delta", "Activation Abs Mean Delta"),
        ("std_delta", "Activation Std Delta"),
        ("zero_fraction_delta", "Activation Zero Fraction Delta"),
        ("q95_delta", "Activation Q95 Delta"),
        ("q99_delta", "Activation Q99 Delta"),
    ]
    for key, title in specs:
        path = plots_dir / f"{run_id}__activation_{key}__epoch-{epoch:03d}.png"
        result = _draw_bars(str(path), title, rows, "layer", key)
        if result:
            saved.append(result)
    return saved


def infer_run_id(run_dir):
    return run_dir.name


def main():
    parser = argparse.ArgumentParser(
        description="Create activation before/after delta CSV and plots."
    )
    parser.add_argument("--run-dir", required=True, help="exp/runs/<run_id> path")
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_id = args.run_id or infer_run_id(run_dir)
    before_path, after_path = find_pair(run_dir, args.epoch)

    rows = build_delta_rows(read_rows(before_path), read_rows(after_path))
    output_csv = (
        run_dir / "metrics"
        / f"{run_id}__activation_distribution_delta__epoch-{args.epoch:03d}.csv"
    )
    write_rows(output_csv, rows)
    plot_paths = save_delta_plots(rows, run_dir, run_id, args.epoch)

    print(f"[Activation Delta] before={before_path}")
    print(f"[Activation Delta] after ={after_path}")
    print(f"[Activation Delta] csv   ={output_csv}")
    for path in plot_paths:
        print(f"[Activation Delta] plot  ={path}")


if __name__ == "__main__":
    main()
