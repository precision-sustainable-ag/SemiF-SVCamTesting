import argparse
import ast
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
from pathlib import Path
from skimage.measure import shannon_entropy


WINDOW_HEIGHT = 800
WINDOW_WIDTH = 1200


def draw_bboxes_on_image(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    scale = min(WINDOW_WIDTH / image.shape[1], WINDOW_HEIGHT / image.shape[0])
    resized = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
    return interactive_roi_selector(resized, scale)


def interactive_roi_selector(image, scale_factor, window_name="Select boxes"):
    boxes, current_box = [], []
    clone = image.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_box
        if event == cv2.EVENT_LBUTTONDOWN and len(current_box) < 4:
            current_box.append((x, y))
            cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(clone, f"{x},{y}", (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.imshow(window_name, clone)

            if len(current_box) == 4:
                xs, ys = [p[0] for p in current_box], [p[1] for p in current_box]
                cv2.rectangle(clone, (min(xs), min(ys)), (max(xs), max(ys)), (0, 0, 255), 2)
                boxes.append([
                    int(min(xs) / scale_factor), int(min(ys) / scale_factor),
                    int(max(xs) / scale_factor), int(max(ys) / scale_factor)
                ])
                current_box.clear()
                print(f"📦 Added box #{len(boxes)}. Press 'n' for another or Enter to finish.")

    print("\n🖱️  Click 4 points to define a bounding box.")
    print("➡️ Press [n] to start another box, [Enter] to finish, or [Esc] to cancel all.\n")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.resizeWindow(window_name, WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.imshow(window_name, clone)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13: break  # Enter
        elif key == 27:
            print("❌ Cancelled selection.")
            boxes.clear()
            break

    cv2.destroyWindow(window_name)
    return boxes


class FocusMetricCalculator:
    @staticmethod
    def compute(gray_image):
        return {
            "laplacian_var": cv2.Laplacian(gray_image, cv2.CV_64F).var(),
            "tenengrad": np.mean((cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3) ** 2 +
                                  cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3) ** 2)),
            "sobel_mag_mean": np.mean(np.sqrt(cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3) ** 2 +
                                              cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3) ** 2)),
            "entropy": shannon_entropy(gray_image),
            "fft_energy": np.mean(np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray_image)))))
        }


class FocusEvaluatorApp:
    def __init__(self, image_dir, input_csv, extension=".png", reset_bboxes=False):
        self.image_dir = Path(image_dir)
        self.input_csv = Path(input_csv)
        self.df = pd.read_csv(self.input_csv) if self.input_csv.exists() else pd.DataFrame()
        self.extension = extension
        self.reset_bboxes = reset_bboxes

        self.bbox_cols = sorted([col for col in self.df.columns if re.match(r"^bbox\d+$", col)])

        if self.reset_bboxes:
            bbox_related = [col for col in self.df.columns if col.startswith("bbox")]
            self.df.drop(columns=bbox_related, inplace=True, errors="ignore")
            self.bbox_cols = []

        self.metric_keys = list(FocusMetricCalculator.compute(np.ones((10, 10))).keys())

    def assign_or_draw_bboxes(self):
        grouped = self.df.groupby("z_height")

        for z_val, group in grouped:
            existing_bbox_cols = sorted([col for col in self.df.columns if re.match(r"^bbox\d+$", col)])
            has_bboxes = group[existing_bbox_cols].notna().any(axis=1) if existing_bbox_cols else False
            print(has_bboxes)

            if has_bboxes.any():
                source = group[has_bboxes].iloc[0]
                for i, row in group[~has_bboxes].iterrows():
                    for col in existing_bbox_cols:
                        self.df.at[i, col] = source[col]
            else:
                # Draw new bboxes
                for image_name in group["image_name"]:
                    image_path = self.image_dir / f"{image_name}{self.extension}"
                    if image_path.exists():
                        new_boxes = draw_bboxes_on_image(image_path)
                        for j, box in enumerate(new_boxes):
                            col = f"bbox{j+1}"
                            self.df[col] = self.df.get(col, pd.NA)
                            self.df.loc[self.df["z_height"] == z_val, col] = str(tuple(box))
                            if col not in self.bbox_cols:
                                self.bbox_cols.append(col)
                        break

    def process_metrics(self):
        new_rows = []
        for idx, row in self.df.iterrows():
            image_path = self.image_dir / f"{row['image_name']}{self.extension}"
            if not image_path.exists():
                print(f"❌ Image not found: {image_path}")
                continue

            already_computed = all(
                not pd.isna(row.get(f"{bbox_col}_{k}"))
                for bbox_col in self.bbox_cols
                for k in self.metric_keys
            )
            if already_computed:
                print(f"✅ Skipping already processed: {row['image_name']}")
                continue

            print(f"📷 Processing: {row['image_name']}")

            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            for bbox_col in self.bbox_cols:
                bbox_str = row.get(bbox_col)
                if pd.isna(bbox_str):
                    continue
                x1, y1, x2, y2 = ast.literal_eval(bbox_str)
                crop = img[y1:y2, x1:x2]
                metrics = FocusMetricCalculator.compute(crop)
                for k, v in metrics.items():
                    self.df.at[idx, f"{bbox_col}_{k}"] = v

    def run(self):
        self.assign_or_draw_bboxes()
        self.process_metrics()
        # Save to output CSV
        self.df.to_csv(self.input_csv, index=False)
        print(f"✅ Finished and saved to {self.input_csv}")
        return self.df

# Re-define the SharpnessPlotter class properly without markdown syntax issues


class SharpnessPlotter:
    def __init__(self, df: pd.DataFrame, input_dir: str):
        self.df = df
        self.output_dir = Path(input_dir).parent / "plots" / Path(input_dir).name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metric_prefixes = ["laplacian_var", "tenengrad", "sobel_mag_mean", "entropy", "fft_energy"]
        self.bbox_metrics = self._extract_bbox_metric_columns()

    def _extract_bbox_metric_columns(self):
        """
        Finds all bbox-related metric columns dynamically.
        Example: bbox1_laplacian_var, bbox2_entropy, etc.
        """
        metric_cols = [col for col in self.df.columns if re.match(r"bbox\d+_(\w+)", col)]
        bbox_groups = {}
        for col in metric_cols:
            match = re.match(r"(bbox\d+)_(\w+)", col)
            if match:
                bbox, metric = match.groups()
                if metric in self.metric_prefixes:
                    bbox_groups.setdefault(bbox, []).append(col)
        return bbox_groups

    def plot_metric_vs_focus(self, metric_name: str, by: str = "z_height"):
        """
        Plot the given metric (e.g., 'laplacian_var') vs focus, colored by bbox and styled by 'by' (e.g., z_height or f_number).
        """
        records = []
        for bbox, cols in self.bbox_metrics.items():
            col_name = f"{bbox}_{metric_name}"
            if col_name in self.df.columns:
                subset = self.df[[col_name, "focus", by]].dropna()
                for _, row in subset.iterrows():
                    records.append({
                        "Region": bbox,
                        "focus": row["focus"],
                        by: row[by],
                        "Sharpness": row[col_name]
                    })

        plot_df = pd.DataFrame(records)

        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=plot_df,
            x="focus",
            y="Sharpness",
            hue="Region",
            style=by,
            markers=True,
            dashes=False
        )
        plt.title(f"{metric_name.replace('_', ' ').title()} vs Focus")
        plt.xlabel("Focus Distance")
        plt.ylabel(f"{metric_name.replace('_', ' ').title()} (Sharpness)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{metric_name}_vs_focus.png")

    def plot_metric_vs_aperture(self, metric_name: str, by: str = "z_height"):
        """
        Plot the given metric (e.g., 'laplacian_var') vs aperture (f_number), colored by bbox and styled by 'by'.
        """
        records = []
        for bbox, cols in self.bbox_metrics.items():
            col_name = f"{bbox}_{metric_name}"
            if col_name in self.df.columns:
                subset = self.df[[col_name, "f_number", by]].dropna()
                for _, row in subset.iterrows():
                    records.append({
                        "Region": bbox,
                        "f_number": row["f_number"],
                        by: row[by],
                        "Sharpness": row[col_name]
                    })

        plot_df = pd.DataFrame(records)

        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=plot_df,
            x="f_number",
            y="Sharpness",
            hue="Region",
            style=by,
            markers=True,
            dashes=False
        )
        plt.title(f"{metric_name.replace('_', ' ').title()} vs Aperture")
        plt.xlabel("Aperture (f-number)")
        plt.ylabel(f"{metric_name.replace('_', ' ').title()} (Sharpness)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{metric_name}_vs_aperture.png")
    
    def plot(self):
        """
        Plot all metrics vs focus and aperture.
        """
        print("📊 Generating plots...")
        for metric in self.metric_prefixes:
            self.plot_metric_vs_focus(metric)
            self.plot_metric_vs_aperture(metric)
        print("✅ Plots generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth of field + focus metric pipeline.")
    parser.add_argument("image_dir", help="Directory of images (JPG)")
    parser.add_argument("input_csv", default="calibration.csv", help="Input CSV (can be empty)")
    parser.add_argument("--reset_bboxes", action="store_true", help="Clear existing bbox columns and start fresh")
    parser.add_argument("--ext", default=".png", help="Image file extension (.jpg or .png)")
    args = parser.parse_args()

    app = FocusEvaluatorApp(
        image_dir=args.image_dir,
        input_csv=args.input_csv,
        extension=args.ext,
        reset_bboxes=args.reset_bboxes
    )
    df = app.run()
    # remove any metrics that start with bbox1 or bbox2
    df = df.drop(columns=[col for col in df.columns if "bbox1_" in col or "bbox2_" in col], errors="ignore")

    plotter = SharpnessPlotter(df, args.image_dir)
    plotter.plot()
