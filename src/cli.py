import argparse
import time
import csv
import sys
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np

from src.config import Config
from src.utils.visualization import load_image, save_image, create_comparison_figure, visualize_depth_zones
from src.utils.metrics import evaluate_enhancement
from src.enhancement.baseline import baseline_enhance
from src.enhancement.dase_pipeline import dase_enhance
from src.depth.midas import load_midas_model, dummy_depth_map, segment_depth_zones

def enhance_image(
    image_path: str,
    method: str = 'dase',
    contrast_method: str = 'CLAHE',
    save_path: Optional[str] = None,
    depth_model_bundle=None
) -> np.ndarray:
    """Load and enhance a single underwater image."""
    image = load_image(image_path)
    print(f"[Pipeline] Image loaded: {image_path}  shape={image.shape}")

    if method == 'baseline':
        result = baseline_enhance(image, contrast_method=contrast_method)
    elif method == 'dase':
        result = dase_enhance(image, depth_model_bundle=depth_model_bundle)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'baseline' or 'dase'.")

    if save_path:
        save_image(result, save_path)
        print(f"[Pipeline] Saved: {save_path}")

    return result

def compare_methods(
    image_path: str,
    output_dir: Optional[str] = None,
    depth_model_bundle=None
) -> Dict:
    """Run both baseline and DASE on a single image and compare results."""
    print(f"\n{'='*60}")
    print(f" Comparing methods on: {Path(image_path).name}")
    print(f"{'='*60}")

    image = load_image(image_path)

    t0 = time.time()
    baseline_result = baseline_enhance(image)
    baseline_time = time.time() - t0
    print(f"[Compare] Baseline done in {baseline_time:.2f}s")

    t0 = time.time()
    dase_result = dase_enhance(image, depth_model_bundle=depth_model_bundle)
    dase_time = time.time() - t0
    print(f"[Compare] DASE done in {dase_time:.2f}s")

    baseline_metrics = evaluate_enhancement(image, baseline_result)
    dase_metrics     = evaluate_enhancement(image, dase_result)

    print("\n[Compare] Metrics:")
    print(f"  {'Metric':<14}  {'Baseline':>10}  {'DASE':>10}")
    print(f"  {'-'*14}  {'-'*10}  {'-'*10}")
    for key in ('UIQM', 'UCIQE', 'Entropy', 'Delta_UIQM'):
        b_val = baseline_metrics.get(key, float('nan'))
        d_val = dase_metrics.get(key, float('nan'))
        print(f"  {key:<14}  {b_val:>10.4f}  {d_val:>10.4f}")

    result = {
        'original':         image,
        'baseline':         baseline_result,
        'dase':             dase_result,
        'baseline_metrics': baseline_metrics,
        'dase_metrics':     dase_metrics,
        'baseline_time':    baseline_time,
        'dase_time':        dase_time,
    }

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem
        save_image(baseline_result, f"{output_dir}/{stem}_baseline.png")
        save_image(dase_result,     f"{output_dir}/{stem}_dase.png")
        create_comparison_figure(
            images={
                'Original': image,
                'Baseline': baseline_result,
                'DASE':     dase_result,
            },
            metrics={
                'Baseline': baseline_metrics,
                'DASE':     dase_metrics,
            },
            save_path=f"{output_dir}/{stem}_comparison.png"
        )

    return result

def process_single_image(
    input_path: str,
    output_dir: str,
    methods: List[str] = ['baseline', 'dase'],
    depth_model_bundle=None
) -> Dict[str, Dict[str, float]]:
    """Process one image with all requested methods."""
    image = load_image(input_path)
    stem  = Path(input_path).stem
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_metrics: Dict[str, Dict[str, float]] = {}
    result_images = {'Original': image}

    for method in methods:
        t0 = time.time()
        if method == 'baseline':
            result = baseline_enhance(image)
        elif method == 'dase':
            result = dase_enhance(image, depth_model_bundle=depth_model_bundle)
        else:
            print(f"[Batch] Unknown method '{method}', skipping.")
            continue

        elapsed = time.time() - t0
        out_path = f"{output_dir}/{stem}_{method}.png"
        save_image(result, out_path)

        metrics = evaluate_enhancement(image, result)
        metrics['time_s'] = elapsed
        all_metrics[method] = metrics
        result_images[method.upper()] = result
        print(f"[Batch] {stem} | {method} | UIQM={metrics['UIQM']:.4f} | {elapsed:.2f}s")

    create_comparison_figure(
        images=result_images,
        metrics={k.upper(): v for k, v in all_metrics.items()},
        save_path=f"{output_dir}/{stem}_comparison.png"
    )

    return all_metrics

def batch_process(
    input_dir: str,
    output_dir: str,
    methods: List[str] = ['baseline', 'dase'],
    save_metrics: bool = True
) -> None:
    """Process all images in a directory with all specified methods."""
    input_dir_p = Path(input_dir)
    if not input_dir_p.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    image_paths = [
        p for p in sorted(input_dir_p.iterdir())
        if p.suffix.lower() in Config.IMAGE_EXTENSIONS
    ]
    if not image_paths:
        print(f"[Batch] No images found in {input_dir}")
        return

    print(f"[Batch] Found {len(image_paths)} images. Methods: {methods}")

    depth_model_bundle = None
    if 'dase' in methods:
        try:
            depth_model_bundle = load_midas_model()
        except Exception as e:
            print(f"[Batch] MiDaS unavailable ({e}). Using depth fallback.")

    all_results: List[Dict] = []
    total_start = time.time()

    for img_path in image_paths:
        print(f"\n[Batch] Processing: {img_path.name}")
        try:
            metrics = process_single_image(
                str(img_path), output_dir, methods, depth_model_bundle
            )
            for method, m in metrics.items():
                row = {'file': img_path.name, 'method': method}
                row.update(m)
                all_results.append(row)
        except Exception as e:
            print(f"[Batch] Error on {img_path.name}: {e}")

    total_time = time.time() - total_start
    print(f"\n[Batch] Done. {len(image_paths)} images in {total_time:.1f}s")

    if save_metrics and all_results:
        csv_path = f"{output_dir}/metrics_summary.csv"
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        keys = list(all_results[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"[Batch] Metrics saved: {csv_path}")

def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DASE – Depth-Aware Scene-Adaptive Underwater Enhancement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli --image sample.jpg
  python -m src.cli --image sample.jpg --method dase --save results/out.png
  python -m src.cli --compare sample.jpg --output results/
  python -m src.cli --batch images/ --output results/
        """
    )
    parser.add_argument('--image',   type=str, help='Single image to enhance')
    parser.add_argument('--compare', type=str, help='Compare baseline vs DASE on image')
    parser.add_argument('--batch',   type=str, help='Directory of images for batch processing')
    parser.add_argument('--output',  type=str, default='results', help='Output directory')
    parser.add_argument('--save',    type=str, default=None,      help='Save path for single image output')
    parser.add_argument('--method',  type=str, default='dase',    choices=['baseline', 'dase'],
                        help='Enhancement method')
    parser.add_argument('--contrast', type=str, default='CLAHE',  choices=['HE', 'CLAHE', 'LA'],
                        help='Contrast method for baseline')
    return parser

def main():
    if len(sys.argv) > 1:
        parser = _build_cli()
        args   = parser.parse_args()

        if args.compare:
            compare_methods(args.compare, output_dir=args.output)
        elif args.batch:
            batch_process(args.batch, args.output)
        elif args.image:
            result = enhance_image(
                args.image,
                method=args.method,
                contrast_method=args.contrast,
                save_path=args.save or f"{args.output}/{Path(args.image).stem}_{args.method}.png"
            )
            metrics = evaluate_enhancement(load_image(args.image), result)
            print("\n[Result] Quality Metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
        else:
            parser.print_help()
        return

    print("="*60)
    print("  DASE – Demo Mode")
    print("="*60)
    print("No image path specified. Running a synthetic test…\n")

    h, w = 256, 384
    synthetic = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        depth_factor = y / h
        synthetic[y, :, 0] = int(40 * (1 - depth_factor))
        synthetic[y, :, 1] = int(100 + 60 * (1 - depth_factor))
        synthetic[y, :, 2] = int(120 + 80 * depth_factor)
    noise = np.random.randint(0, 20, synthetic.shape, dtype=np.uint8)
    synthetic = np.clip(synthetic.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    print("Input image shape:", synthetic.shape)

    print("\n[Demo] Running baseline enhancement …")
    baseline_result = baseline_enhance(synthetic, contrast_method='CLAHE')
    baseline_metrics = evaluate_enhancement(synthetic, baseline_result)
    print(f"  UIQM  : {baseline_metrics['UIQM']:.4f}")
    print(f"  UCIQE : {baseline_metrics['UCIQE']:.4f}")
    print(f"  Entropy: {baseline_metrics['Entropy']:.4f}")

    print("\n[Demo] Running DASE enhancement (depth fallback if MiDaS unavailable) …")
    dase_result = dase_enhance(synthetic)
    dase_metrics = evaluate_enhancement(synthetic, dase_result)
    print(f"  UIQM  : {dase_metrics['UIQM']:.4f}")
    print(f"  UCIQE : {dase_metrics['UCIQE']:.4f}")
    print(f"  Entropy: {dase_metrics['Entropy']:.4f}")

    print("\n[Demo] Generating comparison figure …")
    create_comparison_figure(
        images={
            'Original (Synthetic)': synthetic,
            'Baseline (CLAHE)':     baseline_result,
            'DASE':                 dase_result,
        },
        metrics={
            'Baseline (CLAHE)': baseline_metrics,
            'DASE':             dase_metrics,
        },
        save_path='demo_comparison.png'
    )

    print("\n[Demo] Generating depth zone visualization …")
    depth_map = dummy_depth_map(synthetic)
    visualize_depth_zones(synthetic, depth_map, segment_depth_zones, save_path='demo_depth_zones.png')

    print("\n[Demo] Complete! Saved: demo_comparison.png, demo_depth_zones.png")

if __name__ == "__main__":
    main()
