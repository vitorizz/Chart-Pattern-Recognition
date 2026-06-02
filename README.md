# Chart Pattern Recognition: Multi-Timeframe Analysis with Regime Filtering

A computer-vision system that detects 20 candlestick chart-pattern types (Head & Shoulders, Cup & Handle, Double Top, triangles, wedges, etc.) from market chart images, with four post-detection layers designed to raise signal precision and reduce false alerts.

## Overview

Automated chart-pattern tools tend to share three weaknesses: high false-signal rates, alerts that only fire after a pattern has fully formed, and uniform treatment of detections regardless of market regime. This project trains a YOLO26 detector as a backbone and adds four lightweight layers on top to address each of those.

**Pipeline:** YOLO26s detector → confidence calibration → multi-timeframe confirmation → pattern-evolution tracking → regime-aware filtering.

## Approach

- **Detection backbone:** YOLO26s (9.5M params), fine-tuned from COCO weights on 1,450 labeled candlestick images across 20 imbalanced classes. Benchmarked against statistical baselines (logistic regression, random forest), a fine-tuned ResNet18, and an RT-DETR transformer to characterize what each model family contributes.
- **Confidence calibration:** Platt scaling maps raw detector confidence to calibrated probabilities (Brier score 0.0524).
- **Multi-timeframe confirmation:** a 2-of-3 rule that only keeps a pattern if it's detected on at least two of three charts, improving precision by up to +27.6 pp.
- **Pattern-evolution tracking:** progressive cropping (60/75/90/100% width) simulates a pattern forming over time; per-class trajectories are classified as evolving, stable, or dissolving.
- **Regime-aware filtering:** scales each detection by an empirical pattern-reliability weight (from Bulkowski's *Encyclopedia of Chart Patterns*) and a regime-alignment modifier, improving precision from 56.2% to 67.4%.

## Headline Results

| Component | Metric | Result |
|---|---|---|
| YOLO26s detector | Test mAP@50 / mAP@50-95 | 0.273 / 0.155 |
| Confidence calibration | Brier score | 0.0524 |
| Multi-timeframe filter | Precision gain | up to +27.6 pp |
| Regime filter | Precision | 56.2% → 67.4% |

The detector vs. baselines comparison shows the bottleneck is data, not model capacity: even an RT-DETR transformer with ~3× the parameters yields only a marginal gain on ~51 training images per class.

## Tech Stack

Python, PyTorch, Ultralytics YOLO26, RT-DETR, scikit-learn, OpenCV, pandas, NumPy.

## Data

Public Roboflow Universe "chart-pattern-2" dataset — 1,450 candlestick images, 20 classes, YOLO-format annotations (train/val/test = 1,024 / 285 / 141). The notebook downloads it via the Roboflow API (add your own free key as a Colab secret named `roboflow_api_key`).

## Running

Open `chart_pattern_recognition.ipynb` in Google Colab (GPU runtime recommended). Add a Roboflow API key under Runtime → Secrets, then run cells top to bottom. Saved outputs are included so the results are viewable without re-running.

## Notes

Originally developed as a course project. The YOLO26 architecture figure is not redistributed here; see Hidayatullah & Tubagus, 2026, [arXiv:2602.14582](https://arxiv.org/abs/2602.14582).
