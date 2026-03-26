# 🌊 DASE: Depth-Aware Scene-Adaptive Underwater Image Enhancement

> 🚧 **Ongoing Research Project** — This project is actively under development. Architecture, optimizations, and results are subject to change.

---

## 📌 Overview

Underwater images suffer from severe degradation due to wavelength-dependent light absorption, scattering, and color distortion. Traditional enhancement methods often apply global corrections, failing to account for spatial depth variations.

This project proposes **DASE (Depth-Aware Scene-Adaptive Enhancement)** — a hybrid pipeline that combines:

* Depth estimation using **MiDaS**
* Depth-stratified color correction
* Scene-adaptive yellow tone preservation
* Depth-weighted multi-contrast fusion

---

## 🧠 Key Contributions

* 🌊 **Depth-aware color correction** using spatial zone segmentation (near/mid/far)
* 🎯 **Scene-adaptive enhancement** with yellow-region preservation
* ⚡ **Multi-method contrast fusion** (HE + CLAHE + Linear Adjustment)
* 🧩 Hybrid approach combining classical methods + deep depth estimation

---

## 🏗️ Methodology

For a more detailed explanation of the methodology, please see [docs/methodology.md](docs/methodology.md).

### Pipeline:
Input Image → Depth Estimation (MiDaS) → Depth Zone Segmentation → Depth-Aware Color Correction → Contrast Fusion (HE + CLAHE + LA) → Enhanced Image

---

## 📂 Project Structure

```text
dase-underwater-enhancement/
│
├── src/
│   ├── config.py
│   ├── enhancement/
│   │   ├── dase_pipeline.py
│   │   ├── baseline.py
│   │
│   ├── depth/
│   │   ├── midas.py
│   │
│   ├── utils/
│   │   ├── metrics.py
│   │   ├── visualization.py
│   │
│   ├── cli.py
│
├── data/
├── results/
├── docs/
│   ├── methodology.md
│
├── underwater_enhancement.py  # original script (monolithic reference)
├── requirements.txt
├── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/gopu07/DASE-Underwater-Image-Enhancement.git
cd DASE-Underwater-Image-Enhancement
pip install -r requirements.txt
```

*(Note: It is highly recommended to use a virtual environment or conda environment)*

---

## 🚀 Usage

The project provides a unified Command-Line Interface (CLI).

### Single Image Example

```bash
python -m src.cli --image data/input.jpg --method dase --save results/enhanced.png
```

### Demonstration Results
Below is an example comparing the Baseline algorithm to the DASE pipeline:
![Comparison Results](results/demo_comparison.png)

### Compare Baseline vs DASE

```bash
python -m src.cli --compare data/input.jpg --output results/
```

### Batch Processing

```bash
python -m src.cli --batch data/input_dir/ --output results/
```

### Quick Demo (Synthetic Data)

```bash
python -m src.cli
```

---

## 📊 Evaluation Metrics

The pipeline evaluates outputs using standard quantitative measures:

* PSNR (Peak Signal-to-Noise Ratio)
* SSIM (Structural Similarity)
* UIQM (Underwater Image Quality Measure)
* UCIQE (Underwater Color Image Quality Evaluation)
* Entropy

---

## 🔬 Current Status
✔ Depth-aware enhancement pipeline implemented
✔ Evaluation metrics integrated
✔ Batch processing support
🚧 Coral-UWNet integration (planned)
🚧 Learning-based refinement (future work)
🚧 Benchmarking on UIEB dataset

---

## 🔮 Future Work
* Integrate deep model (Coral-UWNet)
* Real-time optimization
* Extensive benchmarking
* Ablation studies
* Paper submission

---

## 📖 Citation

```bibtex
@article{dase_underwater_2026,
  title={Depth-Aware Scene-Adaptive Underwater Image Enhancement},
  author={Devraj Solanki},
  year={2026}
}
```

---

## ⚠️ Disclaimer
This is an **ongoing research project**. Results and architecture may evolve significantly.

---

## 👨‍💻 Author
**Devraj Solanki**  
Electronics & Instrumentation Engineering  
Nirma University
