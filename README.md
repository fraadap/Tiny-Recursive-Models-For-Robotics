# 🤖 TinyRecursiveModels for Robotics

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 🧠 Applying TinyRecursiveModels (TRM) to robotic manipulation tasks with improved sample efficiency

## 📋 Abstract

Robotic manipulation tasks require models that can efficiently learn complex behaviors from limited demonstrations through behavioral cloning. Current state-of-the-art vision-language-action (VLA) models like **Pi0** and **OpenVLA** achieve impressive performance but suffer from poor sample efficiency due to their massive parameter counts.

This project investigates the adaptation of **TinyRecursiveModels (TRM)**, a lightweight architecture that implements recursive reasoning through iterative processing, to robotic manipulation tasks. We evaluate TRM on the **LIBERO-Spatial** benchmark, focusing on pick-and-place manipulation tasks.

### ✨ Key Highlights

- 🔄 **Recursive Reasoning**: Achieves strong performance through iterative processing with shared parameters
- 📉 **Sample Efficiency**: Significantly fewer parameters than billion-parameter VLA models
- 🎯 **LIBERO Benchmark**: Evaluated on 10 pick-and-place manipulation tasks
- 💡 **Lightweight**: 256-dimensional hidden representations with only 16 recursive blocks

---

## 🏗️ Architecture

Our **TRMPolicy** architecture consists of four main components:

| Component | Description |
|-----------|-------------|
| 🖼️ **Visual Encoder** | ResNet18 + projection layer (ImageNet1K-V1 pretrained) |
| 📝 **Text Encoder** | CLIP-ViT-Large (frozen) |
| 🔄 **Recursive Module** | 16 RecursiveBlocks with shared parameters |
| 🎮 **Action Head** | MLP mapping to 7-DoF robot actions |

### 🔧 Multimodal Encoding

- **Visual Features**: RGB images (128×128×3) → ResNet18 → 512-dim features
- **Text Features**: Task descriptions → CLIP → 768-dim → projected to 256-dim
- **Fusion**: Concatenation → 768-dim → Fusion Adapter → 256-dim

### 🔁 Recursive Reasoning Module

Each **RecursiveBlock** contains:
- Layer Normalization
- 2-layer MLP (256 → 1024 → 256) with GELU activation
- Residual connection

The same block is applied **16 times** iteratively, enabling the model to refine its understanding progressively.

---

## 📊 Dataset

We use **LIBERO-Spatial** from the [LIBERO benchmark](https://github.com/Lifelong-Robot-Learning/LIBERO):

- 📦 **10 pick-and-place tasks** with a black bowl
- 🎬 **~50 demonstrations** per task
- 📸 **Multimodal observations**: RGB (workspace + wrist cameras), proprioception, language instructions
- 📈 **80/20 train-test split**

---

## 🚀 Getting Started

### 📋 Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- ~10GB disk space for dataset

### 💻 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Tiny-Recursive-Models-For-Robotics.git
   cd Tiny-Recursive-Models-For-Robotics
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or install the minimal dependencies:
   ```bash
   pip install torch torchvision transformers einops opencv-python h5py numpy scipy pillow
   ```

### 📥 Download Dataset

The notebook automatically downloads the LIBERO dataset. You can also download it manually:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
python LIBERO/benchmark_scripts/download_libero_datasets.py --datasets libero_spatial --use-huggingface
mkdir dataset
mv LIBERO/libero/datasets/* dataset/
```

---

## 🏃 Usage

### 📓 Run the Notebook

Open and run the Jupyter notebook:

```bash
jupyter notebook trm_on_libero.ipynb
```

Or in VS Code, simply open `trm_on_libero.ipynb` and run cells sequentially.

### ⚙️ Configuration

Modify training hyperparameters in the `TrainingConfig` dataclass:

```python
@dataclass
class TrainingConfig:
    lr: float = 3e-4              # Learning rate
    hidden_dim: int = 256         # Hidden dimension
    num_recursions: int = 8       # Number of recursive steps
    epochs: int = 20              # Training epochs
    batch_size: int = 64          # Batch size
    weight_decay: float = 1e-4    # L2 regularization
    grad_clip: float = 1.0        # Gradient clipping
    dropout: float = 0.1          # Dropout rate
    freeze_backbone: bool = True  # Freeze ResNet backbone
    augmentation: bool = False    # Enable data augmentation
```

---

## 📁 Project Structure

```
Tiny-Recursive-Models-For-Robotics/
├── 📓 trm_on_libero.ipynb     # Main training notebook
├── 📄 requirements.txt         # Python dependencies
├── 📄 requirements.in          # Minimal dependencies
├── 📂 dataset/                 # LIBERO dataset (after download)
├── 📂 evaluation_videos/       # Generated evaluation videos
└── 📂 final_aml_report/        # LaTeX report
    ├── main.tex
    └── refs.bib
```

---

## 🔬 Training Details

| Parameter | Value |
|-----------|-------|
| **Optimizer** | AdamW |
| **Loss Function** | 0.7 × MSE + 0.3 × L1 |
| **Action Normalization** | Z-score |
| **Early Stopping** | Based on validation loss |
| **Gradient Clipping** | 1.0 |

---

## 📚 References

- [TinyRecursiveModels](https://arxiv.org/abs/2501.07835) - Jolicoeur-Martineau et al.
- [LIBERO Benchmark](https://github.com/Lifelong-Robot-Learning/LIBERO) - Liu et al.
- [OpenVLA](https://openvla.github.io/) - Kim et al.
- [Pi0](https://www.physicalintelligence.company/blog/pi0) - Physical Intelligence

---

## 👥 Authors

- **Francesco D'Aprile**
- **Sara Lazzaroni**
- **Riccardo Bastiani**


---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
