# Robust Full Body 3D Human Pose Estimation

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Official Implementation** of the Master's Thesis: *Robust Full Body 3D Human Pose Estimation* (AIMS South Africa, 2025).

## 📝 Abstract

Despite the presence of expert models for specific body parts (e.g., hands and faces), existing unified approaches often fail to deliver consistent, high-fidelity 3D representations of the entire human body.

This repository implements a **Composition-of-Experts Framework**. Instead of training a monolithic network, this pipeline strategically fuses state-of-the-art specialized models within the unified SMPL-X parameter space:

* **Body:** [SMPLest-X](https://github.com/smplest-x) for global pose and shape.
* **Hands:** [WiLoR](https://github.com/wilor) for high-fidelity hand articulation.
* **Face:** [EMOCA](https://github.com/emoca) for expressive facial geometry.

The result is a unified mesh that preserves the sub-millimeter accuracy of expert models while maintaining global anatomical coherence.

## 🏗️ Architecture

![Pipeline Architecture](PUT_YOUR_IMAGE_PATH_HERE.png)

The pipeline operates in four stages:
1.  **Global Estimation:** Inferring base body parameters using SMPLest-X.
2.  **Expert Extraction:** Regressing high-fidelity parameters for hands (WiLoR) and face (EMOCA).
3.  **Parameter Transformation:** Aligning expert coordinate systems (e.g., specific mirroring for left-hand MANO parameters) to the SMPL-X standard.
4.  **Fusion:** Synthesizing the final mesh.

## 📂 Repository Structure

* `adapters/`: Interfaces for the expert models (WiLoR, EMOCA, SMPLest-X).
* `fusion/`: Core logic for parameter composition and coordinate transformation.
* `data/`: Input images and output storage.
* `pretrained_models/`: Checkpoints for the backbone and expert models.
* `evaluation/`: Scripts for V2V and PA-V2V metric calculation.
* `bash_scripts/`: Utilities for running large-scale evaluation on the EHF dataset.
* `main.py`: The primary inference entry point.

## 🛠️ Installation

This codebase was developed and tested on **Python 3.10** and **PyTorch 1.12.1**.


# Clone the repository
git clone --recursive [https://github.com/Roda10/3d_whole_body_pipeline.git](https://github.com/Roda10/3d_whole_body_pipeline.git)
cd 3d_whole_body_pipeline

# Create environment
conda create -n fusion_body python=3.10
conda activate fusion_body

# Install dependencies
pip install -r requirements.txt


> **Note:** You will need to download the official SMPL-X models (Neutral, Male, Female) from the [SMPL-X Project Page](https://smpl-x.is.tue.mpg.de/) and place them in the `pretrained_models` directory.

## 🚀 Usage

To run the fusion pipeline on a single image:

```bash
python main.py --input_path data/your_image.jpg --output_dir data/output/ --visualize True
```

## 🎓 Citation

If you find this work useful for your research, please cite the thesis:

```bibtex
@mastersthesis{toha2025robust,
  title={Robust Full Body 3D Human Pose Estimation},
  author={Toha, Rodéo Oswald Y.},
  school={African Institute for Mathematical Sciences (AIMS), South Africa},
  year={2025},
  month={June},
  note={Supervised by Dr. Rolandos Alexandros Potamias and Dr. Jiankang Deng}
}
```

## 🙏 Acknowledgements

This research was conducted at the **African Institute for Mathematical Sciences (AIMS), South Africa**, and supervised by Dr. Rolandos Alexandros Potamias and Dr. Jiankang Deng. We utilize code from the open-source community, specifically the implementations of SMPL-X, WiLoR, and EMOCA.

```
```
