# Real-Time Human Reconstruction and Animation using Feed-Forward Gaussian Splatting

[![arXiv](https://img.shields.io/badge/arXiv-2604.10259-brightgreen.svg)](http://arxiv.org/abs/2604.10259)

This is the official repository for the paper **Real-Time Human Reconstruction and Animation using Feed-Forward Gaussian Splatting** (HumanGS). 

## 1. System Requirements
This codebase was developed and tested in the following environment:
* Ubuntu 24.04
* Python 3.11
* PyTorch 2.12.1
* CUDA 13.0

## 2. Installation

First, create and activate a Conda environment:
```bash
conda create -n HumanGS python=3.11
conda activate HumanGS
```

Clone this repository and the required rasterization submodule:
```
git clone https://github.com/Devdoot57/HumanGS.git
cd HumanGS
```

Run the provided installation script to set up PyTorch, standard dependencies, PyTorch3D, and Differential Gaussian Rasterization:
```
bash install.sh
```

### SMPL-X Body Models
You must download the neutral SMPL-X models to run the pipeline. 
1. Register and download the models from the [official SMPL-X website](https://smpl-x.is.tue.mpg.de/).
2. Create a `body_models` directory in the root of this repository.
3. Place the downloaded `.npz` and `.pkl` files exactly as shown below:

```text
HumanGS/
└── body_models/
    └── smplx/
        ├── SMPLX_NEUTRAL.npz
        └── SMPLX_NEUTRAL.pkl
```

## 3. Data Preparation

### Datasets
**THuman2.1:**
1. Visit the [THuman2.0-Dataset Repository](https://github.com/ytrock/THuman2.0-Dataset/blob/main/README.md).
2. Fill out and submit the dataset request form to the authors to gain access to the 3D scans. Extract and save these under `datasets/THuman2.1/data/model/`.
3. Download the SMPL-X fittings and save them under `datasets/THuman2.1/data/smplx/`.

**AvatarReX:**
1. Visit the [AvatarReX Dataset Guide](https://github.com/lizhe00/AnimatableGaussians/blob/master/AVATARREX_DATASET.md).
2. Download the required subjects and place them under `datasets/AvatarReX/`.

The expected directory structure is:
```text
HumanGS/
└── datasets/
    ├── THuman2.1/
    │   └── data/
    │       ├── model/
    │       └── smplx/
    └── AvatarReX/
```

### Preprocessing
We provide bash scripts to automate the data preprocessing and generation of evaluation indices.
To preprocess **THuman2.1:**
```bash
bash preprocess_thuman21.sh
```

*(You can override the default paths using `SCAN_DIR` and `SMPLX_DIR` environment variables if your data is stored elsewhere).*

To preprocess **AvatarReX**:
```bash
bash preprocess_avatarrex.sh
```
*(You can override the default path using the `DATA_DIR` environment variable).*

## 4. Training

To train the HumanGS model from scratch, execute the training script. By default, this is configured to run with 1 node and 4 processes per node using `torchrun`.
```bash
bash train.sh
```

## 5. Evaluation & Inference

To run inference and evaluate the model on the generated test splits, execute the evaluation script. This defaults to 1 process.
```bash
bash eval.sh
```

## 6. Asset Creation, Visualization & Animation

HumanGS predicts a canonical, animatable 3D Gaussian asset from input images in a single forward pass. The asset is saved as a standard `.ply` file, so it can also be opened in common 3DGS viewers.

### 6.1 Create a Canonical Asset

`create_asset.py` loads the trained checkpoint, takes one subject from the dataset, and saves its canonical Gaussians.

```bash
python create_asset.py --config configs/HumanGS_config.yaml --save_dir ./assets --subject_idx 27
```

The asset is written to `<save_dir>/<scene_name>_asset.ply`. The model, dataset, and checkpoint path are read from the same config used for training and evaluation.

### 6.2 Visualize an Asset

`visualize_asset.py` renders a 360° turntable video of a canonical asset.

```bash
python visualize_asset.py --asset_path ./assets/<scene_name>_asset.ply --output_video canonical_spin.mp4
```

### 6.3 Animate an Asset

`animate.py` animates a canonical asset with an SMPL-X motion sequence using linear blend skinning (LBS). The asset is retargeted to the motion skeleton, so any asset can be driven by any motion sequence.

**Default motion source:** by default, the motion is taken from the **`subject00`** sequence of the [THuman4.0](https://github.com/ZhengZerong/THuman4.0-Dataset) dataset. To use it, download THuman4.0 and make sure `subject00/smpl_params.npz` exists:

```text
HumanGS/
└── datasets/
    └── THuman4.0/
        └── subject00/
            └── smpl_params.npz
```

```bash
python animate.py \
    --asset_path ./assets/<scene_name>_asset.ply \
    --raw_data_root ./datasets/THuman4.0 \
    --motion_subject subject00 \
    --output_video thuman4_animation.mp4
```

To drive the asset with a different subject's motion, change `--motion_subject` (e.g. `subject01`).
