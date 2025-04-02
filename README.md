# Audio-Deepfake-Detection
# Enhanced AASIST and AASIST2 Project

Welcome to the **Enhanced AASIST and AASIST2** repository! This project focuses on improving speech anti-spoofing systems using advanced machine learning techniques. It includes two main components: `main.py` (Enhanced AASIST) and `AASIST2` (my customized version). Please find the detailed assessment in the file: **Momenta Assessment.pdf** located in this repository.

---

## Table of Contents
- [About the Project](#about-the-project)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Results](#results)

---

## About the Project

### Enhanced AASIST
The `main.py` script implements enhancements to the original AASIST model, which is based on a RawNet2 encoder and a graph network module for speech anti-spoofing.

### AASIST2
The `AASIST2` version builds upon Enhanced AASIST by replacing residual blocks with Res2Net blocks, enabling multi-scale feature extraction. It also incorporates dynamic chunk size (DCS) and adaptive large margin fine-tuning (ALMFT) strategies to improve performance on short utterances.

---

## Features

- **Enhanced Anti-Spoofing**: Improved detection of spoofed speech using advanced techniques like Res2Net.
- **Multi-Scale Feature Extraction**: Leverages Res2Net blocks for better temporal representation.
- **Dynamic Chunk Size (DCS)**: Adapts to varying speech durations.
- **Pre-trained Models**: Supports pre-trained wav2vec 2.0 XLS-R for robust feature extraction.
- **Evaluation Metrics**: Includes Equal Error Rate (EER) and min t-DCF for performance benchmarking.

---

## Installation

Follow these steps to set up the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/waliapriyanshu/Audio-Deepfake-Detection.git
   cd Audio-Deepfake-Detection
   ```

2. Set up a Python environment:
   ```bash
   conda create -n aasist_env python=3.8
   conda activate aasist_env
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install additional libraries for `AASIST2`:
   ```bash
   pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html
   ```

---

## Usage

### Running Enhanced AASIST (`main.py`)
To train or evaluate the Enhanced AASIST model:
```bash
python main.py --train --dataset_path /path/to/dataset
```

### Running AASIST2 (`AASIST2`)
To train or evaluate your customized version:
```bash
python AASIST2.py --train --dataset_path /path/to/dataset --use_res2net True
```

---

## Datasets

This project uses the ASVspoof datasets:

1. **ASVspoof 2019 LA Dataset**: Used for training.
2. **ASVspoof 2021 LA and DF Datasets**: Used for evaluation.

Datasets can be downloaded from their respective official sources: https://datashare.ed.ac.uk/handle/10283/3336.

---

## Results

### Performance Metrics:
| Model           | Dataset          | EER (%) | min t-DCF |
|-----------------|------------------|---------|-----------|
| Enhanced AASIST | ASVspoof 2021 LA | 0.90    | 0.210     |
| AASIST2         | ASVspoof 2021 DF | 2.85    | -         |

AASIST2 demonstrates significant improvements in short utterance evaluations due to its multi-scale feature extraction capabilities.
