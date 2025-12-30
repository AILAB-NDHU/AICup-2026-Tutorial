# AICup 2026 - Go Game AI Tutorial

## Introduction

This repository contains tutorial materials for the AICup 2026 Competition, which focuses on building AI models for Go game analysis. The competition consists of two main challenges:

### 1. Rank Prediction
Predict the skill level (rank) of Go players based on their game records. The model classifies players into 20 different ranks ranging from **13 kyu (13k)** to **7 dan (7d)**.

### 2. Player Identification
Identify whether a target game was played by the same player as a set of reference games. This task involves learning unique playing styles and patterns through metric learning techniques.

Both tutorials demonstrate end-to-end deep learning pipelines using PyTorch, including:
- Feature extraction from SGF contents
- Model architecture design using ResNet
- Training and validation procedures
- Inference on test data
- Submission file generation

**Note:** These tutorials are provided for reference purposes only. Participants are encouraged to modify and improve the code, experiment with different architectures, and use any libraries or frameworks they prefer.

---

## File Structure

Files and directories in this repository are organized as follows:

Additional files will be generated during feature extraction and model training.

Make sure to place your dataset in the `dataset/` directory as shown below.

```
AICup-2026-Tutorial/
│
├── dataset/                                    # Dataset directory
│   ├── player_identification_test_sample.csv  # Sample test data for player
│   └── rank_prediction_test_sample.csv        # Sample test data for rank prediction
│
├── network.py                                  # Neural network architectures
│   ├── GoRankResNet                            # ResNet model for rank prediction
│   └── GoPlayerResNet                          # ResNet model for player identification
│
├── utils.py                                    # Utility functions for SGF parsing
│   ├── SGFParseRankPrediction                  # Feature extractor for rank prediction
│   └── SGFParsePlayerIdentification            # Feature extractor for player identification
│
├── player-identification-tutorial.ipynb        # Tutorial notebook for player identification
├── rank-prediction-tutorial.ipynb             # Tutorial notebook for rank prediction
│
└── README.md                                   # This file
```

---

## Running Environment

### System Requirements

The code has been tested on the following environments:

- **Operating Systems:**
  - Windows 11
  - Ubuntu 24.04 LTS

- **Python Version:**
  - Python 3.10

- **Deep Learning Framework:**
  - PyTorch 2.8.0
  - CUDA 12.8 (for GPU acceleration)

### Hardware Requirements

- **GPU Recommended:** Training deep learning models benefits significantly from GPU acceleration. A GPU with at least 8GB VRAM is recommended.
- **Disk Space:** 
  - Rank prediction features: ~20GB
  - Player identification features: ~160GB
  - Ensure sufficient disk space before running feature extraction

### Dependencies

Install the required Python packages:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install pandas numpy tqdm scikit-learn matplotlib seaborn
pip install webdataset sgfmill
```

---

## Getting Started

### 1. Rank Prediction Tutorial

Follow the [rank-prediction-tutorial.ipynb](rank-prediction-tutorial.ipynb) notebook to:
- Extract features from SGF contents using WebDataset
- Train a ResNet model to classify 20 rank levels
- Validate model performance with confusion matrix
- Generate predictions on test data

**Expected Baseline Performance:**
- Same rank accuracy: ~17%
- ±1 rank accuracy: ~37%

### 2. Player Identification Tutorial

Follow the [player-identification-tutorial.ipynb](player-identification-tutorial.ipynb) notebook to:
- Extract move-by-move features from SGF contents
- Train a ResNet embedding model using Triplet Loss
- Perform zero-shot inference on test data
- Generate binary predictions for player identification

**Expected Baseline Performance:**
- Accuracy: 70-80% (with proper training until loss ~0.06-0.1)

### 3. Pre-trained Models
Pre-trained models for both tasks are available in [link-rank](https://drive.google.com/drive/folders/1cTOdy-CQMocG4hEb3_jz0rZdvikEX1QK?usp=sharing) and [link-player](https://drive.google.com/drive/folders/1K9ZVY6Dbxg3ZzJ3Jz8jvKIjxdng8jdcJ?usp=sharing). You can use these models directly for inference or as a starting point for further training.

---

## Usage Tips

1. **Feature Extraction:** Both tutorials include a feature extraction step that can take approximately 1 hour. If you've already extracted features, you can skip this section and proceed directly to model training.

2. **GPU Acceleration:** For optimal performance, run the training on a system with GPU support. Adjust `BATCH_SIZE` in the notebooks based on your GPU memory.

3. **Customization:** Feel free to modify:
   - Model architectures in [network.py](network.py)
   - Feature extraction logic in [utils.py](utils.py)
   - Training hyperparameters in the notebooks
   - Loss functions and optimization strategies

4. **Checkpoints:** Models are automatically saved during training. The best model (based on performance) is saved separately for easy access.

---

## Competition Submission

After training your models and running inference on test data, submission files will be generated:
- `submission-rank.csv` - Rank prediction results
- `submission-player.csv` - Player identification results

Submit these files according to the competition guidelines.

---

## Acknowledgments

This tutorial was prepared by **Serkan Kavak, NDHU AI Lab** as a reference for the AICup 2026 Competition.

For any errors or issues, please open an issue on the project's GitHub repository.

---

## License

This project is provided for educational and competition purposes. Participants are free to use, modify, and distribute the code as needed for the competition.

Good luck with your models!
