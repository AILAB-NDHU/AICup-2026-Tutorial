# AICup 2026 - Go Game AI Tutorial

## Introduction

This repository contains tutorial materials for the AICup 2026 Competition, which focuses on building AI models for Go game analysis. The competition consists of two main challenges:

### 1. Rank Prediction
Predict the skill level (rank) of Go players based on their game records. The model classifies players into 10 different ranks ranging from **12 kyu (12k)** to **6 dan (6d)**.

### 2. Player Identification
Identify the player who played target games based on their game records. This task involves learning unique playing styles and patterns through metric learning techniques.

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
│   ├── rank_prediction_train.csv               # Training data for rank prediction
│   ├── rank_prediction_test.csv                # Test data for rank prediction
│   ├── player_identification_train.csv         # Training data for player identification
│   ├── player_identification_candidates.csv    # Candidate pool for player identification
│   └── player_identification_test.csv          # Test data for player identification
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

## Dataset Configuration

- **Rank Prediction Train**: 1,000,000 games (10 classes, 100,000 games per class)

Contains labeled games for training models to predict the rank of a player.

| Column | Description |
|--------|-------------|
| `game_id` | Unique identifier for the game record (e.g., `train_000000`) |
| `sgf_content` | The full SGF string of the game |
| `target_color` | The color ("B" or "W") of the player whose rank is being predicted |
| `rank` | The rank label (target class). Classes: D (12-10k), C (9-7k), B (6-4k), A (3-1k), 1D, 2D, 3D, 4D, 5D, 6D |

- **Rank Prediction Public Test**: 400 questions (20 games per question)

Contains N-shot like questions where the model must determine the rank of the target player based on a set of their games.

| Column | Description |
|--------|-------------|
| `question_id` | Unique identifier for the test question (e.g., `test_000000`) |
| `num_games` | Number of games provided in the question |
| `sgf_1` ... `sgf_20` | The SGF content of the games played by the target player |
| `color_1` ... `color_20` | The color ("B" or "W") played by the target player in the respective game |

- **Player Identification Train**: 200,000 games (1000 players, 200 games per player)

This file serves as big data for training player identification models.

| Column | Description |
|--------|-------------|
| `game_id` | Unique identifier for the record (e.g., `train_000000`) |
| `player_id` | Hashed unique identifier of the player |
| `sgf_content` | The full SGF string of the game |
| `color` | The color ("B" or "W") played by the candidate player |


- **Player Identification Candidates**: 400 players (100 games per player), Total 40,000 games

This file serves as the database/gallery of known players. It contains multiple games for each candidate player.

| Column | Description |
|--------|-------------|
| `game_id` | Unique identifier for the record (e.g., `cand_000000`) |
| `player_id` | Hashed unique identifier of the player |
| `sgf_content` | The full SGF string of the game |
| `color` | The color ("B" or "W") played by the candidate player |

- **Player Identification Public Test**: 400 questions (20 games per question)

Contains questions where the model gets a set of games from an unknown player and must identify which `player_id` from the candidates pool they belong to.

| Column | Description |
|--------|-------------|
| `question_id` | Unique identifier for the test question |
| `num_games` | Number of games provided in the query |
| `sgf_1` ... `sgf_20` | The SGF content of the query games |
| `color_1` ... `color_20` | The color ("B" or "W") played by the query player in the respective game |


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
  - Rank prediction features: ~180 GB
  - Player identification features: ~160 GB
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
- Train a ResNet model to classify 10 rank levels
- Validate model performance with confusion matrix
- Generate predictions on test data
- Expected Baseline Score: ~0.51 on the test set using pre-trained model

### 2. Player Identification Tutorial

Follow the [player-identification-tutorial.ipynb](player-identification-tutorial.ipynb) notebook to:
- Extract move-by-move features from SGF contents
- Train a ResNet embedding model using Triplet Loss
- Perform zero-shot inference on test data
- Generate top-5 predictions for player identification
- Expected Baseline Score: ~0.30 on the test set using pre-trained model


### 3. Pre-trained Models
Pre-trained models for both tasks are available in [link-rank](https://drive.google.com/drive/folders/1cTOdy-CQMocG4hEb3_jz0rZdvikEX1QK?usp=sharing) and [link-player](https://drive.google.com/drive/folders/1K9ZVY6Dbxg3ZzJ3Jz8jvKIjxdng8jdcJ?usp=sharing). You can use these models directly for inference or as a starting point for further training.

---

## Usage Tips

1. **Feature Extraction:** Both tutorials include a feature extraction step that can take approximately 2 hours. If you've already extracted features, you can skip this section and proceed directly to model training.

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

## Scoring

### Rank Prediction: 
Each row $i$ in the test set is scored based on the predicted rank:
$$
score_i = 
\begin{cases}
1, & \text{exact rank match} \\ 
e^{-1}, & \text{within} \pm 1 \text{ rank (excluding exact match)} \\ 
0, & \text{otherwise}
\end{cases}
$$

The final score is the average over all test samples:
$$
\text{Final Score} = \frac{1}{N} \sum_{i=1}^{N} score_i
$$

### Player Identification: 
Each row $i$ in the test set is scored based on whether the correct player ID is within the top-5 predictions:
$$
score_i(r) = e^{- (r - 1)}, \quad r \in \{1,2,3,4,5\}
$$
where $r$ is the placement of the correct player ID in the top-5 list. If the correct ID is not in the top-5, the score is 0.

The final score is the average over all test samples:
$$
\text{Final Score} = \frac{1}{N} \sum_{i=1}^{N} score_i(r)
$$

---

## Acknowledgments

This tutorial was prepared by **Serkan Kavak, NDHU AI Lab** as a reference for the AICup 2026 Competition.

For any errors or issues, please open an issue on the project's GitHub repository.

---

## License

This project is provided for educational and competition purposes. Participants are free to use, modify, and distribute the code.

Good luck with your models!
