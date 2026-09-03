# AICup 2026 - Go Game AI Tutorial

> **English** | [繁體中文](README.md)

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

Place the officially released dataset in the repository root, keeping the `training/` and `tests/` directories exactly as distributed:

```
AICup-2026-Tutorial/
│
├── training/                                       # Training set (from the official release)
│   └── train_D.csv ... train_6D.csv                # One CSV per rank level (10 files, shared by both tasks)
│
├── tests/                                          # Test sets (from the official release)
│   ├── rank_prediction_test_public.csv             # Rank prediction, public
│   ├── rank_prediction_test_private.csv            # Rank prediction, private
│   ├── player_identification_test_public.csv       # Player identification queries, public
│   ├── player_identification_test_private.csv      # Player identification queries, private
│   ├── player_identification_candidates_public.csv # Candidate pool, public
│   └── player_identification_candidates_private.csv# Candidate pool, private
│
├── network.py                                      # Neural network architectures
│   ├── GoRankResNet                                # ResNet model for rank prediction
│   └── GoPlayerResNet                              # ResNet model for player identification
│
├── utils.py                                        # Utility functions for SGF parsing
│   ├── SGFParseRankPrediction                      # Feature extractor for rank prediction
│   └── SGFParsePlayerIdentification                # Feature extractor for player identification
│
├── rank-prediction.ipynb                           # Tutorial notebook for rank prediction
├── player-identification.ipynb                     # Tutorial notebook for player identification
│
├── README.md                                       # Traditional Chinese version
└── README.en.md                                    # This file
```

Feature directories, checkpoints and submission files are generated while running the notebooks.

## Dataset Configuration

The official release is the authoritative definition of the data and the submission format. This README summarises it; where the two disagree, follow the official release.

- **Training Set (shared by both tasks)**: 1,000,000 games (10 classes, 100,000 games per class), split into 10 files `train_<LEVEL>.csv` (one per rank level)

A single unified training set serves both tasks. Each row is one game:

| Column        | Description                                                                                              |
| ------------- | -------------------------------------------------------------------------------------------------------- |
| `player_id`   | Hashed unique identifier of the player (consistent across the whole training set)                        |
| `game_id`     | Identifier of the game record (e.g. `g_0000001`), unique **within** each rank file                       |
| `rank`        | The rank label of the player. Classes: D (10~12k), C (7~9k), B (4~6k), A (1~3k), 1D, 2D, 3D, 4D, 5D, 6D  |
| `color`       | The color ("B" or "W") played by the player in this game                                                 |
| `sgf_content` | The anonymised SGF string of the game                                                                    |

Usage per task (see the notebooks):

- Rank prediction: use `rank` as the label; `player_id` is not needed.
- Player identification: group rows by `player_id`; every game of a player forms a positive pair.

- **Test Sets**: each task has a **public** and a **private** set of 400 questions (5 to 20 games per question)

Both are released at the same time. Public scores appear on the leaderboard during the competition; private scores are published after it ends and decide the final ranking.

| Column                   | Description                                                                    |
| ------------------------ | ------------------------------------------------------------------------------ |
| `question_id`            | Identifier of the question, prefixed `pub_` or `priv_` (e.g. `pub_q_0001`)     |
| `num_games`              | Number of games provided in the question                                       |
| `sgf_1` ... `sgf_20`     | The SGF content of the games; only the first `num_games` columns are filled     |
| `color_1` ... `color_20` | The color ("B" or "W") played in the respective game                           |

- **Player Identification Candidates**: 400 players per pool (100 games per player), 40,000 games per pool

This file is the gallery of known players. Public and private pools are disjoint and must not be mixed.

| Column        | Description                                                        |
| ------------- | ------------------------------------------------------------------ |
| `game_id`     | Identifier of the record, prefixed `pub_` or `priv_`               |
| `player_id`   | Hashed identifier of the candidate, prefixed `pub_p_` or `priv_p_` |
| `sgf_content` | The anonymised SGF string of the game                              |
| `color`       | The color played by the candidate player                           |

---

## Running Environment

### System Requirements

The tutorials have been run end-to-end on:

- **Operating System:** Ubuntu 22.04.5 LTS
- **Python:** 3.10.12
- **Deep Learning Framework:** PyTorch 2.6.0+cu124, CUDA 12.4

The notebooks rely on the Linux `fork` start method, because the multiprocessing worker functions are defined inside the notebook. On macOS or Windows, set `NPROC = 1` or move those functions into a separate `.py` file.

### Hardware Requirements

- **GPU:**
  - Rank prediction: at least 9 GB VRAM (training ~8.6 GB, inference ~0.6 GB)
  - Player identification: at least 4 GB VRAM (training ~2 GB, inference ~4 GB)
  - Lower the `BATCH_SIZE` (and `GPU_BATCH` for inference) if VRAM is tight; this affects speed, not correctness.
- **Disk Space:**
  - Rank prediction features: ~1.7 GB (stored as gzipped WebDataset shards)
  - Player identification features: ~15 GB, spread over about 1,000,000 small files — check both free space and free inodes before extracting.

### Dependencies

Install the required Python packages:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install pandas numpy matplotlib
pip install webdataset sgfmill
```

---

## Getting Started

### 1. Rank Prediction Tutorial

Follow the [rank-prediction.ipynb](rank-prediction.ipynb) notebook to:

- Extract features from SGF contents using WebDataset
- Train a ResNet model to classify 10 rank levels
- Generate predictions for both the public and the private test set
- Baseline score: **0.4631** on the public test set

### 2. Player Identification Tutorial

Follow the [player-identification.ipynb](player-identification.ipynb) notebook to:

- Extract move-by-move features from SGF contents
- Train a ResNet embedding model using Triplet Loss
- Match query games against the candidate pool by embedding distance
- Generate top-5 predictions for both the public and the private test set
- Baseline score: **0.2845** on the public test set

Both baselines come from running the notebooks as they are, with no pre-trained weights.

---

## Usage Tips

1. **Feature Extraction:** Both notebooks start with a one-off feature extraction step. Once the features exist, later runs can skip straight to the training section.

2. **GPU Acceleration:** Run training on a machine with a GPU. Adjust `BATCH_SIZE` to fit your GPU memory.

3. **Customization:** Feel free to modify:
   - Model architectures in [network.py](network.py)
   - Feature extraction logic in [utils.py](utils.py)
   - Training hyperparameters in the notebooks
   - Loss functions and optimization strategies

4. **Checkpoints:** Every epoch is saved, and the best checkpoint is kept separately — by validation accuracy for rank prediction, by training loss for player identification.

---

## Competition Submission

Each task takes **one** submission file that covers both test sets. Predict the public and the private questions, then concatenate the results:

```
rank_prediction_test_public.csv   (400 questions)  ─┐
                                                    ├─→  submission_rank.csv   (800 rows)
rank_prediction_test_private.csv  (400 questions)  ─┘
```

Player identification works the same way, producing `submission_player.csv` with 800 rows. **A submission missing either half is invalid and will not be scored.**

| File                    | Columns                                   |
| ----------------------- | ----------------------------------------- |
| `submission_rank.csv`   | `question_id`, `pred_rank`                |
| `submission_player.csv` | `question_id`, `top_1` ... `top_5`        |

Copy the `pub_` / `priv_` prefixes verbatim. For player identification, the five predictions must be distinct and must come from the candidate pool of the same test set as the question — answering a `pub_` question with a `priv_p_` player invalidates the submission.

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

Public and private scores are computed independently, each averaged over its own 400 questions.

---

## Acknowledgments

This tutorial was prepared by **Serkan Kavak, NDHU AI Lab** as a reference for the AICup 2026 Competition.

For any errors or issues, please open an issue on the project's GitHub repository.

---

## License

This project is provided for educational and competition purposes. Participants are free to use, modify, and distribute the code.

Good luck with your models!
