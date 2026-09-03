# AICup 2026 - 圍棋 AI Tutorial

> [English](README.en.md) | **繁體中文**

## 簡介

本 repository 提供 AICup 2026 競賽的 tutorial 教材，主題是以 AI 分析圍棋棋譜。競賽分為兩個子題：

### 1. 棋力預測（Rank Prediction）

依據玩家的棋譜預測其棋力等級，共 10 個等級，範圍從 **12 級（12k）** 到 **6 段（6d）**。

### 2. 玩家辨識（Player Identification）

依據棋譜找出下這些棋的是誰。這個任務要透過 metric learning 學出每位玩家獨特的下棋風格。

兩份 tutorial 都示範了完整的 PyTorch 深度學習流程：

- 從 SGF 棋譜抽取特徵
- 以 ResNet 設計模型架構
- 訓練與驗證
- 對測試資料推論
- 產生提交檔

**注意：** 這些 tutorial 僅供參考。歡迎自行修改與改進程式碼、嘗試不同架構，也可以使用任何你偏好的函式庫或框架。

---

## 檔案結構

請把官方發佈的資料集放到 repository 根目錄，`training/` 與 `tests/` 兩個目錄保持發佈時的結構：

```
AICup-2026-Tutorial/
│
├── training/                                       # 訓練集（官方發佈）
│   └── train_D.csv ... train_6D.csv                # 每個等級一個 CSV（10 檔，兩個子題共用）
│
├── tests/                                          # 測試集（官方發佈）
│   ├── rank_prediction_test_public.csv             # 棋力預測 public
│   ├── rank_prediction_test_private.csv            # 棋力預測 private
│   ├── player_identification_test_public.csv       # 玩家辨識題目 public
│   ├── player_identification_test_private.csv      # 玩家辨識題目 private
│   ├── player_identification_candidates_public.csv # 候選庫 public
│   └── player_identification_candidates_private.csv# 候選庫 private
│
├── network.py                                      # 網路架構
│   ├── GoRankResNet                                # 棋力預測用的 ResNet
│   └── GoPlayerResNet                              # 玩家辨識用的 ResNet
│
├── utils.py                                        # SGF 解析工具
│   ├── SGFParseRankPrediction                      # 棋力預測的特徵抽取器
│   └── SGFParsePlayerIdentification                # 玩家辨識的特徵抽取器
│
├── rank-prediction.ipynb                           # 棋力預測 tutorial notebook
├── player-identification.ipynb                     # 玩家辨識 tutorial notebook
│
├── README.md                                       # 本文件
└── README.en.md                                    # English version
```

特徵目錄、checkpoint 與提交檔會在執行 notebook 的過程中產生。

## 資料集說明

資料與提交格式的正式定義以官方發佈為準。本文件是摘要，兩者有出入時以官方發佈為準。

- **訓練集（兩個子題共用）**：100 萬局（10 個等級、每級 10 萬局），分成 10 個 `train_<LEVEL>.csv`，每個等級一檔

單一份訓練集同時服務兩個子題。每一列是一局棋：

| 欄位          | 說明                                                                                          |
| ------------- | --------------------------------------------------------------------------------------------- |
| `player_id`   | 玩家的雜湊代號（在整份訓練集中一致）                                                          |
| `game_id`     | 棋譜代號（例如 `g_0000001`），只在**單一等級檔案內**唯一                                       |
| `rank`        | 玩家的等級標籤。10 級為 D (10~12k), C (7~9k), B (4~6k), A (1~3k), 1D, 2D, 3D, 4D, 5D, 6D      |
| `color`       | 該玩家在這局的執色（"B" 或 "W"）                                                              |
| `sgf_content` | 已匿名化的 SGF 棋譜字串                                                                       |

各子題的用法（詳見 notebook）：

- 棋力預測：以 `rank` 為標籤，用不到 `player_id`。
- 玩家辨識：依 `player_id` 分組，同一位玩家的任兩局即構成正例配對。

- **測試集**：兩個子題各有 **public** 與 **private** 兩份，每份 400 題（每題 5 到 20 局棋譜）

兩份同時發佈。Public 分數在競賽期間即時公布於排行榜；Private 分數於競賽結束後公布，並作為最終排名依據。

| 欄位                     | 說明                                                              |
| ------------------------ | ----------------------------------------------------------------- |
| `question_id`            | 題號，帶 `pub_` 或 `priv_` 前綴（例如 `pub_q_0001`）              |
| `num_games`              | 該題實際提供的棋譜數                                              |
| `sgf_1` ... `sgf_20`     | 棋譜內容；只有前 `num_games` 個欄位有值                           |
| `color_1` ... `color_20` | 對應棋譜的執色（"B" 或 "W"）                                      |

- **玩家辨識候選庫**：每池 400 位玩家（每人 100 局），每池共 40,000 局

這是已知玩家的資料庫。Public 與 private 兩池互斥，不可混用。

| 欄位          | 說明                                                     |
| ------------- | -------------------------------------------------------- |
| `game_id`     | 棋譜代號，帶 `pub_` 或 `priv_` 前綴                      |
| `player_id`   | 候選玩家的雜湊代號，帶 `pub_p_` 或 `priv_p_` 前綴        |
| `sgf_content` | 已匿名化的 SGF 棋譜字串                                  |
| `color`       | 該候選玩家的執色                                         |

---

## 執行環境

### 系統需求

兩份 tutorial 曾在以下環境完整執行：

- **作業系統：** Ubuntu 22.04.5 LTS
- **Python：** 3.10.12
- **深度學習框架：** PyTorch 2.6.0+cu124、CUDA 12.4

Notebook 依賴 Linux 的 `fork` 啟動方式，因為多進程的 worker 函式定義在 notebook 內。在 macOS 或 Windows 上執行時，請把 `NPROC` 設為 1，或把那些函式移到獨立的 `.py` 檔。

### 硬體需求

- **GPU：**
  - 棋力預測：至少 9 GB VRAM（訓練約 8.6 GB、推論約 0.6 GB）
  - 玩家辨識：至少 4 GB VRAM（訓練約 2 GB、推論約 4 GB）
  - VRAM 不足時調小 `BATCH_SIZE`（推論則是 `GPU_BATCH`），只影響速度，不影響正確性。
- **磁碟空間：**
  - 棋力預測特徵：約 1.7 GB（存成 gzip 壓縮的 WebDataset 分片）
  - 玩家辨識特徵：約 15 GB，散在約 100 萬個小檔案 —— 抽取前請同時確認剩餘空間與剩餘 inode 數量。

### 相依套件

安裝所需的 Python 套件：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install pandas numpy matplotlib
pip install webdataset sgfmill
```

---

## 開始使用

### 1. 棋力預測 Tutorial

跟著 [rank-prediction.ipynb](rank-prediction.ipynb) 完成：

- 以 WebDataset 從 SGF 棋譜抽取特徵
- 訓練 ResNet 模型分類 10 個等級
- 對 public 與 private 兩份測試集產生預測
- Baseline 分數：Public 測試集 **0.4631**

### 2. 玩家辨識 Tutorial

跟著 [player-identification.ipynb](player-identification.ipynb) 完成：

- 從 SGF 棋譜逐手抽取特徵
- 以 Triplet Loss 訓練 ResNet embedding 模型
- 用 embedding 距離把題目比對到候選庫
- 對 public 與 private 兩份測試集產生 Top-5 預測
- Baseline 分數：Public 測試集 **0.2845**

兩個 baseline 都是照 notebook 原樣跑出來的，沒有使用預訓練權重。

---

## 使用提示

1. **特徵抽取：** 兩份 notebook 開頭都有一次性的特徵抽取。特徵產出後，之後重跑可以直接從訓練那節開始。

2. **GPU 加速：** 請在具備 GPU 的機器上訓練，並依 GPU 記憶體調整 `BATCH_SIZE`。

3. **自行修改：** 以下都可以改：
   - [network.py](network.py) 的模型架構
   - [utils.py](utils.py) 的特徵抽取邏輯
   - Notebook 內的訓練超參數
   - 損失函數與最佳化策略

4. **Checkpoint：** 每個 epoch 都會存檔，最佳的那個另外保留 —— 棋力預測依驗證集 accuracy，玩家辨識依訓練 loss。

---

## 競賽提交

每個子題只上傳**一份**提交檔，內容涵蓋兩份測試集。Public 與 private 的題目都要預測，結果串接起來：

```
rank_prediction_test_public.csv   （400 題）  ─┐
                                               ├─→  submission_rank.csv   （800 列）
rank_prediction_test_private.csv  （400 題）  ─┘
```

玩家辨識同理，產生 800 列的 `submission_player.csv`。**缺任何一份的題目即為缺題，該次提交無效、不予計分。**

| 檔案                    | 欄位                                      |
| ----------------------- | ----------------------------------------- |
| `submission_rank.csv`   | `question_id`、`pred_rank`                |
| `submission_player.csv` | `question_id`、`top_1` ... `top_5`        |

`pub_` / `priv_` 前綴請原樣照抄。玩家辨識的五個人選必須相異，且只能填該題所屬測試集候選庫裡的玩家 —— `pub_` 開頭的題目填了 `priv_p_` 開頭的玩家，會被判為無效提交。

---

## 計分方式

### 棋力預測

測試集中每一列 $i$ 依預測的等級計分：

$$
score_i =
\begin{cases}
1, & \text{完全正確} \\
e^{-1}, & \text{相鄰一級（不含完全正確）} \\
0, & \text{其餘}
\end{cases}
$$

最終分數為所有題目的平均：

$$
\text{Final Score} = \frac{1}{N} \sum_{i=1}^{N} score_i
$$

### 玩家辨識

測試集中每一列 $i$ 依正解是否落在 Top-5 計分：

$$
score_i(r) = e^{- (r - 1)}, \quad r \in \{1,2,3,4,5\}
$$

其中 $r$ 是正解在 Top-5 名單中的名次。若正解不在 Top-5，該題得 0 分。

最終分數為所有題目的平均：

$$
\text{Final Score} = \frac{1}{N} \sum_{i=1}^{N} score_i(r)
$$

Public 與 private 分數各自獨立計算，各取自己那 400 題的平均。

---

## 致謝

本 tutorial 由 **NDHU AI Lab 的 Serkan Kavak** 製作，作為 AICup 2026 競賽的參考教材。

若發現錯誤或有任何問題，請在本專案的 GitHub repository 開 issue。

---

## 授權

本專案供教學與競賽用途。參賽者可自由使用、修改與散布本程式碼。

祝你的模型一切順利！
