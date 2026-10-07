# Music Genre Classification: Dilated CNN + Classical Algorithms

**UE24CS352A Machine Learning, Mini-Project**
**Problem statement:** Music Classification through CNN and Classical Algorithms

This project implements the approach of *"Combining CNN and Classical Algorithms for Music Genre Classification"* (Li, Xue, Zhang, Stanford). A small Dilated CNN is trained on MFCC features of music clips, and its layer activations are then used as input features for four classical classifiers: Logistic Regression (LR), Gaussian Discriminant Analysis (GDA), Random Forest (RF) and SVM. These are compared against the same classifiers trained on raw and PCA-reduced MFCCs.

## Team
| Name | SRN |
|------|-----|
| Nandita N Patil | PES2UG24CS304 |
| Nakul Jayan | PES2UG24CS298 |

## Dataset
- **GTZAN Genre Collection** (Tzanetakis & Cook), originally from Kaggle: [GTZAN Dataset - Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
- **Genres used (5):** blues, classical, hiphop, metal, pop (100 clips each, 500 total)
- **Trimmed subset:** to keep the repo small, each clip is cut to its first **13 s** (originals are 30 s). The trimmed `.wav` files are in [`dataset/gtzan_trimmed/`](dataset/gtzan_trimmed), so no separate download is needed.
- **Split:** 80/20 stratified train/test (400 / 100 clips, 20 test clips per genre), fixed seed (42), identical across all experiments.

## Approach
1. **Preprocessing:** extract 20 MFCCs per clip with librosa (22,050 Hz), fixed at 560 frames, giving a `(560, 20)` array per clip.
2. **Baselines:** LR, GDA (`LinearDiscriminantAnalysis`, shared covariance), RF (100 trees, max depth 7) and SVM (RBF, `gamma="scale"`) on raw flattened MFCCs (11,200 features) and on PCA-50 (fit on the training set only). No feature scaling is applied, as in the paper.
3. **Dilated CNN** (6,277 parameters): two units of `Conv1D (dilated) -> AvgPool -> Dropout(0.5) -> BatchNorm`, then Flatten and a 5-way softmax.
   - Unit 1: 8 filters, kernel 16, dilation 8, pool 32 (output 17 x 8 = 136 features)
   - Unit 2: 24 filters, kernel 16, dilation 2, pool 4 (output 4 x 24 = 96 features)
   - Adam (lr 0.001), cross-entropy loss, batch size 16, early stopping (patience 15) on a 15% validation split of the training data, inputs standardised per MFCC coefficient with training statistics.
4. **CNN as feature extractor:** the flattened activations of unit 1 and unit 2 are fed to the four classical models.
5. **Evaluation:** train/test accuracy table, confusion matrices, and 3D PCA plots of the feature spaces.

## Repository structure
```
ml_mini_project/
├── README.md
├── requirements.txt
├── .gitignore
├── dataset/gtzan_trimmed/        # 5 genres, 13 s clips (.wav)
├── notebooks/
│   └── music_classification.ipynb
├── features/
│   └── mfcc_features.npz         # cached MFCCs: X (500x560x20), y, genres
├── models/
│   └── dilated_cnn.keras         # trained dilated CNN
└── results/                      # accuracy_table.csv and all plots
```

## Setup and run

### Option A: Google Colab (recommended)
1. Click the **Open in Colab** badge above (or open `notebooks/music_classification.ipynb` in Colab).
2. Run all cells top to bottom (Runtime -> Run all). The first cell clones this repo to get the dataset.
3. Outputs are written to `features/`, `models/` and `results/` inside the Colab session, and the last cell zips them for download.

Developed and run on Colab (Python 3.13, TensorFlow 2.21, Keras 3.13).

### Option B: Local
```bash
git clone https://github.com/NanditaPatil-dotcom/ml_mini_project.git
cd ml_mini_project
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/music_classification.ipynb
```
For a local run, set `REPO_DIR` in the notebook's config cell to your local repo path and skip the `git clone` line.

Without a GPU, a full run takes a few minutes (MFCC extraction is the slowest step).

### Loading the saved files
```python
import numpy as np, tensorflow as tf

d = np.load("features/mfcc_features.npz", allow_pickle=True)   # d["X"], d["y"], d["genres"]
model = tf.keras.models.load_model("models/dilated_cnn.keras")
```
If `features/mfcc_features.npz` is present, the notebook loads it instead of re-extracting MFCCs from the audio.

## Results
Accuracy on the 400 training / 100 test clips, shown as train / test. Baseline rows (raw MFCC, PCA-50) do not depend on the CNN and are deterministic; rows that depend on the trained CNN are given as approximate values (`~`), since they can shift slightly between runs, hardware and library versions.

| Features | LR | GDA | RF | SVM |
|----------|----|-----|----|-----|
| Raw MFCC (11,200) | 1.000 / 0.70 | 0.905 / 0.73 | 1.000 / 0.76 | 0.813 / 0.76 |
| PCA-50 | 0.963 / 0.69 | 0.850 / 0.78 | 0.985 / 0.71 | 0.853 / 0.76 |
| DCNN layer 1 (136) | ~0.97 / ~0.76 | ~0.95 / ~0.74 | ~0.99 / ~0.79 | ~0.92 / ~0.81 |
| DCNN layer 2 (96) | ~0.93 / ~0.82 | ~0.93 / ~0.78 | ~0.99 / ~0.81 | ~0.89 / ~0.81 |

**Dilated CNN alone (end-to-end):** train ~0.87, test ~0.78

The exact values for the committed run are in [`results/accuracy_table.csv`](results/accuracy_table.csv). Seeds are fixed for the split, Random Forest and TensorFlow. Plots in `results/`:
- `cnn_confusion_matrix.png`, `lr_confusion_matrix.png` (LR on DCNN layer 1)
- `pca_raw.png`, `pca_layer1.png`, `pca_layer2.png` (3D PCA of test features)
- `cnn_training_curves.png`, `mfcc_examples.png`

## Key findings
- **CNN features improve the classical models.** Averaged over the four classifiers, test accuracy goes from about 0.74 on raw MFCCs to about 0.78 on layer 1 and about 0.80 on layer 2. Layer 2 beats raw features for every classifier, by roughly 5 to 12 points.
- **Classical models can match or beat the CNN that produced the features.** The CNN alone reaches about 0.78 test accuracy, while LR, RF and SVM on layer 2 reach about 0.8, and RF and SVM on layer 1 are similar. The margins are small (see the caveat below).
- **Overfitting is reduced.** On raw input LR and RF fit the training set perfectly (1.00) but score 0.70 and 0.76 on test. With layer 2 features the train/test gap for LR shrinks from 0.30 to roughly 0.1, using a feature vector more than 100x smaller (96 vs 11,200).
- **PCA alone does not help consistently.** Compared with raw features it lowers LR (0.70 to 0.69) and RF (0.76 to 0.71), raises GDA (0.73 to 0.78) and leaves SVM unchanged (0.76).
- **Blues and hip-hop are the hardest genres.** Classical is almost always identified correctly, and metal and pop are rarely confused with other genres. Blues and hip-hop overlap with each other and with metal. The PCA plots show the same pattern, with layer 2 separating classical, metal and pop more cleanly than raw features.
- **Caveat:** with only 100 test clips, one clip equals one percentage point and the standard error is about 4 points, so differences of 1-3 points between individual cells are within noise. The consistent advantage of layer-2 features over raw features across all four classifiers is more reliable than any single cell.

## Differences from the paper
- Clips are trimmed to 13 s (paper: 30 s), so feature sizes and accuracies differ from the paper's.
- CNN layer sizes are adapted to the shorter input; early stopping on a validation split is used.
- **SVM on raw MFCCs did not collapse here** (0.76 test, versus 0.28 in the paper). We use `gamma="scale"`, which adapts the RBF width to the feature variance; a different kernel width is a likely reason the paper's SVM overfit so severely. SVM here underfits slightly on raw input (0.81 train) rather than memorising it.
- In the paper metal was the hardest class; in our run it is one of the easiest, and blues and hip-hop are hardest.
- The paper's absolute accuracies are higher (CNN 0.84-0.87), most likely because of the longer clips and different tuning.

## References
1. Li, Xue, Zhang. *Combining CNN and Classical Algorithms for Music Genre Classification.* Stanford University.
2. Tzanetakis & Cook. *Musical genre classification of audio signals.* IEEE Trans. Speech and Audio Processing, 10(5), 2002.
