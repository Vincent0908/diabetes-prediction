#  Diabetes Prediction — ML Project
A complete Machine Learning project that predicts diabetes risk using clinical measurements from the **Pima Indians Diabetes Dataset** (UCI/Kaggle).

---

## 📋 Project Overview

| Item | Detail |
|------|--------|
| **Problem Type** | Binary Classification |
| **Dataset** | Pima Indians Diabetes (768 samples, 8 features) |
| **Models** | Logistic Regression + Neural Network (Keras) |
| **Deployment** | Streamlit Web Application |

---

## 🗂️ Project Structure

```
diabetes-prediction/
├── data/
│   └── diabetes.csv              # Raw dataset (downloaded from UCI)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Dataset loading and validation
│   ├── preprocessing.py          # Cleaning, feature engineering, scaling
│   ├── train.py                  # Model training (LR + NN)
│   ├── evaluate.py               # Metrics and visualization utilities
│   └── eda.py                    # Exploratory data analysis
│
├── models/
│   ├── logistic_model.pkl        # Saved Logistic Regression
│   ├── nn_model.keras            # Saved Neural Network
│   └── scaler.pkl                # Fitted StandardScaler
│
├── outputs/
│   ├── eda_01_class_distribution.png
│   ├── eda_02_feature_distributions.png
│   ├── eda_03_correlation_heatmap.png
│   ├── eda_04_boxplots.png
│   ├── eda_05_target_correlation.png
│   ├── cm_logistic.png
│   ├── cm_nn.png
│   ├── roc_comparison.png
│   ├── nn_training_history.png
│   └── metrics_summary.json
│
├── app.py                        # Streamlit deployment app
├── main.py                       # Full pipeline entry point
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone or download the project

```bash
cd diabetes-prediction
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Step 1 — Run the full ML pipeline

This will perform EDA, preprocess data, train both models, evaluate them,
and save all outputs:

```bash
python main.py
```

### Step 2 — Launch the Streamlit app

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 📊 Pipeline Stages

### 1. Exploratory Data Analysis (EDA)
- Class distribution analysis (1.87:1 imbalance)
- Feature distributions stratified by diabetes outcome
- Correlation matrix and heatmap
- Boxplot outlier analysis
- Feature-target correlation ranking

### 2. Data Preprocessing
- **Zero replacement**: Biologically impossible zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` are replaced with column medians
- **Feature Engineering**:
  - `BMI_Category` — clinical BMI bucket (underweight/normal/overweight/obese)
  - `Glucose_Insulin_Ratio` — proxy for insulin resistance
  - `Age_Pregnancies` — interaction feature
- **Stratified train/test split** (80/20)
- **StandardScaler** fitted on training set only (no data leakage)

### 3. Models

#### Logistic Regression
- Solver: `lbfgs`, C=1.0
- 5-fold cross-validation on training set
- CV Accuracy: ~0.77

#### Neural Network (Keras)
- Architecture: `Dense(64)→Dropout(0.3)→Dense(32)→Dropout(0.2)→Dense(1)`
- Optimizer: Adam (lr=0.001)
- Callbacks: EarlyStopping + ReduceLROnPlateau

### 4. Evaluation Metrics

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.7013 | 0.5909 | 0.4815 | 0.5306 | 0.8217 |
| Neural Network | 0.7208 | 0.6078 | 0.5741 | 0.5905 | 0.8106 |

> **Note on metrics**: ROC-AUC (~0.82) is a better indicator of model quality
> than raw accuracy here, given the class imbalance (65% negative class).

---

## 🌐 Streamlit App Features

- **Tab 1 – Prediction**: Enter patient data → select model → view prediction with probability
- **Tab 2 – Model Performance**: Confusion matrices, ROC curves, training history, feature importance
- **Tab 3 – EDA**: All exploratory visualizations + raw data preview

---

## ⚠️ Limitations

1. **Class imbalance**: Dataset is not perfectly balanced (65% no diabetes). Techniques like SMOTE or class weights could improve recall on the minority class.
2. **Dataset size**: 768 samples is small for a neural network — the LR model performs comparably.
3. **Medical disclaimer**: This tool is for educational purposes only. It is not a substitute for clinical diagnosis.

---

## 📦 Key Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| scikit-learn | 1.5.1 | Logistic Regression, preprocessing, metrics |
| TensorFlow/Keras | 2.17.0 | Neural Network |
| Pandas | 2.2.2 | Data manipulation |
| NumPy | 1.26.4 | Numerical computing |
| Matplotlib | 3.9.0 | Visualizations |
| Streamlit | 1.37.0 | Web deployment |
| joblib | 1.4.2 | Model serialization |

---

## Vincent Robin Christan Zebua

Built as a final project demonstrating the complete ML lifecycle:
data → EDA → preprocessing → modeling → evaluation → deployment.
