[README.md](https://github.com/user-attachments/files/33991707/README.md)
# Breast Cancer Prediction

Machine-learning capstone: predicts whether a breast tumour is **benign** or **malignant** from 30 cell-nucleus measurements (Wisconsin Diagnostic Breast Cancer dataset).

> Educational project only. Not a medical device and not for real diagnosis.

## Contents
| File | Purpose |
|---|---|
| `Breast_Cancer_Prediction.ipynb` | Full workflow: EDA, train/test split, 4 models, comparison, over/underfit checks, final test evaluation |
| `app.py` | Streamlit web app (single patient and batch CSV prediction) |
| `breast_cancer_model.joblib` | Trained model (Logistic Regression pipeline) |
| `Breast_Cancer_Diagnostics.csv` | Dataset (used for slider ranges and reference charts) |
| `requirements.txt` | Python dependencies |
| `render.yaml` | Render deployment settings |

## Results (held-out test set)
Accuracy about 96.5%, recall about 92.9%, ROC-AUC about 0.996 (Logistic Regression).

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
