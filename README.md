# Loan Default Prediction

A machine learning project to predict whether a loan applicant is likely to default.

## Project Structure

```
Loan_Default_Prediction/
├── data/
│   ├── raw/           # Original, unmodified datasets
│   └── processed/     # Cleaned and feature-engineered data
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Modeling.ipynb
│   └── 04_Evaluation.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── models/            # Saved trained models
├── app.py             # Streamlit web app
├── requirements.txt
└── README.md
```

## Setup

```bash
cd Loan_Default_Prediction
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. Place your raw dataset in `data/raw/`.
2. Run the notebooks in order (`01` through `04`) or use the scripts directly.
3. Train the model:

```bash
python src/train.py
```

4. Launch the prediction app:

```bash
streamlit run app.py
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_EDA.ipynb` | Exploratory data analysis |
| `02_Preprocessing.ipynb` | Data cleaning and feature engineering |
| `03_Modeling.ipynb` | Model training and hyperparameter tuning |
| `04_Evaluation.ipynb` | Model evaluation and interpretation |
