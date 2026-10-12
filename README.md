# Formula 1 Podium Prediction — Milestone 1

**CSEN 903: Advanced Computer Lab, Winter 2026 — Team 121**

| Member | ID |
|---|---|
| Omar Yasser | 58-1906 |
| Anas Jabr | 58-5506 |
| Omar Elkhayat | 58-9780 |
| Fawzy Kareem | 58-9264 |

**Kaggle notebook:** https://www.kaggle.com/code/omaryasser102/milestone-1-121-acl

## Task

Predict whether a driver finishes on the podium (`positionOrder <= 3`) using only information available **after qualifying and before the race starts**. Each row of the modelling table is one driver entering one race.

## Repository contents

| Path | Description |
|---|---|
| `notebook/milestone-1-121-acl.ipynb` | Run-all notebook with outputs: audit, cleaning, data-engineering questions, feature engineering, modelling, ablation, XAI and the inference function |
| `report/Team121_Milestone1_Report.pdf` | Analytical report |

## Data

[Formula 1 World Championship (1950–2024)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) (Kaggle / Ergast), 14 relational CSV tables. The data is not stored in this repository.

## How to run

1. Open the notebook on Kaggle (or upload `notebook/milestone-1-121-acl.ipynb` to a new Kaggle notebook).
2. Attach the dataset above via **Add Input**.
3. Turn **Internet** on (the first cell installs `lime`) and use a CPU session.
4. Run all cells. The notebook finds the dataset under `/kaggle/input` automatically.

Main libraries: pandas, NumPy, matplotlib, seaborn, scikit-learn, TensorFlow/Keras, SHAP, LIME.

## Approach

- **Audit and cleaning:** `\N` sentinels converted to `NaN` on copies of the raw tables, dates and lap/qualifying/pit-stop times typed, primary and foreign keys verified, and 1950s shared drives resolved to one row per `(raceId, driverId)`.
- **Features (20, all known before the race):** grid and qualifying, driver and constructor form computed from earlier races only, driver profile, race context and circuit attributes.
- **Split by season:** train 1950–2019, validation 2020–2021, test 2022–2024.
- **Models:** logistic regression (baseline), random forest and a shallow Keras neural network, all tuned on validation. The final model is the simplest one that is not significantly worse than the best on validation PR-AUC, using a paired bootstrap.
- **Metrics:** PR-AUC (primary), ROC-AUC (secondary), and F1 at a threshold tuned on validation.

## Results (unseen test set, 2022–2024)

| Model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| **Logistic regression (selected)** | 0.930 | **0.700** | 0.660 |
| Random forest | 0.933 | 0.695 | 0.665 |
| Neural network | 0.920 | 0.659 | 0.631 |

The most important information is the starting grid position, followed by the driver's recent and career form. Explanations (permutation importance, SHAP, LIME) describe what the model relies on and are **not causal effects**.
