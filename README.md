# ML Assignment 2 — Classification Models on Breast Cancer Dataset

## a. Problem Statement

The goal of this project is to build and compare multiple machine learning classification models for **breast cancer diagnosis** — predicting whether a tumor is **Malignant** or **Benign** based on features computed from digitized images of fine needle aspirate (FNA) of a breast mass.

Early and accurate diagnosis of breast cancer is critical in clinical practice. This project explores how different ML algorithms perform on this task, evaluating them across six standard metrics to understand their strengths and weaknesses.

---

## b. Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic)  
**Source:** UCI Machine Learning Repository  
**Link:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic  

| Property | Value |
|---|---|
| Total Instances | 569 |
| Number of Features | 30 (all numeric, continuous) |
| Target Variable | diagnosis (1 = Benign, 0 = Malignant) |
| Class Split | 357 Benign, 212 Malignant |
| Missing Values | None |

The 30 features are computed from ten real-valued measurements of each cell nucleus present in the FNA image:

1. Radius (mean of distances from center to perimeter)
2. Texture (standard deviation of gray-scale values)
3. Perimeter
4. Area
5. Smoothness (local variation in radius lengths)
6. Compactness (perimeter² / area - 1.0)
7. Concavity (severity of concave portions)
8. Concave points (number of concave portions)
9. Symmetry
10. Fractal dimension ("coastline approximation" - 1)

For each of these 10 measurements, the **mean**, **standard error**, and **worst (largest)** values were computed, giving 30 features total.

**Train/Test Split:** 80/20 stratified split (random_state=42)  
- Training set: 455 samples  
- Test set: 114 samples  

---

## c. GitHub Repository Link

**Repository:** [https://github.com/Meghs123/ml-assignment-2-2025AC05122.git](https://github.com/Meghs123/ml-assignment-2-2025AC05122.git)

**Repository Contents:**
```
ml-assignment-2/
│── app.py                    # Streamlit web application
│── requirements.txt          # Python dependencies
│── README.md                 # This file
│── test_data.csv             # Test split data (114 samples)
│── model/
│   ├── train_models.py       # Training script for all 5 models
│   ├── scaler.pkl            # Fitted StandardScaler
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   └── model_results.csv     # Saved evaluation results
```

---

## d. Models Used

Five classification models were trained on the Breast Cancer Wisconsin dataset. All features were standardized using `StandardScaler` before training. Below is the comparison table with all six evaluation metrics.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| kNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) | 0.9561 | 0.9929 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

### Hyperparameters Used

- **Logistic Regression:** `C=1.0`, `solver='lbfgs'`, `max_iter=5000`
- **Decision Tree:** `max_depth=5`, `min_samples_split=5`
- **kNN:** `n_neighbors=7`, `weights='uniform'`, `metric='minkowski'`
- **Naive Bayes:** Gaussian (default parameters)
- **Random Forest:** `n_estimators=150`, `max_depth=None`

---

### Model Performance Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performer on this dataset. Achieved the highest accuracy (98.25%) and F1 score (0.9861) with only 2 misclassifications out of 114 test samples. The near-perfect AUC of 0.9954 confirms that the scaled features create a nearly linearly separable space. Feature standardization was key — without it, the model struggled to converge. This result shows that when the underlying relationship between features and target is approximately linear, simple models can outperform more complex ones. |
| Decision Tree | Weakest performer among the five models. Despite limiting the tree depth to 5 to control overfitting, it produced 9 misclassifications. The axis-aligned decision boundaries don't capture the diagonal relationships in this 30-dimensional space efficiently. The AUC of 0.9163 is noticeably lower than the others, meaning the probability ranking of samples is less reliable. However, it offers full interpretability — you can trace exactly why each prediction was made, which has clinical value. |
| kNN | Second-best model overall and the only one to achieve perfect recall of 1.0 for the benign class, meaning every truly benign case was correctly identified. This happened because the 7 nearest neighbors in the standardized feature space cluster clearly for benign samples. The 3 false positives (malignant cases predicted as benign) are a potential concern in a medical setting. The choice of k=7 (tested k=3, 5, 7) balanced bias and variance well for this dataset. |
| Naive Bayes | Performed reasonably well but is held back by the conditional independence assumption. Many features in this dataset are highly correlated — for example, radius_mean, perimeter_mean, and area_mean essentially measure the same property at different scales. Despite this theoretical limitation, the AUC is quite high (0.9868), showing that the probability estimates from Naive Bayes are well-calibrated even when the hard class predictions are less accurate. It produced 8 errors total. |
| Random Forest (Ensemble) | Solid mid-tier performance. The ensemble of 150 decision trees reduces the variance that hurts the single Decision Tree, boosting accuracy from 92.11% to 95.61%. The AUC of 0.9929 is nearly as good as Logistic Regression, showing that the ensemble's probability estimates are excellent. However, the hard predictions (5 errors) didn't quite match Logistic Regression's performance. Random Forest also offers feature importance scores, which showed that worst_perimeter, worst_concave_points, and mean_concave_points are the top 3 most important features. |
| **Overall Winner** | **Logistic Regression** is the clear winner for this dataset. It achieves the highest scores across all six metrics. The Breast Cancer Wisconsin dataset's features — once standardized — form a nearly linearly separable space, which is ideal for logistic regression. This is a practical reminder that model complexity doesn't always translate to better performance. For clinical deployment, Logistic Regression also has the advantage of interpretability through its coefficients, making it easier for doctors to understand the model's reasoning. |

---

## How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/Meghs123/ml-assignment-2-2025AC05122.git
   cd ml-assignment-2
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. (Optional) Run the training script separately and it would provide .pkl files for each model during training:
   ```bash
   cd model
   python train_models.py
   ```

---

## Live Streamlit App

**Deployed App:** []()

---

## Technologies Used

- Python 3.10
- Streamlit (web app framework)
- Scikit-learn (ML models and metrics)
- Pandas & NumPy (data handling)
- Matplotlib & Seaborn (visualization)
