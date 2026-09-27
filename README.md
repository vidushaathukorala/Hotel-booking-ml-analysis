# Hotel Booking Demand – Machine Learning Analysis

An end-to-end machine learning project on the **Hotel Booking Demand** dataset (119,390 bookings, 32 features). The project predicts booking cancellations, forecasts the average daily rate (ADR) and segments customers. Built as Year 2, Semester 1 coursework for the AI module at APIIT Sri Lanka.

## Business problem

A hotel management company wants to:
1. **Predict cancellations** before arrival, to reduce revenue loss and plan overbooking
2. **Predict ADR (average daily rate)**, to support pricing and revenue forecasting
3. **Segment customers**, for targeted marketing and loyalty programmes

## Notebooks

| Notebook | Task | Methods |
|---|---|---|
| `Task_01_Data_Preprocessing.ipynb` | Data cleaning and feature engineering | Missing-value handling, 6 engineered features (total guests, total nights, total revenue, season, weekend ratio, lead-time category), label encoding, removal of leakage columns, scaling |
| `Task_02_Classification.ipynb` | Cancellation prediction | Logistic Regression, Decision Tree, Random Forest, GridSearchCV, ROC-AUC |
| `Task_03_Regression.ipynb` | ADR prediction | Linear Regression, Decision Tree, Random Forest |
| `Task_04_Clustering.ipynb` | Customer segmentation | K-Means (elbow method), Agglomerative Clustering, DBSCAN |

Tasks 02–04 use scikit-learn `Pipeline` and `ColumnTransformer`, with one-hot encoding for categorical features and standard scaling for numerical ones.

## Results

**Classification (is_canceled)**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.703 | 0.683 | 0.368 | 0.478 |
| Decision Tree | 0.766 | 0.741 | 0.567 | 0.642 |
| Random Forest | 0.765 | 0.725 | 0.587 | 0.649 |

Random Forest ROC-AUC: **0.823**

**Regression (ADR, standardised)**

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 0.737 | 0.532 | 0.393 |
| Decision Tree | 0.751 | 0.433 | 0.370 |
| Random Forest | 0.663 | 0.403 | 0.509 |

**Clustering**

| Model | Silhouette ↑ | Davies-Bouldin ↓ |
|---|---|---|
| K-Means (k=4) | 0.208 | 1.477 |
| Agglomerative | 0.184 | 1.957 |
| DBSCAN | 0.237 | 0.900 |

## How to run

1. Install the libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
2. Unzip `hotel_bookings_cleaned.zip` into the same folder as the notebooks. It contains `hotel_bookings_cleaned 1.csv`, which Tasks 02–04 read.
3. Open the notebooks in Jupyter or Google Colab and run Tasks 02–04.

To re-run Task 01 from the raw data, download `hotel_bookings.csv` from Kaggle (link below) and place it in the same folder.

## Dataset

Hotel Booking Demand, by Antonio, N., de Almeida, A. and Nunes, L. (2019) 'Hotel booking demand datasets', *Data in Brief*, 22, pp. 41–49. It is available on [Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand).

## Tech

Python, pandas, NumPy, scikit-learn, Matplotlib, seaborn, Jupyter / Google Colab

## Author

**Vidusha Athukoralage**, BSc (Hons) Artificial Intelligence undergraduate, APIIT Sri Lanka
