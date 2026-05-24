# Student Performance Prediction Project

This project aims to predict student performance based on various academic, demographic, and social factors using machine learning models. Specifically, it focuses on predicting whether a student will 'Pass' (final grade G3 >= 10) or 'Fail' (final grade G3 < 10) in a mathematics course.

## Project Workflow

The project follows a standard machine learning workflow:

1.  **Environment Setup & Data Loading:** Essential libraries are imported, and the dataset is loaded.
2.  **Data Quality Checks & Initial Exploration:** Basic checks for missing values and duplicate rows are performed.
3.  **Exploratory Data Analysis (EDA):** Key features are visualized to understand their distribution and relationship with the target variable.
4.  **Feature Engineering & Preprocessing:** Categorical variables are converted into numerical formats using one-hot encoding.
5.  **Feature Selection:** Correlations with the target variable `G3` are examined.
6.  **Data Splitting:** The dataset is divided into training and testing sets.
7.  **Model Training:** Logistic Regression and Random Forest Classifier models are trained.
8.  **Model Evaluation:** Models are assessed using accuracy, confusion matrix, and classification reports.
9.  **Model Export:** Trained models are saved for future use.
10. **Streamlit Application:** A user-friendly web application is developed to interact with the trained model.

## Dataset

The dataset used is `student-mat.csv`, which contains student grades and various attributes. It was obtained from KaggleHub.

-   **Source:** UCI Student Performance Dataset by Paulo Cortez
-   **Key Features:** `G1` (first period grade), `G2` (second period grade), `age`, `Medu` (mother's education), `Fedu` (father's education), `traveltime`, `studytime`, `failures`, `internet` access, etc.
-   **Target Variable:** `G3` (final grade), converted to a binary 'Pass' (G3 >= 10) / 'Fail' (G3 < 10) for classification.

## Key Steps and Findings

### 1. Data Loading and Initial Exploration
-   The `student-mat.csv` dataset was loaded into a Pandas DataFrame `df` using `delimiter=';'`.
-   Initial checks confirmed no missing values and no duplicate rows.
-   The dataset contains 395 entries and 33 columns.

### 2. Exploratory Data Analysis (EDA)
-   **Distribution of G3:** A histogram showed the distribution of final grades.
-   **Grades Correlation:** Strong positive correlations were observed between `G1`, `G2`, and `G3` (e.g., G2 and G3 correlation of 0.90).
-   **Impact of Factors on G3:** Box plots revealed insights, such as a negative relationship between previous failures and final grades.
-   **Overall Correlation:** A heatmap displayed correlations across all numerical features.

### 3. Feature Engineering
-   Categorical variables were transformed into numerical representations using **one-hot encoding** (`pd.get_dummies`), resulting in `df_encoded` with 42 features.

### 4. Feature Selection
-   Features highly correlated with `G3` were identified, including `G1`, `G2`, `Medu`, and `failures`.

### 5. Data Splitting
-   The `df_encoded` dataset was split into training (80%) and testing (20%) sets (`X_train`, `X_test`, `y_train`, `y_test`).

### 6. Model Training & Evaluation (Classification)
-   The continuous `G3` variable was converted to a binary target: 'Pass' (G3 >= 10) and 'Fail' (G3 < 10).
-   **Logistic Regression Model:**
    -   Accuracy: **0.92**
    -   A classification report and confusion matrix were generated.
-   **Random Forest Classifier Model:**
    -   Accuracy: **0.92**
    -   A classification report and confusion matrix were also generated.
-   Both models showed comparable performance, with slight differences in false positive/negative rates.

### 7. Model Export
-   Both the trained `LogisticRegression` model and `RandomForestClassifier` model were saved using `joblib` as `logistic_regression_model.joblib` and `random_forest_model.joblib`, respectively.

### 8. Streamlit Application
-   A Streamlit application (`app.py`) was developed to provide an interactive interface for predicting student performance using the trained Logistic Regression model.

## Project Files

-   `_T-bvTtKf35m.ipynb`: The Jupyter notebook containing the full analysis and code.
-   `student_data_cleaned.csv`: The cleaned and preprocessed dataset after one-hot encoding.
-   `preprocess_data.py`: A Python script encapsulating the data loading and one-hot encoding logic.
-   `logistic_regression_model.joblib`: The saved trained Logistic Regression model.
-   `random_forest_model.joblib`: The saved trained Random Forest Classifier model.
-   `app.py`: The Streamlit web application code.
-   `requirements.txt`: A list of Python dependencies required to run the project and the Streamlit app.

## How to Run the Streamlit Application

1.  **Ensure you have the necessary files:** Make sure `app.py`, `requirements.txt`, and `logistic_regression_model.joblib` are in the same directory.

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

    This command will launch the Streamlit application in your web browser. If running in a cloud environment like Google Colab, it will provide a public URL.

## Future Work

-   Hyperparameter tuning for both Logistic Regression and Random Forest models.
-   Exploring other classification algorithms (e.g., SVM, Gradient Boosting).
-   Further in-depth EDA on factors like 'absence impact on grades', 'family support', and 'parental education'.
-   Implementing a feature importance analysis for the Random Forest model.
-   Containerizing the Streamlit application using Docker.
