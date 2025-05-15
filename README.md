# Diabetes Prediction and Analysis Notebook (R Version)

This project notebook performs a binary classification analysis to predict diabetes risk based on various health and lifestyle factors from a publicly available dataset. The workflow includes data loading, comprehensive exploratory data analysis, deriving a binary target variable based on clinical guidelines, and training a Logistic Regression model.

## Project Goal

The goal is to build a binary classification model that can predict a person's likelihood of having diabetes based on clinical indicators (specifically, derived from Fasting Blood Glucose) and other health metrics within the dataset. The project also aims to identify which features are most indicative of this risk.

## Analysis Steps

The notebook follows these key steps:

1.  **Data Loading and Initial Inspection:** Load the dataset and perform initial checks for structure and missing values.
2.  **Data Cleaning and Preprocessing:** Handle missing values (explicitly addressed '...1' index column and checked for others), convert categorical features to factors.
3.  **Exploratory Data Analysis (EDA):** Conduct detailed EDA including:
    *   Correlation heatmap for numerical features.
    *   Pair plots for a subset of numerical features to visualize relationships and distributions.
    *   Histograms for key numerical features.
    *   Boxplots for selected features.
4.  **Feature Engineering and Predictor Creation:** Derive a binary 'Diabetes' target variable based on the common clinical threshold for Fasting Blood Glucose (\( > 125 \)).
5.  **Model Building and Evaluation:**
    *   Split the data into training (80%) and testing (20%) sets.
    *   Train a **Logistic Regression** model to predict the derived 'Diabetes' target.
    *   Evaluate model performance on the test set using metrics including **Accuracy**, **Confusion Matrix**, and **ROC Curve / AUC**.
6.  **Model Insights:** Analyze the trained model by computing and visualizing permutation feature importance to understand which variables were most influential in the predictions.

## Data Source

The dataset used in this analysis comes from the **"Diabetes Prediction Dataset"** on Kaggle.

*   **Dataset Name:** Diabetes Prediction Dataset
*   **Source Link:** [https://www.kaggle.com/datasets/marshalpatel3558/diabetes-prediction-dataset](https://www.kaggle.com/datasets/marshalpatel3558/diabetes-prediction-dataset)

The specific file used is `diabetes_dataset.csv`. **Please download this file and place it in the same directory as the R Markdown notebook file (`diabetes_prediction_analysis_notebook.Rmd`) for the code to run correctly.** The notebook includes flexible path handling.

## Key Findings

Based on the analysis using Logistic Regression to predict the *derived* 'Diabetes' variable:

*   **Data Quality:** The dataset (N=10,000) was relatively clean with minimal missing values.
*   **Target Variable Distribution:** The derived 'Diabetes' target showed a distribution of approximately **42.5%** (4254 instances) for "No Diabetes" and **57.5%** (5746 instances) for "Diabetes" in the full dataset.
*   **Model Performance:** The Logistic Regression model achieved exceptionally high performance metrics on the test set:
    *   **Accuracy:** **100%**
    *   **AUC (Area Under the ROC Curve):** **1.00**
    *   The Confusion Matrix showed **0** false positives and **0** false negatives.
*   **Feature Importance:** Permutation importance analysis using the `vip` package identified features such as **Fasting Blood Glucose** and **HbA1c** as having the highest impact on the model's prediction, which aligns with clinical understanding as Fasting Blood Glucose was used to derive the target variable.

## Project Structure

```
.
├── diabetes_dataset.csv                 # Raw data file (must be placed here)
└── diabetes_model.Rmd # The main R Markdown notebook
└── diabetes_model.pdf # The PDF output of the R Markdown notebook
```

*(Note: The Rmd filename is assumed; replace `diabetes_prediction_analysis_notebook.Rmd` if your file has a different name)*

## How to Run the Analysis

To reproduce the analysis and generate the report:

1.  **Install R and RStudio:** If you don't have them, download and install R ([https://cran.r-project.org/](https://cran.r-project.org/)) and RStudio ([https://posit.co/download/rstudio-desktop/](https://posit.co/download/rstudio-desktop/)).
2.  **Download the Data:** Download the `diabetes_dataset.csv` file from the Kaggle dataset link provided above.
3.  **Set up Project Directory:** Create a directory for this project. Place the R Markdown notebook file (e.g., `diabetes_prediction_analysis_notebook.Rmd`) and the downloaded `diabetes_dataset.csv` file directly within this directory.
4.  **Install Required R Packages:** Open RStudio, open the R Markdown file, and install the necessary packages by running the following command in the R console:

    ```R
    install.packages(c("readr", "dplyr", "ggplot2", "reshape2", "caret", "caTools", "pROC", "vip", "ggpairs", "corrplot", "knitr", "rmarkdown", "here"))
    ```

5.  **Run the Analysis:** Open the R Markdown notebook file in RStudio. Click the "Knit" button (usually located in the top toolbar) and select "Knit to PDF".

This will execute the R code chunks within the R Markdown file, perform the entire analysis workflow, and generate a PDF report (e.g., `diabetes_prediction_analysis_notebook.pdf`) containing the detailed steps, visualizations, and results.

## Limitations and Future Work

The exceptionally high predictive performance (100% accuracy, 1.00 AUC) achieved by the Logistic Regression model predicting the *derived* 'Diabetes' target suggests a very strong, possibly trivial, relationship between the features and the target. Since the target was derived directly from the `Fasting_Blood_Glucose` feature using a simple threshold, and `Fasting_Blood_Glucose` is present in the features used for prediction, the model is essentially learning the threshold rule itself. This is not a typical real-world predictive scenario where the target is truly unknown.

Potential future work to make the analysis more clinically relevant and robust could include:

*   **Addressing Potential Data Leakage:** Remove `Fasting_Blood_Glucose` (and possibly `HbA1c`, which is highly correlated with blood glucose) from the predictor variables if aiming to predict diabetes *status* based on other, less directly diagnostic indicators.
*   **Predicting True Diabetes Diagnosis:** Obtain a dataset with a clinically determined 'Diabetes' diagnosis as the target variable, rather than one derived from a single measurement or threshold rule.
*   **Predicting Blood Glucose:** Revert to the initial objective of predicting the continuous `Fasting_Blood_Glucose` level using regression, potentially employing advanced models and hyperparameter tuning as explored in the previous attempt.
*   **Exploring Other Algorithms:** Experiment with classifiers like Random Forest, XGBoost, or SVM if predicting a binary target, or explore other regressors for the continuous target.
*   **Sophisticated Preprocessing:** Implement more advanced techniques for handling categorical variables (e.g., one-hot encoding for models that don't handle factors) and missing values (e.g., imputation).
*   **Subgroup Analysis:** Investigate differences in prediction or feature importance across demographic subgroups (e.g., Sex, Ethnicity).

## Author

*   Manas Reddy

## Acknowledgements

This notebook incorporates ideas based on insights from AI models like Google Gemini 2.5 (2025) and ChatGPT Model 4o (2025).