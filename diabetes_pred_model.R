# Load necessary libraries
library(readr)
library(dplyr)
library(ggplot2)
library(reshape2)
library(caret) # For train/test split
library(lightgbm) # For LightGBM model
library(corrplot) # For correlation plot

# Load the dataset
df <- read_csv('diabetes_dataset.csv')

# Display the first few rows
print(head(df))

# Check for missing values
print(colSums(is.na(df)))

# Get information about the data frame
print(str(df))

# Plot histograms
# Base R hist() requires numeric data. Categorical columns are plotted with pie charts or bar plots.
hist_cols <- c('Age', 'Dietary_Intake_Calories', 'Fasting_Blood_Glucose', 'BMI')
par(mfrow=c(2, 2)) # Adjust grid size as we have fewer histograms
for (col in hist_cols) {
  hist(df[[col]], main = paste(col, 'Distribution'), xlab = col, probability = TRUE, col = 'lightblue', border = 'black')
  lines(density(df[[col]], na.rm = TRUE), col = 'blue', lwd = 2)
}
par(mfrow=c(1, 1)) # Reset plot layout


# Plot pie charts
pie_cols <- c('Alcohol_Consumption', 'Sex', 'Ethnicity', 'Smoking_Status', 'Physical_Activity_Level', 'Previous_Gestational_Diabetes')
par(mfrow=c(2, 3)) # Arrange plots in a 2x3 grid
for (col in pie_cols) {
  counts <- table(df[[col]])
  pie(counts, main = paste('Pie Plot of', col), col = rainbow(length(counts)))
}
par(mfrow=c(1, 1)) # Reset plot layout


# Fill missing values in 'Alcohol_Consumption'
df$Alcohol_Consumption[is.na(df$Alcohol_Consumption)] <- 'Not Reported'
print(colSums(is.na(df)))

# Ordinal Encoding
# In R, factors are commonly used for categorical data and are often automatically
# treated as numeric in models. For explicit ordinal encoding similar to Python's
# OrdinalEncoder, we can convert to factors with ordered levels and then to numeric.
# However, for tree-based models like LightGBM, often factor conversion is sufficient.
# Let's convert the specified columns to factors.
factor_cols <- c('Sex', 'Smoking_Status', 'Ethnicity', 'Alcohol_Consumption', 'Physical_Activity_Level', 'Previous_Gestational_Diabetes')
for (col in factor_cols) {
  df[[col]] <- as.factor(df[[col]])
}

# For columns already numeric but treated as categorical in the Python code,
# we will not apply factor conversion here to keep them as numeric for regression.
# The ordinal encoding in the Python code was applied to these columns after filling NA.
# Since the R LightGBM implementation can handle factors, we will use factors for
# the categorical columns.

print(str(df))

# Plot boxplots
boxplot_cols <- c('Age', 'Dietary_Intake_Calories', 'Fasting_Blood_Glucose', 'BMI', 'Alcohol_Consumption', 'Previous_Gestational_Diabetes')
par(mfrow=c(2, 3)) # Arrange plots in a 2x3 grid
for (col in boxplot_cols) {
  boxplot(df[[col]], main = paste('Box Plot of', col), ylab = col)
}
par(mfrow=c(1, 1)) # Reset plot layout

# Calculate and plot correlation heatmap
# Select only numeric columns for correlation
numeric_df <- df %>% select_if(is.numeric)
corr_matrix <- cor(numeric_df, use = "complete.obs") # Handle potential NAs in numeric columns for correlation
corrplot(corr_matrix, method = "color", type = "full", order = "hclust",
         tl.col = "black", tl.srt = 45, addCoef.col = "black", number.cex = 0.7,
         main = "Correlation Heatmap")


# Prepare data for LightGBM
# Drop 'Unnamed: 0' and 'Fasting_Blood_Glucose' from features
X <- df %>% select(-`...1`, -Fasting_Blood_Glucose)
y <- df$Fasting_Blood_Glucose

# Split data into training and testing sets
set.seed(42) # for reproducibility
train_indices <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_indices, ]
X_test <- X[-train_indices, ]
y_train <- y[train_indices]
y_test <- y[-train_indices]

print(paste("Number of training samples:", nrow(X_train)))
print(paste("Number of training labels:", length(y_train)))

# Train LightGBM regression model
# LightGBM in R uses lgb.Dataset and lgb.train
# Identify categorical features for LightGBM
categorical_features_names <- c('Sex', 'Ethnicity', 'Physical_Activity_Level', 'Alcohol_Consumption', 'Smoking_Status', 'Previous_Gestational_Diabetes')
# Get 0-based indices of these columns in the X_train data frame
# Need to adjust for potential column reordering if using select in dplyr earlier
# A more robust way is to get column names from X_train after splitting
current_X_train_cols <- colnames(X_train)
categorical_feature_indices <- which(current_X_train_cols %in% categorical_features_names) - 1 # -1 for 0-based index

dtrain <- lgb.Dataset(data = as.matrix(X_train), label = y_train,
                      categorical_feature = categorical_feature_indices)

# Define parameters
params <- list(objective = "regression", metric = "l2")

model <- lgb.train(params = params,
                   data = dtrain,
                   nrounds = 100 # You might need to tune nrounds
                   # Removed: categorical_feature = categorical_feature_indices
)

# Evaluate the model (R-squared calculation remains the same)
predictions <- predict(model, as.matrix(X_test))

# Calculate R-squared
ssr <- sum((y_test - predictions)^2)
sst <- sum((y_test - mean(y_test))^2)
r_squared <- 1 - (ssr / sst)

print(paste("R-squared score on the test set:", r_squared))

