# Homeloandefault
Domain Background

An important fraction of the population finds it difficult to get their home loans approved due to insufficient or absent credit history. This prevents them to buy their own dream homes and at times even forces them to rely on other sources of money which may be unreliable and have exorbitant interest rates. Conversely, it is a major challenge for banks and other finance lending agencies to decide for which candidates to approve housing loans. The credit history is not always a sufficient tool for decisions, since it is possible that those borrowers with a long credit history can still default on the loan and some people with a good chance of loan repayment may simply not have a sufficiently long credit history.

A number of recent researchers have applied machine learning to predict the loan default risk. This is important since a machine learning-based classification tool to predict the loan default risk which uses more features than just the traditional credit history can be of great help for both, potential borrowers, and the lending institutions.

Problem Statement

The problem and associated data has been provided by Home Call Credit Group for a Kaggle competition. The problem can be described as, “A binary classification problem where the inputs are various features describing the financial and behavioral history of the loan applicants, in order to predict whether the loan will be repaid or defaulted.”

Project Novelty

The notebook provides a complete end-to-end machine learning workflow for building a binary classifier, and includes methods like automated feature engineering for connecting relational databases, comparison of different classifiers on imbalanced data, and hyperparameter tuning using Bayesian optimization.



<img width="1201" height="771" alt="image" src="https://github.com/user-attachments/assets/9d45355f-246e-49cd-b4d3-21be505ebad6" />


Project Design and Solution

The project has been divided into five parts-

Data Preparation - Before starting the modeling, we need to import the necessary libraries and the datasets. If there are more than one files, then all need to be imported before we can look at the feature types and number of rows/columns in each file.

Exploratory Data Analysis - After data importing, we can investigate the data and answer questions like- How many features are present and how are they interlinked? What is the data quality, are there missing values? What are the different data types, are there many categorical features? Is the data imbalanced? And most importantly, are there any obvious patterns between the predictor and response features?

Feature Engineering - After exploring the data distributions, we can conduct feature engineering to prepare the data for model training. This includes operations like replacing outliers, imputing missing values, one-hot encoding categorical variables, and rescaling the data. Since there are number of relational databases, we can use extract, transform, load (ETL) processes using automated feature Engineering with Featuretools to connect the datasets. The additional features from these datasets will help improve the results over the base case (logistic regression).

Classifier Models: Training, Prediction and Comparison - After the dataset is split into training and testing sets, we can correct the data imbalances by undersampling the majority class. Then, we can training the different classifier models (Logistic Regression, Random Forest, Decision Tree, Gaussian Naive Bayes, XGBoost, Gradient Boosting, LightGBM) and compare their performance on the test data using metrics like accuracy, F1-score and ROC AUC. After choosing the best classifier, we can use K-fold cross validation to select the best model. This will help us choose parameters that correspond to the best performance without creating a separate validation dataset.

Hyperparameter Tuning - After choosing the binary classifier, we can tune the hyperparameters for improving the model results through grid search, random search, and Bayesian optimization (Hypertopt library). The hyperparameter tuning process will use an objective function on the given domain space, and an optimization algorithm to give the results. The ROC AUC validation scores from all three methods for different iterations can be compared to see trends.

Package/Library Requirements

The following packages need to be installed for running the project notebook.

sklearn - For models and metrics
warnings - For preventing warnings
numpy - For basic matrix handling
matplotlib - For figure plotting
pandas - For creating dataframes
seaborn - For figure plotting
timeit - For tracking times
os - for setting work directory
random - For creating random seeds
csv - For saving csv files
json - For creating json files
itertools - For creating iterators for efficient looping
pprint - For pretty printing data structures
pydash - for doing “stuff” in a functional way (utility library).
gc - Garbage collector for deleting data
re - Raw string notation for regular expression patterns
featuretools - Automated feature engineering
xgboost - XGBoost model
lightgbm - LightGBM model
hyperopt - Bayesian hyperparameter optimization
