# AI-ML-Internship

Predicting No-Shows for Car Service Appointments
The Business Problem 
At Balancee we face a challenge: customers who book appointments but don't show up. This creates two main problems: 
1. Wasted Resources - When someone books an appointment, the company prepares the resources for repair and allocates staff time. If the customer doesn't show, these resources are wasted. 
2. Lost Revenue Opportunities- Booked time slots are unavailable to other customers. When someone no-shows, the company loses potential revenue from customers who would have actually shown up for the repair. 
The Solution 
The goal was to build a predictive model that identifies which customers are likely to miss their appointments. By predicting no-shows in advance, the company can: 
1.     Send targeted reminders to high-risk customers 
2.     Overbook strategically to compensate for expected no-shows 
3.     Allocate resources more efficiently 
4.     Reduce revenue losses from empty time slots 
This predictive approach helps transform an unpredictable problem into a manageable business process. 
Approach 
The goal is to predict no-shows (no_show: 0 = attended, 1 = no-show) for car service appointments using the dataset provided contains (1200 rows, 21 columns). 
Data Cleaning
1.	How I handled missing values: by imputing numerical columns with the median and categorical columns with the mode.
2.	Converted first_time_user to binary (0 = no, 1 = yes).
3.	Addressed invalid booking_hour entries (assumed imputed with median).
4.	Identified and addressed invalid booking hours and negative lead times
5.	Dropped user_id column as it’s non-predictive.
 
Exploratory Data Analysis (EDA)
1.	Analyzed numerical features using df.describe().
2.	Confirmed balanced classes (~50% no-show, ~50% attended) using countplot, eliminating the need for balancing classes.
3.	Identified numerical and categorical columns for preprocessing.
4.	Used correlation matrix to check for correlated features to avoid multicollinearity
 
Model Selection and Training:
1.	Chose three models: Logistic Regression, Decision Tree, and Random Forest.
2.	Built a pipeline with:
-	Numerical preprocessing: Median imputation, StandardScaler.
-	Categorical preprocessing: Mode imputation, OneHotEncoder.
3.	Tuned hyperparameters using RandomizedSearchCV (20 iterations, 5-fold CV, scoring='precision').
4.	Split data: 80% train, 20% test used a seed to ensure reproducibility
5.	Evaluated models using accuracy, precision, recall, and F1-score .
 
Assumptions
1.	Missing values are missing at random, and imputation (median/mode) does not introduce significant bias.
2.	The no_show variable is accurately labeled.
3.	The dataset is representative of the car service customer population.
4.	RandomizedSearchCV with 20 iterations sufficiently explores the hyperparameter space.
5.	Features like previous_missed_appointments and distance_to_station_km have a direct impact on no-shows.

Key Findings
1.	Class Distribution: The no_show variable is balanced (~50% each class), so low precision is due to model or feature issues, not imbalance.
2.	Model Performance:
-	Logistic Regression: Precision = 0.477, Recall = 0.574, F1 = 0.521, Accuracy = 0.525
Moderate performance but struggles with non-linear patterns.
-	Decision Tree: Precision = 0.443, Recall = 0.648, F1 = 0.526, Accuracy = 0.475
High recall but low precision and accuracy, suggesting overfitting.
-	 Random Forest: Precision = 0.490, Recall = 0.676, F1 = 0.568, Accuracy = 0.538.
Best performer with highest precision, recall, F1, and accuracy, but precision remains below 50%.
 
3.	Insight: Random Forest is the best model due to its ensemble approach, but all models have low precision, indicating many false positives when predicting no-shows.
 
Suggested Improvements
1.	Data Quality:
-	Fix invalid lead_time_days (e.g., negative values) by setting a minimum (e.g., 0).
-	Clean booking_hour by converting to numeric and imputing invalid entries.
-	Collect additional features (e.g., customer demographics, appointment urgency).

2.	Feature Engineering:
-	Create is_weekend (1 for Saturday/Sunday, 0 otherwise) to capture weekend effects.
-	Group booking_hour into time slots (e.g., morning: 6-12, afternoon: 12-18).

3.	Model Tuning:
-	Increase n_iter in RandomizedSearchCV (e.g., 50) for better hyperparameter optimization.
-	Try scoring='f1' to balance precision and recall.
-	Test other models (e.g., XGBoost) for potential improvement.

4.	Business Application:
-	Use Random Forest to flag high-risk no-show customers for reminders or incentives.
-	Develop a simple dashboard showing no-show probabilities and key drivers (e.g., prior no-shows, distance).
 

 

 

