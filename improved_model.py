import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Loading and preparing data...")
# Load the data
df = pd.read_csv('lecture_hall_allocations_simple.csv')

# Feature Engineering
# Create day type (weekday vs weekend)
df['Is_Weekend'] = df['Day'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)

# Create capacity bins
df['Capacity_Bin'] = pd.cut(df['Capacity'], bins=[0, 100, 200, 500, 1000], 
                           labels=['Small', 'Medium', 'Large', 'Extra Large'])

# Create time of day categories
df['Time_Category'] = pd.cut(df['Start_Hour'], 
                            bins=[8, 10, 12, 14, 16, 18], 
                            labels=['Morning', 'Late Morning', 'Noon', 'Afternoon', 'Late Afternoon'])

# Create duration feature
df['Duration'] = df['End_Hour'] - df['Start_Hour']

# Create day of week numerical feature
day_order = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
df['Day_Num'] = df['Day'].map(day_order)

# Prepare data for modeling
# Define features and target
X = df.drop(['Is_Allocated', 'Course_Code', 'Group'], axis=1)
y = df['Is_Allocated']

# Define categorical and numerical features
categorical_features = ['Floor', 'Hall_No', 'Day', 'Capacity_Bin', 'Time_Category']
numerical_features = ['Capacity', 'Start_Hour', 'End_Hour', 'Is_Tutorial', 'Is_Weekend', 'Duration', 'Day_Num']

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Addressing class imbalance with SMOTE...")
# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(preprocessor.fit_transform(X_train), y_train)

# Create and evaluate multiple models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Neural Network': MLPClassifier(max_iter=1000, random_state=42)
}

print("Training and evaluating multiple models...")
# Evaluate models with cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    print(f"Training {name}...")
    # Train the model
    model.fit(X_train_resampled, y_train_resampled)
    
    # Predict on test set
    X_test_transformed = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_transformed)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Store results
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'report': report,
        'predictions': y_pred
    }
    
    print(f"{name} - Accuracy: {accuracy:.4f}")
    print(f"{name} - Classification Report:")
    print(classification_report(y_test, y_pred))

# Find the best model
best_model_name = max(results, key=lambda k: results[k]['report']['1']['f1-score'])
best_model = results[best_model_name]['model']
print(f"\nBest model: {best_model_name} with F1-score: {results[best_model_name]['report']['1']['f1-score']:.4f}")

# Fine-tune the best model with GridSearchCV
print(f"\nFine-tuning {best_model_name} with GridSearchCV...")

if best_model_name == 'Logistic Regression':
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga']
    }
    grid_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    
elif best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_model = RandomForestClassifier(class_weight='balanced', random_state=42)
    
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10]
    }
    grid_model = GradientBoostingClassifier(random_state=42)
    
else:  # Neural Network
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    }
    grid_model = MLPClassifier(max_iter=1000, random_state=42)

# Perform grid search
grid_search = GridSearchCV(grid_model, param_grid, cv=cv, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resampled)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Evaluate the best model from grid search
best_grid_model = grid_search.best_estimator_
X_test_transformed = preprocessor.transform(X_test)
y_pred_grid = best_grid_model.predict(X_test_transformed)

print("\nFinal Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_grid))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_grid))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_grid))

# Plot ROC curve
if hasattr(best_grid_model, "predict_proba"):
    y_score = best_grid_model.predict_proba(X_test_transformed)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()
    
    # Plot Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig('precision_recall_curve.png')
    plt.close()

# Create a pipeline with the preprocessor and the best model
final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', best_grid_model)
])

# Train the final pipeline on the entire dataset
print("\nTraining final model on the entire dataset...")
X_resampled, y_resampled = smote.fit_resample(preprocessor.fit_transform(X), y)
final_pipeline.fit(X, y)  # Using original data for the final model

# Save the final model
joblib.dump(final_pipeline, 'improved_lecture_hall_model.pkl')
print("Improved model saved as 'improved_lecture_hall_model.pkl'")

# Create a function for making predictions
def predict_allocation(floor, hall_no, capacity, day, start_hour, end_hour, is_tutorial=0):
    # Create a DataFrame with the input data
    data = {
        'Floor': [floor],
        'Hall_No': [hall_no],
        'Capacity': [capacity],
        'Day': [day],
        'Start_Hour': [start_hour],
        'End_Hour': [end_hour],
        'Is_Tutorial': [is_tutorial],
        'Is_Weekend': [1 if day in ['Saturday', 'Sunday'] else 0],
        'Duration': [end_hour - start_hour],
        'Day_Num': [day_order.get(day, 0)]
    }
    
    # Create capacity bin
    if capacity <= 100:
        capacity_bin = 'Small'
    elif capacity <= 200:
        capacity_bin = 'Medium'
    elif capacity <= 500:
        capacity_bin = 'Large'
    else:
        capacity_bin = 'Extra Large'
    
    data['Capacity_Bin'] = [capacity_bin]
    
    # Create time category
    if start_hour < 10:
        time_category = 'Morning'
    elif start_hour < 12:
        time_category = 'Late Morning'
    elif start_hour < 14:
        time_category = 'Noon'
    elif start_hour < 16:
        time_category = 'Afternoon'
    else:
        time_category = 'Late Afternoon'
    
    data['Time_Category'] = [time_category]
    
    # Create DataFrame
    df_pred = pd.DataFrame(data)
    
    # Make prediction
    prediction = final_pipeline.predict(df_pred)
    probability = final_pipeline.predict_proba(df_pred)
    
    return {
        'allocation_predicted': bool(prediction[0]),
        'probability': probability[0][1]
    }

# Example usage
print("\nExample prediction with improved model:")
result = predict_allocation(
    floor='B1 Floor',
    hall_no='C2-L101',
    capacity=175,
    day='Monday',
    start_hour=9.0,
    end_hour=10.0
)
print(result)
