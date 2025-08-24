import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load the data
df = pd.read_csv('lecture_hall_allocations_simple.csv')

# Display basic information
print("Dataset shape:", df.shape)
print("\nSample data:")
print(df.head())

# Data exploration
print("\nMissing values:")
print(df.isnull().sum())

print("\nDistribution of target variable (Is_Allocated):")
print(df['Is_Allocated'].value_counts())
print(df['Is_Allocated'].value_counts(normalize=True))

# Create some visualizations
plt.figure(figsize=(12, 6))
sns.countplot(x='Day', hue='Is_Allocated', data=df)
plt.title('Allocation by Day of Week')
plt.savefig('allocation_by_day.png')
plt.close()

plt.figure(figsize=(12, 6))
sns.countplot(x='Floor', hue='Is_Allocated', data=df)
plt.title('Allocation by Floor')
plt.savefig('allocation_by_floor.png')
plt.close()

# Create time of day bins for better visualization
df['Time_Bin'] = pd.cut(df['Start_Hour'], bins=[8, 10, 12, 14, 16, 18], 
                        labels=['8-10 AM', '10-12 PM', '12-2 PM', '2-4 PM', '4-6 PM'])

plt.figure(figsize=(12, 6))
sns.countplot(x='Time_Bin', hue='Is_Allocated', data=df)
plt.title('Allocation by Time of Day')
plt.savefig('allocation_by_time.png')
plt.close()

# Feature Engineering
# Create day type (weekday vs weekend)
df['Is_Weekend'] = df['Day'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)

# Create capacity bins
df['Capacity_Bin'] = pd.cut(df['Capacity'], bins=[0, 100, 200, 500, 1000], 
                           labels=['Small', 'Medium', 'Large', 'Extra Large'])

# Prepare data for modeling
# Define features and target
X = df.drop(['Is_Allocated', 'Course_Code', 'Group'], axis=1)
y = df['Is_Allocated']

# Define categorical and numerical features
categorical_features = ['Floor', 'Hall_No', 'Day', 'Capacity_Bin']
numerical_features = ['Capacity', 'Start_Hour', 'End_Hour', 'Is_Tutorial', 'Is_Weekend']

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

# Create and train the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_names = (
    numerical_features + 
    list(model.named_steps['preprocessor']
         .named_transformers_['cat']
         .named_steps['onehot']
         .get_feature_names_out(categorical_features))
)

# Get feature importances if the model is trained successfully
importances = model.named_steps['classifier'].feature_importances_
indices = np.argsort(importances)[::-1]

# Print feature ranking
print("\nFeature ranking:")
for f in range(min(10, len(feature_names))):
    if f < len(indices):
        print(f"{f + 1}. {feature_names[indices[f]]} ({importances[indices[f]]})")

# Plot feature importances
plt.figure(figsize=(12, 6))
plt.title("Feature importances")
plt.bar(range(min(10, len(indices))), 
        [importances[i] for i in indices[:10]],
        align="center")
plt.xticks(range(min(10, len(indices))), 
           [feature_names[i] for i in indices[:10]], rotation=90)
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Save the model
joblib.dump(model, 'lecture_hall_allocation_model.pkl')
print("\nModel saved as 'lecture_hall_allocation_model.pkl'")

# Create a function for making predictions on new data
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
        'Is_Weekend': [1 if day in ['Saturday', 'Sunday'] else 0]
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
    
    # Create DataFrame
    df_pred = pd.DataFrame(data)
    
    # Make prediction
    prediction = model.predict(df_pred)
    probability = model.predict_proba(df_pred)
    
    return {
        'allocation_predicted': bool(prediction[0]),
        'probability': probability[0][1]
    }

# Example usage
print("\nExample prediction:")
result = predict_allocation(
    floor='B1 Floor',
    hall_no='C2-L101',
    capacity=175,
    day='Monday',
    start_hour=9.0,
    end_hour=10.0
)
print(result)

