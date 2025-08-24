
import pandas as pd
import joblib
import numpy as np

def predict_allocation(floor, hall_no, capacity, day, start_hour, end_hour, is_tutorial=0):
    # Load the model
    model = joblib.load('improved_lecture_hall_model.pkl')
    
    # Day order mapping
    day_order = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    
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
    prediction = model.predict(df_pred)
    probability = model.predict_proba(df_pred)
    
    return {
        'allocation_predicted': bool(prediction[0]),
        'probability': probability[0][1]
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict lecture hall allocation with improved model')
    parser.add_argument('--floor', type=str, required=True, help='Floor (e.g., "B1 Floor")')
    parser.add_argument('--hall', type=str, required=True, help='Hall number (e.g., "C2-L101")')
    parser.add_argument('--capacity', type=float, required=True, help='Capacity of the hall')
    parser.add_argument('--day', type=str, required=True, help='Day of the week')
    parser.add_argument('--start', type=float, required=True, help='Start hour (e.g., 9.0 for 9 AM)')
    parser.add_argument('--end', type=float, required=True, help='End hour (e.g., 10.0 for 10 AM)')
    parser.add_argument('--tutorial', type=int, default=0, help='Is it a tutorial? (0 or 1)')
    
    args = parser.parse_args()
    
    result = predict_allocation(
        floor=args.floor,
        hall_no=args.hall,
        capacity=args.capacity,
        day=args.day,
        start_hour=args.start,
        end_hour=args.end,
        is_tutorial=args.tutorial
    )
    
    print(f"Allocation prediction: {'Allocated' if result['allocation_predicted'] else 'Available'}")
    print(f"Probability of allocation: {result['probability']:.2f}")
