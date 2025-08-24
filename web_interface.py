from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import joblib
import os
from hall_recommender import LectureHallRecommender

app = Flask(__name__)

# Load models
model_path = 'improved_lecture_hall_model.pkl'
recommender_path = 'lecture_hall_recommender.pkl'

# Check if models exist
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None

if os.path.exists(recommender_path):
    recommender = joblib.load(recommender_path)
else:
    recommender = None

# Load data for populating dropdowns
try:
    df = pd.read_csv('lecture_hall_allocations_final.csv')
    floors = sorted(df['Floor'].unique())
    hall_nos = sorted(df['Hall_No'].unique())
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    time_slots = sorted([(i, f"{i:02d}:00") for i in range(8, 18)])  # 8 AM to 5 PM
except:
    floors = []
    hall_nos = []
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    time_slots = [(i, f"{i:02d}:00") for i in range(8, 18)]

# Helper function to convert time string to float
def time_to_float(time_str):
    try:
        hour = int(time_str.split(':')[0])
        return float(hour)
    except:
        return 0.0

@app.route('/')
def index():
    return render_template('index.html', 
                          floors=floors, 
                          hall_nos=hall_nos, 
                          days=days,
                          time_slots=time_slots)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded. Please train the model first.'})
    
    # Get values from the form
    floor = request.form.get('floor')
    hall_no = request.form.get('hall_no')
    capacity = float(request.form.get('capacity', 0))
    day = request.form.get('day')
    start_time = time_to_float(request.form.get('start_time', '9:00'))
    end_time = time_to_float(request.form.get('end_time', '10:00'))
    is_tutorial = int(request.form.get('is_tutorial', 0))
    
    # Day order mapping for additional features
    day_order = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    
    # Create data for prediction
    data = {
        'Floor': [floor],
        'Hall_No': [hall_no],
        'Capacity': [capacity],
        'Day': [day],
        'Start_Hour': [start_time],
        'End_Hour': [end_time],
        'Is_Tutorial': [is_tutorial],
        'Is_Weekend': [1 if day in ['Saturday', 'Sunday'] else 0],
        'Duration': [end_time - start_time],
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
    if start_time < 10:
        time_category = 'Morning'
    elif start_time < 12:
        time_category = 'Late Morning'
    elif start_time < 14:
        time_category = 'Noon'
    elif start_time < 16:
        time_category = 'Afternoon'
    else:
        time_category = 'Late Afternoon'
    
    data['Time_Category'] = [time_category]
    
    # Create DataFrame for prediction
    df_pred = pd.DataFrame(data)
    
    try:
        # Make prediction
        prediction = model.predict(df_pred)
        probability = model.predict_proba(df_pred)[0][1]  # Probability of being allocated
        
        return jsonify({
            'prediction': bool(prediction[0]),
            'probability': float(probability),
            'message': 'Allocated' if prediction[0] else 'Available',
            'probability_percent': f"{probability * 100:.2f}%"
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/recommend', methods=['POST'])
def recommend():
    if not recommender:
        return jsonify({'error': 'Recommender not loaded. Please create the recommender first.'})
    
    # Get values from the form
    required_capacity = float(request.form.get('required_capacity', 0))
    day = request.form.get('day')
    start_time = time_to_float(request.form.get('start_time', '9:00'))
    end_time = time_to_float(request.form.get('end_time', '10:00'))
    is_tutorial = int(request.form.get('is_tutorial', 0))
    prefer_floor = request.form.get('prefer_floor')
    
    if prefer_floor == "":
        prefer_floor = None
    
    try:
        # Get recommendation
        hall, prob = recommender.recommend_hall(
            required_capacity=required_capacity,
            day=day,
            start_time=start_time,
            end_time=end_time,
            is_tutorial=is_tutorial,
            prefer_floor=prefer_floor
        )
        
        recommendation = {
            'floor': hall['Floor'],
            'hall_no': hall['Hall_No'],
            'capacity': float(hall['Capacity']),
            'utilization_rate': float(hall['Utilization_Rate'] * 100),
            'availability_probability': float(prob * 100) if prob else None,
            'total_slots': int(hall['Total_Slots']),
            'allocated_slots': int(hall['Allocated_Slots'])
        }
        
        return jsonify(recommendation)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/visualizations/<path:filename>')
def visualization_file(filename):
    return send_from_directory('visualizations', filename)

@app.route('/visualizations')
def visualizations():
    # Get list of visualization files
    viz_dir = 'visualizations'
    if os.path.exists(viz_dir):
        visualizations = [f for f in os.listdir(viz_dir) if f.endswith(('.png', '.html'))]
    else:
        visualizations = []
    
    return render_template('visualizations.html', visualizations=visualizations)

@app.route('/dashboard')
def dashboard():
    # Check if dashboard HTML files exist
    dashboard_file = 'visualizations/utilization_dashboard.html'
    heatmap_file = 'visualizations/heatmap_dashboard.html'
    
    has_dashboard = os.path.exists(dashboard_file)
    has_heatmap = os.path.exists(heatmap_file)
    
    return render_template('dashboard.html', 
                          has_dashboard=has_dashboard,
                          has_heatmap=has_heatmap)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    
    app.run(debug=True, port=5000)
