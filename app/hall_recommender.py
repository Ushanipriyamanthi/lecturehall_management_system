import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

class LectureHallRecommender:
    def __init__(self, data):
        self.data = data
        self.halls_df = self._prepare_halls_data()
        self.similarity_matrix = self._calculate_similarity()
        
    def _prepare_halls_data(self):
        # Extract unique halls with their attributes
        halls = self.data[['Floor', 'Hall_No', 'Capacity', 'Exam_Capacity']].drop_duplicates()
        
        # Calculate utilization rate for each hall
        utilization = self.data.groupby(['Floor', 'Hall_No'])['Is_Allocated'].mean().reset_index()
        utilization.columns = ['Floor', 'Hall_No', 'Utilization_Rate']
        
        # Merge halls with utilization
        halls_df = pd.merge(halls, utilization, on=['Floor', 'Hall_No'], how='left')
        
        # Calculate additional metrics
        time_slots = self.data.groupby(['Floor', 'Hall_No']).size().reset_index(name='Total_Slots')
        allocated_slots = self.data[self.data['Is_Allocated'] == 1].groupby(['Floor', 'Hall_No']).size().reset_index(name='Allocated_Slots')
        
        # Merge with halls_df
        halls_df = pd.merge(halls_df, time_slots, on=['Floor', 'Hall_No'], how='left')
        halls_df = pd.merge(halls_df, allocated_slots, on=['Floor', 'Hall_No'], how='left')
        halls_df['Allocated_Slots'] = halls_df['Allocated_Slots'].fillna(0)
        
        # Calculate availability score (inverse of utilization)
        halls_df['Availability_Score'] = 1 - halls_df['Utilization_Rate']
        
        # Fill NaN values
        halls_df = halls_df.fillna(0)
        
        return halls_df
    
    def _calculate_similarity(self):
        # Select numerical features for similarity calculation
        features = self.halls_df[['Capacity', 'Exam_Capacity', 'Utilization_Rate', 'Availability_Score']]
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(scaled_features)
        
        return similarity
    
    def get_similar_halls(self, floor, hall_no, top_n=5):
        # Find the index of the target hall
        try:
            idx = self.halls_df[(self.halls_df['Floor'] == floor) & (self.halls_df['Hall_No'] == hall_no)].index[0]
        except IndexError:
            print(f"Hall {hall_no} on {floor} not found.")
            return None
        
        # Get similarity scores
        similarity_scores = list(enumerate(self.similarity_matrix[idx]))
        
        # Sort by similarity
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N similar halls (excluding itself)
        similar_halls_indices = [i[0] for i in similarity_scores[1:top_n+1]]
        
        # Return similar halls
        return self.halls_df.iloc[similar_halls_indices]
    
    def recommend_hall(self, required_capacity, day, start_time, end_time, is_tutorial=0, prefer_floor=None, model=None):
        """
        Recommend the best hall based on requirements and availability.
        """
        # Create a copy of halls dataframe for scoring
        halls_scored = self.halls_df.copy()
        
        # Filter halls that meet the capacity requirement
        halls_scored = halls_scored[halls_scored['Capacity'] >= required_capacity]
        
        # If a preferred floor is specified, prioritize halls on that floor
        if prefer_floor:
            halls_scored['Floor_Match'] = halls_scored['Floor'].apply(lambda x: 1 if x == prefer_floor else 0)
        else:
            halls_scored['Floor_Match'] = 0
        
        # Calculate capacity efficiency (how well the capacity matches the requirement)
        # Lower score for halls that are much larger than needed
        halls_scored['Capacity_Efficiency'] = 1 - ((halls_scored['Capacity'] - required_capacity) / halls_scored['Capacity']).clip(0, 0.5)
        
        # Calculate final score
        # Higher availability, capacity efficiency, and floor match are better
        halls_scored['Final_Score'] = (
            halls_scored['Availability_Score'] * 0.5 +
            halls_scored['Capacity_Efficiency'] * 0.3 +
            halls_scored['Floor_Match'] * 0.2
        )
        
        # Sort by final score
        recommended_halls = halls_scored.sort_values('Final_Score', ascending=False)
        
        # Check if the top recommended halls are likely to be available at the specified time
        if model:
            for _, hall in recommended_halls.iterrows():
                # Prepare data for prediction
                prediction_data = {
                    'Floor': hall['Floor'],
                    'Hall_No': hall['Hall_No'],
                    'Capacity': hall['Capacity'],
                    'Day': day,
                    'Start_Hour': start_time,
                    'End_Hour': end_time,
                    'Is_Tutorial': is_tutorial,
                    'Is_Weekend': 1 if day in ['Saturday', 'Sunday'] else 0
                }
                
                # Add additional features needed for the model
                day_order = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
                prediction_data['Day_Num'] = day_order.get(day, 0)
                prediction_data['Duration'] = end_time - start_time
                
                # Create capacity bin
                if hall['Capacity'] <= 100:
                    capacity_bin = 'Small'
                elif hall['Capacity'] <= 200:
                    capacity_bin = 'Medium'
                elif hall['Capacity'] <= 500:
                    capacity_bin = 'Large'
                else:
                    capacity_bin = 'Extra Large'
                prediction_data['Capacity_Bin'] = capacity_bin
                
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
                prediction_data['Time_Category'] = time_category
                
                # Create DataFrame for prediction
                pred_df = pd.DataFrame([prediction_data])
                
                try:
                    # Use the model to predict if the hall is likely to be available
                    prediction = model.predict(pred_df)
                    probability = model.predict_proba(pred_df)[0][1]  # Probability of being allocated
                    
                    # If prediction is 0 (available), return this hall
                    if prediction[0] == 0:
                        return hall, 1 - probability  # Return availability probability
                except:
                    # If model prediction fails, just continue to the next hall
                    pass
        
        # If no halls are predicted to be available or if model is None, return the top scored hall
        return recommended_halls.iloc[0], None
