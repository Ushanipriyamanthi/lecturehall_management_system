import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from hall_recommender import LectureHallRecommender

print("Loading data...")
# Load the data
df = pd.read_csv('lecture_hall_allocations_final.csv')

# Create capacity bins if not already present
if 'Capacity_Bin' not in df.columns:
    df['Capacity_Bin'] = pd.cut(df['Capacity'], bins=[0, 100, 200, 500, 1000], 
                               labels=['Small', 'Medium', 'Large', 'Extra Large'])

# Load the improved model
model = joblib.load('improved_lecture_hall_model.pkl')

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
    
    def recommend_hall(self, required_capacity, day, start_time, end_time, is_tutorial=0, prefer_floor=None):
        """
        Recommend the best hall based on requirements and availability.
        
        Parameters:
            required_capacity (int): The minimum capacity required for the hall
            day (str): Day of the week (Monday, Tuesday, etc.)
            start_time (float): Start time in hours (e.g., 9.0 for 9 AM)
            end_time (float): End time in hours (e.g., 10.0 for 10 AM)
            is_tutorial (int, optional): Whether it's a tutorial session (1) or not (0)
            prefer_floor (str, optional): Preferred floor for the hall
        
        Returns:
            dict: Dictionary containing recommendation details
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
                # If model prediction fails, just return the top hall based on score
                pass
        
        # If no halls are predicted to be available, return the top scored hall
        return recommended_halls.iloc[0], None

# Create the recommender
print("Creating lecture hall recommender...")
recommender = LectureHallRecommender(df)

# Example usage
print("\nExample recommendations:")

# Example 1: Recommend a hall for a class of 50 students
required_capacity = 50
day = 'Monday'
start_time = 10.0
end_time = 11.0
is_tutorial = 0

recommended_hall, availability_prob = recommender.recommend_hall(
    required_capacity=required_capacity,
    day=day,
    start_time=start_time,
    end_time=end_time,
    is_tutorial=is_tutorial
)

print(f"\nRecommendation for a class of {required_capacity} students on {day} from {start_time} to {end_time}:")
print(f"Recommended Hall: {recommended_hall['Floor']} - {recommended_hall['Hall_No']}")
print(f"Capacity: {recommended_hall['Capacity']}")
print(f"Utilization Rate: {recommended_hall['Utilization_Rate']*100:.2f}%")
if availability_prob:
    print(f"Availability Probability: {availability_prob*100:.2f}%")

# Example 2: Recommend a hall for a tutorial of 100 students with floor preference
required_capacity = 100
day = 'Wednesday'
start_time = 14.0
end_time = 15.0
is_tutorial = 1
prefer_floor = 'B1 Floor'

recommended_hall, availability_prob = recommender.recommend_hall(
    required_capacity=required_capacity,
    day=day,
    start_time=start_time,
    end_time=end_time,
    is_tutorial=is_tutorial,
    prefer_floor=prefer_floor
)

print(f"\nRecommendation for a tutorial of {required_capacity} students on {day} from {start_time} to {end_time} (preferring {prefer_floor}):")
print(f"Recommended Hall: {recommended_hall['Floor']} - {recommended_hall['Hall_No']}")
print(f"Capacity: {recommended_hall['Capacity']}")
print(f"Utilization Rate: {recommended_hall['Utilization_Rate']*100:.2f}%")
if availability_prob:
    print(f"Availability Probability: {availability_prob*100:.2f}%")

# Example 3: Find similar halls to a specific hall
floor = 'B1 Floor'
hall_no = 'C2-L101'
similar_halls = recommender.get_similar_halls(floor, hall_no)

print(f"\nHalls similar to {floor} - {hall_no}:")
print(similar_halls[['Floor', 'Hall_No', 'Capacity', 'Utilization_Rate']])

# Create a function for making recommendations
def get_hall_recommendation(required_capacity, day, start_time, end_time, is_tutorial=0, prefer_floor=None):
    """
    Function to get hall recommendations for external use.
    """
    hall, prob = recommender.recommend_hall(
        required_capacity=required_capacity,
        day=day,
        start_time=start_time,
        end_time=end_time,
        is_tutorial=is_tutorial,
        prefer_floor=prefer_floor
    )
    
    return {
        'floor': hall['Floor'],
        'hall_no': hall['Hall_No'],
        'capacity': hall['Capacity'],
        'utilization_rate': hall['Utilization_Rate'] * 100,
        'availability_probability': prob * 100 if prob else None,
        'total_slots': hall['Total_Slots'],
        'allocated_slots': hall['Allocated_Slots']
    }

# Save the recommender to be used by the web interface
print("\nSaving recommender model...")
joblib.dump(recommender, 'lecture_hall_recommender.pkl')


# Create visualizations for the recommender
print("\nCreating recommender visualizations...")

# Visualize hall utilization rates
plt.figure(figsize=(12, 8))
sns.histplot(recommender.halls_df['Utilization_Rate'] * 100, bins=20, kde=True)
plt.title('Distribution of Lecture Hall Utilization Rates')
plt.xlabel('Utilization Rate (%)')
plt.ylabel('Count')
plt.savefig('visualizations/utilization_distribution.png')
plt.close()

# Visualize capacity vs utilization
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='Capacity', 
    y='Utilization_Rate', 
    hue='Floor', 
    size='Total_Slots',
    sizes=(50, 200),
    alpha=0.7,
    data=recommender.halls_df
)
plt.title('Lecture Hall Capacity vs Utilization Rate')
plt.xlabel('Capacity')
plt.ylabel('Utilization Rate')
plt.savefig('visualizations/capacity_vs_utilization.png')
plt.close()

print("Recommendation system analysis complete.")
