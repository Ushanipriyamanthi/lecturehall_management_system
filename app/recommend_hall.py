
import pandas as pd
import joblib

def recommend_hall(required_capacity, day, start_time, end_time, is_tutorial=0, prefer_floor=None):
    """
    Recommend a lecture hall based on requirements and availability.
    
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
    # Load the recommender
    recommender = joblib.load('lecture_hall_recommender.pkl')
    
    # Get recommendation
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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Recommend lecture halls')
    parser.add_argument('--capacity', type=int, required=True, help='Required capacity')
    parser.add_argument('--day', type=str, required=True, help='Day of the week')
    parser.add_argument('--start', type=float, required=True, help='Start time (e.g., 9.0 for 9 AM)')
    parser.add_argument('--end', type=float, required=True, help='End time (e.g., 10.0 for 10 AM)')
    parser.add_argument('--tutorial', type=int, default=0, help='Is it a tutorial? (0 or 1)')
    parser.add_argument('--floor', type=str, help='Preferred floor')
    
    args = parser.parse_args()
    
    result = recommend_hall(
        required_capacity=args.capacity,
        day=args.day,
        start_time=args.start,
        end_time=args.end,
        is_tutorial=args.tutorial,
        prefer_floor=args.floor
    )
    
    print(f"Recommended Hall: {result['floor']} - {result['hall_no']}")
    print(f"Capacity: {result['capacity']}")
    print(f"Utilization Rate: {result['utilization_rate']:.2f}%")
    if result['availability_probability']:
        print(f"Availability Probability: {result['availability_probability']:.2f}%")
    print(f"Total Time Slots: {result['total_slots']}")
    print(f"Currently Allocated Slots: {result['allocated_slots']}")
