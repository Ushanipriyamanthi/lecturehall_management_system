import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta

# Set style for matplotlib
plt.style.use('ggplot')
sns.set_palette("viridis")

print("Loading data...")
# Load the data
df = pd.read_csv('lecture_hall_allocations_final.csv')

# Create day type (weekday vs weekend)
df['Is_Weekend'] = df['Day'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)

# Create capacity bins if not already present
if 'Capacity_Bin' not in df.columns:
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

print("Analyzing utilization patterns...")
# Calculate utilization metrics
total_slots = df.shape[0]
allocated_slots = df[df['Is_Allocated'] == 1].shape[0]
utilization_rate = allocated_slots / total_slots * 100

print(f"Overall utilization rate: {utilization_rate:.2f}%")

# Utilization by day
utilization_by_day = df.groupby('Day')['Is_Allocated'].mean() * 100
utilization_by_day = utilization_by_day.reindex(index=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# Utilization by floor
utilization_by_floor = df.groupby('Floor')['Is_Allocated'].mean() * 100

# Utilization by time
utilization_by_time = df.groupby('Time_Category')['Is_Allocated'].mean() * 100
utilization_by_time = utilization_by_time.reindex(index=['Morning', 'Late Morning', 'Noon', 'Afternoon', 'Late Afternoon'])

# Utilization by capacity
utilization_by_capacity = df.groupby('Capacity_Bin')['Is_Allocated'].mean() * 100
utilization_by_capacity = utilization_by_capacity.reindex(index=['Small', 'Medium', 'Large', 'Extra Large'])

# Utilization heatmap data (Day vs Time)
heatmap_data = df.pivot_table(
    values='Is_Allocated',
    index='Day',
    columns='Time_Category',
    aggfunc='mean'
) * 100

# Reorder rows for heatmap
heatmap_data = heatmap_data.reindex(index=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# Create visualizations directory
import os
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# Create static visualizations with matplotlib/seaborn
print("Creating static visualizations...")

# Utilization by day
plt.figure(figsize=(12, 6))
ax = sns.barplot(x=utilization_by_day.index, y=utilization_by_day.values)
plt.title('Lecture Hall Utilization by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Utilization Rate (%)')
plt.xticks(rotation=45)
for i, v in enumerate(utilization_by_day.values):
    ax.text(i, v + 1, f"{v:.1f}%", ha='center')
plt.tight_layout()
plt.savefig('visualizations/utilization_by_day.png')
plt.close()

# Utilization by floor
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=utilization_by_floor.index, y=utilization_by_floor.values)
plt.title('Lecture Hall Utilization by Floor')
plt.xlabel('Floor')
plt.ylabel('Utilization Rate (%)')
for i, v in enumerate(utilization_by_floor.values):
    ax.text(i, v + 1, f"{v:.1f}%", ha='center')
plt.tight_layout()
plt.savefig('visualizations/utilization_by_floor.png')
plt.close()

# Utilization by time
plt.figure(figsize=(12, 6))
ax = sns.barplot(x=utilization_by_time.index, y=utilization_by_time.values)
plt.title('Lecture Hall Utilization by Time of Day')
plt.xlabel('Time of Day')
plt.ylabel('Utilization Rate (%)')
plt.xticks(rotation=45)
for i, v in enumerate(utilization_by_time.values):
    ax.text(i, v + 1, f"{v:.1f}%", ha='center')
plt.tight_layout()
plt.savefig('visualizations/utilization_by_time.png')
plt.close()

# Utilization by capacity
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=utilization_by_capacity.index, y=utilization_by_capacity.values)
plt.title('Lecture Hall Utilization by Capacity')
plt.xlabel('Capacity')
plt.ylabel('Utilization Rate (%)')
for i, v in enumerate(utilization_by_capacity.values):
    ax.text(i, v + 1, f"{v:.1f}%", ha='center')
plt.tight_layout()
plt.savefig('visualizations/utilization_by_capacity.png')
plt.close()

# Heatmap of utilization (Day vs Time)
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.1f')
plt.title('Lecture Hall Utilization Heatmap (Day vs Time)')
plt.tight_layout()
plt.savefig('visualizations/utilization_heatmap.png')
plt.close()

# Create interactive visualizations with Plotly
print("Creating interactive visualizations...")

# Utilization by day (interactive)
fig = px.bar(
    x=utilization_by_day.index,
    y=utilization_by_day.values,
    labels={'x': 'Day of Week', 'y': 'Utilization Rate (%)'},
    title='Lecture Hall Utilization by Day of Week',
    text=[f"{val:.1f}%" for val in utilization_by_day.values]
)
fig.update_layout(xaxis_title='Day of Week', yaxis_title='Utilization Rate (%)')
fig.write_html('visualizations/utilization_by_day.html')

# Utilization by floor (interactive)
fig = px.bar(
    x=utilization_by_floor.index,
    y=utilization_by_floor.values,
    labels={'x': 'Floor', 'y': 'Utilization Rate (%)'},
    title='Lecture Hall Utilization by Floor',
    text=[f"{val:.1f}%" for val in utilization_by_floor.values]
)
fig.update_layout(xaxis_title='Floor', yaxis_title='Utilization Rate (%)')
fig.write_html('visualizations/utilization_by_floor.html')

# Utilization heatmap (interactive)
fig = px.imshow(
    heatmap_data,
    labels=dict(x="Time of Day", y="Day of Week", color="Utilization Rate (%)"),
    x=heatmap_data.columns,
    y=heatmap_data.index,
    color_continuous_scale='YlGnBu',
    text_auto='.1f'
)
fig.update_layout(title='Lecture Hall Utilization Heatmap')
fig.write_html('visualizations/utilization_heatmap.html')

# Create a dashboard with multiple visualizations
print("Creating dashboard...")
dashboard = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Utilization by Day of Week',
        'Utilization by Floor',
        'Utilization by Time of Day',
        'Utilization by Capacity'
    ),
    specs=[[{'type': 'bar'}, {'type': 'bar'}],
           [{'type': 'bar'}, {'type': 'bar'}]]
)

# Add utilization by day
dashboard.add_trace(
    go.Bar(
        x=utilization_by_day.index,
        y=utilization_by_day.values,
        text=[f"{val:.1f}%" for val in utilization_by_day.values],
        textposition='auto',
        name='Utilization by Day'
    ),
    row=1, col=1
)

# Add utilization by floor
dashboard.add_trace(
    go.Bar(
        x=utilization_by_floor.index,
        y=utilization_by_floor.values,
        text=[f"{val:.1f}%" for val in utilization_by_floor.values],
        textposition='auto',
        name='Utilization by Floor'
    ),
    row=1, col=2
)

# Add utilization by time
dashboard.add_trace(
    go.Bar(
        x=utilization_by_time.index,
        y=utilization_by_time.values,
        text=[f"{val:.1f}%" for val in utilization_by_time.values],
        textposition='auto',
        name='Utilization by Time'
    ),
    row=2, col=1
)

# Add utilization by capacity
dashboard.add_trace(
    go.Bar(
        x=utilization_by_capacity.index,
        y=utilization_by_capacity.values,
        text=[f"{val:.1f}%" for val in utilization_by_capacity.values],
        textposition='auto',
        name='Utilization by Capacity'
    ),
    row=2, col=2
)

# Update layout
dashboard.update_layout(
    title_text='Lecture Hall Utilization Dashboard',
    height=800,
    showlegend=False
)

dashboard.write_html('visualizations/utilization_dashboard.html')

# Create a separate heatmap dashboard
heatmap_dashboard = go.Figure(data=go.Heatmap(
    z=heatmap_data.values,
    x=heatmap_data.columns,
    y=heatmap_data.index,
    colorscale='YlGnBu',
    text=heatmap_data.values.round(1),
    texttemplate='%{text}%',
    textfont={"size": 12}
))

heatmap_dashboard.update_layout(
    title='Lecture Hall Utilization Heatmap (Day vs Time)',
    xaxis_title='Time of Day',
    yaxis_title='Day of Week',
    height=600
)

heatmap_dashboard.write_html('visualizations/heatmap_dashboard.html')

print("Utilization analysis complete. Visualizations saved to 'visualizations' directory.")

# Additional analysis: Find underutilized and overutilized halls
print("\nIdentifying underutilized and overutilized halls...")
hall_utilization = df.groupby(['Floor', 'Hall_No'])['Is_Allocated'].mean() * 100
hall_utilization = hall_utilization.reset_index()
hall_utilization.columns = ['Floor', 'Hall_No', 'Utilization_Rate']

# Sort by utilization rate
underutilized_halls = hall_utilization.sort_values('Utilization_Rate').head(10)
overutilized_halls = hall_utilization.sort_values('Utilization_Rate', ascending=False).head(10)

print("\nTop 10 Underutilized Halls:")
print(underutilized_halls)

print("\nTop 10 Overutilized Halls:")
print(overutilized_halls)

# Save the results
underutilized_halls.to_csv('visualizations/underutilized_halls.csv', index=False)
overutilized_halls.to_csv('visualizations/overutilized_halls.csv', index=False)

# Create visualizations for hall utilization
plt.figure(figsize=(12, 8))
sns.barplot(x='Hall_No', y='Utilization_Rate', hue='Floor', data=overutilized_halls)
plt.title('Top 10 Most Utilized Lecture Halls')
plt.xlabel('Hall Number')
plt.ylabel('Utilization Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/top_utilized_halls.png')
plt.close()

plt.figure(figsize=(12, 8))
sns.barplot(x='Hall_No', y='Utilization_Rate', hue='Floor', data=underutilized_halls)
plt.title('Top 10 Least Utilized Lecture Halls')
plt.xlabel('Hall Number')
plt.ylabel('Utilization Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/least_utilized_halls.png')
plt.close()

print("Hall utilization analysis complete.")
