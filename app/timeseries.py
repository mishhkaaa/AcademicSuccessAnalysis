# app/timeseries.py
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def simulate_terms(df, n_terms=3):
    """Add simulated term-wise subject columns"""
    np.random.seed(42)  # For reproducibility
    
    for term in range(1, n_terms+1):
        for subject in ['Math', 'Science', 'English', 'History', 'ComputerScience']:
            # Simulate performance with gradual improvement/decline
            noise = np.random.normal(loc=0, scale=5, size=len(df))
            df[f'{subject}_T{term}'] = np.clip(df[subject] + noise, 0, 100)
    
    return df

def create_term_radar(student_data, terms_to_show=3):
    """Radar chart comparing performance across terms"""
    subjects = ['Math', 'Science', 'English', 'History', 'ComputerScience']
    
    fig = go.Figure()
    
    for term in range(1, terms_to_show+1):
        term_scores = [student_data[f'{subj}_T{term}'].values[0] for subj in subjects]
        fig.add_trace(go.Scatterpolar(
            r=term_scores + [term_scores[0]],  # Close the radar
            theta=subjects + [subjects[0]],
            name=f'Term {term}',
            fill='toself'
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title="Performance Across Terms"
    )
    return fig

def create_term_line_chart(student_data):
    """Line chart showing progression over terms"""
    term_cols = [col for col in student_data.columns if '_T' in col]
    subjects = list({col.split('_')[0] for col in term_cols})
    
    fig = go.Figure()
    
    for subject in subjects:
        term_scores = [student_data[f'{subject}_T{i+1}'].values[0] 
                      for i in range(len(term_cols)//5)]
        fig.add_trace(go.Scatter(
            x=[f'Term {i+1}' for i in range(len(term_scores))],
            y=term_scores,
            mode='lines+markers',
            name=subject
        ))
    
    fig.update_layout(
        title="Subject Performance Trend",
        xaxis_title="Academic Term",
        yaxis_title="Score (%)",
        height=400
    )
    return fig
