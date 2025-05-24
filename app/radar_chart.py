import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def create_radar_chart(student_data, subjects=None):
    """
    Create a radar chart for individual student performance across subjects.
    
    Args:
        student_data (pandas.Series): Single student's data
        subjects (list): List of subjects to include in radar chart
        
    Returns:
        plotly.graph_objects.Figure: Radar chart
    """
    
    if subjects is None:
        subjects = ['Math', 'Science', 'English', 'History', 'ComputerScience']
    
    # Get subject scores
    scores = [student_data[subject] for subject in subjects]
    
    # Create readable labels
    labels = ['Math', 'Science', 'English', 'History', 'Computer Science']
    
    # Determine color based on performance category
    performance_category = student_data.get('PerformanceCategory', 'Medium')
    if performance_category == 'High':
        fill_color = 'rgba(34, 139, 34, 0.3)'  # Green
        line_color = 'rgba(34, 139, 34, 1)'
    elif performance_category == 'Low':
        fill_color = 'rgba(220, 20, 60, 0.3)'  # Red
        line_color = 'rgba(220, 20, 60, 1)'
    else:
        fill_color = 'rgba(255, 165, 0, 0.3)'  # Orange
        line_color = 'rgba(255, 165, 0, 1)'
    
    # Create radar chart
    fig = go.Figure()
    
    # Add student performance trace
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=labels,
        fill='toself',
        name=f"{student_data['FirstName']} {student_data['LastName']}",
        fillcolor=fill_color,
        line=dict(color=line_color, width=3),
        marker=dict(size=8, color=line_color)
    ))
    
    # Add class average for comparison (if available)
    if hasattr(student_data, 'index') and len(student_data.index) > 10:  # Assuming we have class data
        # This would need the full dataframe to calculate class averages
        # For now, we'll add a reference line at 75% (good performance threshold)
        reference_scores = [75] * len(subjects)
        
        fig.add_trace(go.Scatterpolar(
            r=reference_scores,
            theta=labels,
            fill=None,
            name='Target Performance (75%)',
            line=dict(color='gray', width=2, dash='dash'),
            marker=dict(size=4, color='gray')
        ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[20, 40, 60, 80, 100],
                ticktext=['20%', '40%', '60%', '80%', '100%'],
                gridcolor='lightgray',
                gridwidth=1,
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color='black'),
                gridcolor='lightgray',
                gridwidth=1,
                linecolor='black',
                linewidth=2
            ),
            bgcolor='white'
        ),
        showlegend=True,
        title=dict(
            text=f"Academic Performance Profile<br>{student_data['FirstName']} {student_data['LastName']} - Grade {student_data['GradeLevel']}",
            x=0.5,
            font=dict(size=16, color='black')
        ),
        height=500,
        width=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig

def create_multi_student_radar(df, student_ids, subjects=None):
    """
    Create a radar chart comparing multiple students.
    
    Args:
        df (pandas.DataFrame): Student dataset
        student_ids (list): List of student IDs to compare
        subjects (list): List of subjects to include
        
    Returns:
        plotly.graph_objects.Figure: Multi-student radar chart
    """
    
    if subjects is None:
        subjects = ['Math', 'Science', 'English', 'History', 'ComputerScience']
    
    labels = ['Math', 'Science', 'English', 'History', 'Computer Science']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    fig = go.Figure()
    
    for i, student_id in enumerate(student_ids[:6]):  # Limit to 6 students for readability
        student_data = df[df['StudentID'] == student_id].iloc[0]
        scores = [student_data[subject] for subject in subjects]
        
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=labels,
            fill='toself',
            name=f"{student_data['FirstName']} {student_data['LastName']}",
            fillcolor=f'rgba({int(colors[i][1:3], 16)}, {int(colors[i][3:5], 16)}, {int(colors[i][5:7], 16)}, 0.2)',
            line=dict(color=colors[i], width=2),
            marker=dict(size=6, color=colors[i])
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[20, 40, 60, 80, 100],
                gridcolor='lightgray'
            ),
            angularaxis=dict(
                tickfont=dict(size=11),
                gridcolor='lightgray'
            )
        ),
        showlegend=True,
        title="Multi-Student Performance Comparison",
        height=600,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.1
        )
    )
    
    return fig

def create_subject_strength_analysis(student_data, class_averages=None):
    """
    Create a detailed analysis of student's subject strengths and weaknesses.
    
    Args:
        student_data (pandas.Series): Single student's data
        class_averages (dict): Class averages for comparison
        
    Returns:
        plotly.graph_objects.Figure: Strength analysis chart
    """
    
    subjects = ['Math', 'Science', 'English', 'History', 'ComputerScience']
    subject_labels = ['Math', 'Science', 'English', 'History', 'Computer Science']
    scores = [student_data[subject] for subject in subjects]
    
    # Calculate relative performance (compared to student's average)
    student_avg = np.mean(scores)
    relative_performance = [score - student_avg for score in scores]
    
    # Determine colors based on performance
    colors = ['green' if rp > 5 else 'red' if rp < -5 else 'orange' for rp in relative_performance]
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    # Add actual scores
    fig.add_trace(go.Bar(
        y=subject_labels,
        x=scores,
        name='Actual Score',
        orientation='h',
        marker=dict(color=colors, opacity=0.7),
        text=[f'{score:.1f}%' for score in scores],
        textposition='inside',
        textfont=dict(color='white', size=12)
    ))
    
    # Add student average line
    fig.add_vline(
        x=student_avg,
        line=dict(color='blue', width=2, dash='dash'),
        annotation_text=f'Personal Avg: {student_avg:.1f}%'
    )
    
    # Add class averages if provided
    if class_averages:
        class_scores = [class_averages.get(subject, 0) for subject in subjects]
        fig.add_trace(go.Scatter(
            y=subject_labels,
            x=class_scores,
            mode='markers',
            name='Class Average',
            marker=dict(color='black', size=10, symbol='diamond')
        ))
    
    fig.update_layout(
        title=f"Subject Performance Analysis - {student_data['FirstName']} {student_data['LastName']}",
        xaxis_title="Score (%)",
        yaxis_title="Subjects",
        height=400,
        showlegend=True,
        xaxis=dict(range=[0, 100])
    )
    
    return fig

def get_performance_insights(student_data):
    """
    Generate text insights about student performance from radar chart data.
    
    Args:
        student_data (pandas.Series): Single student's data
        
    Returns:
        dict: Performance insights and recommendations
    """
    
    subjects = ['Math', 'Science', 'English', 'History', 'ComputerScience']
    subject_labels = ['Math', 'Science', 'English', 'History', 'Computer Science']
    scores = [student_data[subject] for subject in subjects]
    
    # Calculate statistics
    avg_score = np.mean(scores)
    max_score = max(scores)
    min_score = min(scores)
    max_subject = subject_labels[scores.index(max_score)]
    min_subject = subject_labels[scores.index(min_score)]
    
    # Generate insights
    insights = {
        'overall_performance': f"Overall average: {avg_score:.1f}%",
        'strongest_subject': f"Strongest in {max_subject} ({max_score:.1f}%)",
        'weakest_subject': f"Needs improvement in {min_subject} ({min_score:.1f}%)",
        'performance_gap': f"Performance gap: {max_score - min_score:.1f} points"
    }
    
    # Generate recommendations
    recommendations = []
    
    if max_score - min_score > 20:
        recommendations.append(f"Focus on balancing performance across subjects")
        recommendations.append(f"Allocate more study time to {min_subject}")
    
    if avg_score >= 85:
        recommendations.append("Excellent overall performance! Consider advanced coursework")
    elif avg_score >= 70:
        recommendations.append("Good performance with room for improvement")
    elif avg_score >= 60:
        recommendations.append("Focus on consistent study habits across all subjects")
    else:
        recommendations.append("Requires immediate academic support and intervention")
    
    if min_score < 50:
        recommendations.append(f"Urgent attention needed for {min_subject}")
    
    insights['recommendations'] = recommendations
    
    return insights

def create_performance_trend_radar(student_data, historical_data=None):
    """
    Create a radar chart showing performance trends over time (simulated).
    
    Args:
        student_data (pandas.Series): Current student data
        historical_data (dict): Historical performance data
        
    Returns:
        plotly.graph_objects.Figure: Trend radar chart
    """
    
    subjects = ['Math', 'Science', 'English', 'History', 'ComputerScience']
    labels = ['Math', 'Science', 'English', 'History', 'Computer Science']
    
    fig = go.Figure()
    
    # Current performance
    current_scores = [student_data[subject] for subject in subjects]
    
    fig.add_trace(go.Scatterpolar(
        r=current_scores,
        theta=labels,
        fill='toself',
        name='Current Term',
        fillcolor='rgba(0, 123, 255, 0.3)',
        line=dict(color='rgba(0, 123, 255, 1)', width=3),
        marker=dict(size=8, color='rgba(0, 123, 255, 1)')
    ))
    
    # Simulate previous term performance (slightly lower for demonstration)
    if historical_data is None:
        # Simulate historical data
        np.random.seed(int(student_data.get('StudentID', 1)))
        previous_scores = [max(0, score - np.random.normal(3, 2)) for score in current_scores]
    else:
        previous_scores = [historical_data.get(subject, current_scores[i]) for i, subject in enumerate(subjects)]
    
    fig.add_trace(go.Scatterpolar(
        r=previous_scores,
        theta=labels,
        fill='toself',
        name='Previous Term',
        fillcolor='rgba(255, 193, 7, 0.2)',
        line=dict(color='rgba(255, 193, 7, 1)', width=2, dash='dot'),
        marker=dict(size=6, color='rgba(255, 193, 7, 1)')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[20, 40, 60, 80, 100],
                gridcolor='lightgray'
            ),
            angularaxis=dict(
                tickfont=dict(size=12),
                gridcolor='lightgray'
            )
        ),
        showlegend=True,
        title=f"Performance Trend - {student_data['FirstName']} {student_data['LastName']}",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig