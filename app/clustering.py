import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

def perform_clustering(df, n_clusters=4, features=None):
    """
    Perform K-Means clustering on student data.
    
    Args:
        df (pandas.DataFrame): Student dataset
        n_clusters (int): Number of clusters
        features (list): Features to use for clustering
        
    Returns:
        tuple: (clustered_df, cluster_centers)
    """
    
    if features is None:
        # Default features for clustering
        features = ['Math', 'Science', 'English', 'History', 'ComputerScience',
                   'AttendancePercentage', 'StudyHoursPerWeek', 'FinalAverage']
    
    # Prepare data for clustering
    X = df[features].copy()
    
    # Handle any missing values
    X = X.fillna(X.mean())
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to the dataframe
    clustered_df = df.copy()
    clustered_df['Cluster'] = cluster_labels
    clustered_df['Cluster'] = clustered_df['Cluster'].astype(str)
    
    # Get cluster centers (in original scale)
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_centers_df = pd.DataFrame(cluster_centers, columns=features)
    cluster_centers_df.index = [f'Cluster {i}' for i in range(n_clusters)]
    
    return clustered_df, cluster_centers_df

def visualize_clusters(clustered_df, method='pca'):
    """
    Visualize clusters using dimensionality reduction.
    
    Args:
        clustered_df (pandas.DataFrame): DataFrame with cluster labels
        method (str): 'pca' or 'tsne' for dimensionality reduction
        
    Returns:
        plotly.graph_objects.Figure: Cluster visualization
    """
    
    features = ['Math', 'Science', 'English', 'History', 'ComputerScience',
               'AttendancePercentage', 'StudyHoursPerWeek', 'FinalAverage']
    
    X = clustered_df[features].fillna(clustered_df[features].mean())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if method == 'pca':
        # Use PCA for dimensionality reduction
        pca = PCA(n_components=2, random_state=42)
        X_reduced = pca.fit_transform(X_scaled)
        
        # Create scatter plot
        fig = px.scatter(
            x=X_reduced[:, 0], 
            y=X_reduced[:, 1],
            color=clustered_df['Cluster'],
            hover_data={
                'Student': clustered_df['FirstName'] + ' ' + clustered_df['LastName'],
                'Final Average': clustered_df['FinalAverage'],
                'Performance Category': clustered_df['PerformanceCategory']
            },
            title=f"Student Clusters (PCA Visualization)<br>Explained Variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}",
            labels={'x': f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%})',
                   'y': f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%})',
                   'color': 'Cluster'}
        )
        
        # Update traces for better visualization
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        
    else:
        # Fallback to simple 2D plot using two important features
        fig = px.scatter(
            clustered_df,
            x='FinalAverage',
            y='AttendancePercentage',
            color='Cluster',
            hover_data=['FirstName', 'LastName', 'StudyHoursPerWeek'],
            title="Student Clusters: Final Average vs Attendance",
            labels={'FinalAverage': 'Final Average (%)',
                   'AttendancePercentage': 'Attendance (%)'}
        )
        
        fig.update_traces(marker=dict(size=10, opacity=0.7))
    
    fig.update_layout(
        height=500,
        showlegend=True,
        legend=dict(title="Cluster Group")
    )
    
    return fig

def analyze_clusters(clustered_df, cluster_centers_df):
    """
    Analyze and describe the characteristics of each cluster.
    
    Args:
        clustered_df (pandas.DataFrame): DataFrame with cluster labels
        cluster_centers_df (pandas.DataFrame): Cluster centers
        
    Returns:
        dict: Cluster analysis results
    """
    
    cluster_analysis = {}
    
    for cluster_id in clustered_df['Cluster'].unique():
        cluster_data = clustered_df[clustered_df['Cluster'] == cluster_id]
        
        analysis = {
            'count': len(cluster_data),
            'avg_final_score': cluster_data['FinalAverage'].mean(),
            'avg_attendance': cluster_data['AttendancePercentage'].mean(),
            'avg_study_hours': cluster_data['StudyHoursPerWeek'].mean(),
            'performance_distribution': cluster_data['PerformanceCategory'].value_counts().to_dict(),
            'grade_distribution': cluster_data['GradeLevel'].value_counts().to_dict(),
            'top_students': cluster_data.nlargest(3, 'FinalAverage')[['FirstName', 'LastName', 'FinalAverage']].to_dict('records')
        }
        
        # Generate cluster description
        if analysis['avg_final_score'] >= 85:
            description = "High Achievers"
        elif analysis['avg_final_score'] >= 70:
            description = "Good Performers"
        elif analysis['avg_final_score'] >= 55:
            description = "Average Students"
        else:
            description = "At-Risk Students"
            
        analysis['description'] = description
        cluster_analysis[f'Cluster {cluster_id}'] = analysis
    
    return cluster_analysis

def create_cluster_profile_chart(cluster_centers_df):
    """
    Create a radar chart showing the profile of each cluster.
    
    Args:
        cluster_centers_df (pandas.DataFrame): Cluster centers
        
    Returns:
        plotly.graph_objects.Figure: Cluster profile radar chart
    """
    
    # Select key features for the radar chart
    features = ['Math', 'Science', 'English', 'History', 'ComputerScience', 'FinalAverage']
    feature_labels = ['Math', 'Science', 'English', 'History', 'Computer Science', 'Final Average']
    
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (cluster_name, row) in enumerate(cluster_centers_df.iterrows()):
        values = [row[feature] for feature in features]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=feature_labels,
            fill='toself',
            name=cluster_name,
            line=dict(color=colors[i % len(colors)], width=2),
            fillcolor=f'rgba({int(colors[i % len(colors)][1:3], 16)}, '
                     f'{int(colors[i % len(colors)][3:5], 16)}, '
                     f'{int(colors[i % len(colors)][5:7], 16)}, 0.2)',
            marker=dict(size=6, color=colors[i % len(colors)])
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
        title=dict(
            text="Cluster Profiles: Average Scores by Group",
            x=0.5,
            font=dict(size=16)
        ),
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

def find_optimal_clusters(df, max_clusters=8, features=None):
    """
    Find the optimal number of clusters using the elbow method and silhouette analysis.
    
    Args:
        df (pandas.DataFrame): Student dataset
        max_clusters (int): Maximum number of clusters to test
        features (list): Features to use for clustering
        
    Returns:
        plotly.graph_objects.Figure: Elbow curve and silhouette scores
    """
    
    if features is None:
        features = ['Math', 'Science', 'English', 'History', 'ComputerScience',
                   'AttendancePercentage', 'StudyHoursPerWeek', 'FinalAverage']
    
    X = df[features].fillna(df[features].mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_clusters + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        inertias.append(kmeans.inertia_)
        sil_score = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(sil_score)
    
    # Create subplot figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Elbow Method', 'Silhouette Analysis'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Elbow curve
    fig.add_trace(
        go.Scatter(x=list(k_range), y=inertias, mode='lines+markers',
                  name='Inertia', line=dict(color='blue', width=3),
                  marker=dict(size=8)),
        row=1, col=1
    )
    
    # Silhouette scores
    fig.add_trace(
        go.Scatter(x=list(k_range), y=silhouette_scores, mode='lines+markers',
                  name='Silhouette Score', line=dict(color='red', width=3),
                  marker=dict(size=8)),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Number of Clusters", row=1, col=1)
    fig.update_xaxes(title_text="Number of Clusters", row=1, col=2)
    fig.update_yaxes(title_text="Inertia", row=1, col=1)
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
    
    fig.update_layout(
        title="Optimal Number of Clusters Analysis",
        height=400,
        showlegend=False
    )
    
    return fig

def create_cluster_comparison_chart(clustered_df):
    """
    Create a comprehensive comparison chart of clusters across multiple metrics.
    
    Args:
        clustered_df (pandas.DataFrame): DataFrame with cluster labels
        
    Returns:
        plotly.graph_objects.Figure: Cluster comparison chart
    """
    
    # Calculate cluster statistics
    cluster_stats = clustered_df.groupby('Cluster').agg({
        'FinalAverage': 'mean',
        'AttendancePercentage': 'mean',
        'StudyHoursPerWeek': 'mean',
        'Math': 'mean',
        'Science': 'mean',
        'English': 'mean',
        'History': 'mean',
        'ComputerScience': 'mean'
    }).round(2)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Final Average by Cluster', 'Attendance by Cluster',
                       'Study Hours by Cluster', 'Subject Averages by Cluster'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Final Average
    fig.add_trace(
        go.Bar(x=cluster_stats.index, y=cluster_stats['FinalAverage'],
               name='Final Average', marker_color=colors[0]),
        row=1, col=1
    )
    
    # Attendance
    fig.add_trace(
        go.Bar(x=cluster_stats.index, y=cluster_stats['AttendancePercentage'],
               name='Attendance %', marker_color=colors[1]),
        row=1, col=2
    )
    
    # Study Hours
    fig.add_trace(
        go.Bar(x=cluster_stats.index, y=cluster_stats['StudyHoursPerWeek'],
               name='Study Hours/Week', marker_color=colors[2]),
        row=2, col=1
    )
    
    # Subject averages (stacked bar)
    subjects = ['Math', 'Science', 'English', 'History', 'ComputerScience']
    subject_colors = ['#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff7f0e']
    
    for i, subject in enumerate(subjects):
        fig.add_trace(
            go.Bar(x=cluster_stats.index, y=cluster_stats[subject],
                   name=subject, marker_color=subject_colors[i]),
            row=2, col=2
        )
    
    fig.update_layout(
        title="Comprehensive Cluster Analysis",
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    
    return fig

def get_cluster_insights(clustered_df):
    """
    Generate insights and recommendations for each cluster.
    
    Args:
        clustered_df (pandas.DataFrame): DataFrame with cluster labels
        
    Returns:
        dict: Insights for each cluster
    """
    
    insights = {}
    
    for cluster_id in sorted(clustered_df['Cluster'].unique()):
        cluster_data = clustered_df[clustered_df['Cluster'] == cluster_id]
        
        avg_score = cluster_data['FinalAverage'].mean()
        avg_attendance = cluster_data['AttendancePercentage'].mean()
        avg_study_hours = cluster_data['StudyHoursPerWeek'].mean()
        
        # Generate insights based on cluster characteristics
        insight_text = []
        recommendations = []
        
        if avg_score >= 85:
            insight_text.append("🏆 High-achieving students with excellent academic performance")
            recommendations.extend([
                "Consider advanced coursework or enrichment programs",
                "Peer mentoring opportunities",
                "Leadership roles in academic activities"
            ])
        elif avg_score >= 70:
            insight_text.append("✅ Good performers with solid academic foundation")
            recommendations.extend([
                "Focus on consistency across all subjects",
                "Time management skills development",
                "Goal setting for academic improvement"
            ])
        elif avg_score >= 55:
            insight_text.append("⚠️ Average students who need targeted support")
            recommendations.extend([
                "Identify specific subject weaknesses",
                "Implement structured study plans",
                "Consider additional tutoring support"
            ])
        else:
            insight_text.append("🚨 At-risk students requiring immediate intervention")
            recommendations.extend([
                "Comprehensive academic support program",
                "Regular progress monitoring",
                "Address attendance and engagement issues"
            ])
        
        if avg_attendance < 85:
            insight_text.append("📉 Below-average attendance patterns")
            recommendations.append("Focus on improving attendance and engagement")
        
        if avg_study_hours < 5:
            insight_text.append("⏰ Limited study time commitment")
            recommendations.append("Develop better study habits and time management")
        
        insights[f'Cluster {cluster_id}'] = {
            'size': len(cluster_data),
            'characteristics': insight_text,
            'recommendations': recommendations,
            'avg_metrics': {
                'final_average': round(avg_score, 1),
                'attendance': round(avg_attendance, 1),
                'study_hours': round(avg_study_hours, 1)
            }
        }
    
    return insights