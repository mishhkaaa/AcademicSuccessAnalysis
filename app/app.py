import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from radar_chart import create_radar_chart
from clustering import perform_clustering, visualize_clusters
from prediction import predict_performance, plot_predictions
from filters import create_filters
from utils import load_data, generate_recommendations

# Page configuration
st.set_page_config(
    page_title="Student Performance DNA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .tab-header {
        font-size: 1.8rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Student Performance DNA: Visual Analytics for Academic Success</h1>', unsafe_allow_html=True)
    
    # Load data
    try:
        df = load_data()
        if df is None:
            st.error("Please ensure your CSV file is in the data/ folder and run the app from the project root directory.")
            return
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please make sure you have a CSV file with the required columns in the data/ folder.")
        return
    
    # Sidebar filters
    st.sidebar.title("🔧 Dashboard Controls")
    filtered_df, selected_student, selected_grade, selected_category = create_filters(df)
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "👤 Individual Student Analysis", 
        "📈 Class Trends & Correlations", 
        "🔬 Clustering & Predictions", 
        "📋 Summary Insights"
    ])
    
    with tab1:
        st.markdown('<h2 class="tab-header">Individual Student Performance Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Radar chart for selected student
            if selected_student != "All Students":
                student_data = df[df['StudentID'] == selected_student].iloc[0]
                fig_radar = create_radar_chart(student_data)
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.info("Please select a specific student from the sidebar to view individual analysis.")
        
        with col2:
            if selected_student != "All Students":
                student_data = df[df['StudentID'] == selected_student].iloc[0]
                
                # Student info card
                st.markdown("### 📋 Student Information")
                st.write(f"**Name:** {student_data['FirstName']} {student_data['LastName']}")
                st.write(f"**Grade Level:** {student_data['GradeLevel']}")
                st.write(f"**Performance Category:** {student_data['PerformanceCategory']}")
                st.write(f"**Final Average:** {student_data['FinalAverage']:.1f}%")
                st.write(f"**Attendance:** {student_data['AttendancePercentage']:.1f}%")
                st.write(f"**Study Hours/Week:** {student_data['StudyHoursPerWeek']}")
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                recommendations = generate_recommendations(student_data)
                for rec in recommendations:
                    st.write(f"• {rec}")
    
    with tab2:
        st.markdown('<h2 class="tab-header">Class-Wide Trends and Correlations</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Parallel coordinates plot
            st.markdown("### 📊 Parallel Coordinates: Subject Performance")
            subjects = ['Math', 'Science', 'English', 'History', 'ComputerScience']
            
            fig_parallel = go.Figure(data=
                go.Parcoords(
                    line=dict(color=filtered_df['FinalAverage'],
                             colorscale='RdYlBu',
                             showscale=True,
                             colorbar=dict(title="Final Average")),
                    dimensions=[dict(range=[0, 100],
                                   constraintrange=[0, 100],
                                   label=subject,
                                   values=filtered_df[subject]) for subject in subjects]
                )
            )
            fig_parallel.update_layout(title="Student Performance Across Subjects", height=400)
            st.plotly_chart(fig_parallel, use_container_width=True)
        
        with col2:
            # Performance distribution
            st.markdown("### 📈 Performance Distribution")
            fig_dist = px.histogram(filtered_df, x='FinalAverage', 
                                   color='PerformanceCategory',
                                   nbins=20,
                                   title="Distribution of Final Averages")
            fig_dist.update_layout(height=400)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Correlation heatmap
        st.markdown("### 🔥 Correlation Heatmap")
        numeric_cols = ['Math', 'Science', 'English', 'History', 'ComputerScience', 
                       'AttendancePercentage', 'StudyHoursPerWeek', 'FinalAverage']
        corr_matrix = filtered_df[numeric_cols].corr()
        
        fig_heatmap = px.imshow(corr_matrix, 
                               text_auto=True,
                               aspect="auto",
                               title="Correlation Matrix of Academic Variables",
                               color_continuous_scale='RdBu')
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="tab-header">Advanced Analytics: Clustering & Predictions</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Clustering analysis
            st.markdown("### 🎯 Student Clustering Analysis")
            clustered_df, cluster_centers = perform_clustering(filtered_df)
            fig_cluster = visualize_clusters(clustered_df)
            st.plotly_chart(fig_cluster, use_container_width=True)
            
            # Cluster summary
            st.markdown("### 📊 Cluster Summary")
            cluster_summary = clustered_df.groupby('Cluster').agg({
                'FinalAverage': 'mean',
                'AttendancePercentage': 'mean',
                'StudyHoursPerWeek': 'mean'
            }).round(2)
            st.dataframe(cluster_summary)
        
        with col2:
            # Prediction analysis
            st.markdown("### 🔮 Performance Prediction")
            y_true, y_pred, feature_importance = predict_performance(filtered_df)
            
            if y_true is not None and y_pred is not None:
                # Prediction accuracy metrics
                mse = mean_squared_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                
                st.metric("R² Score", f"{r2:.3f}")
                st.metric("RMSE", f"{np.sqrt(mse):.2f}")
                
                # Prediction plot
                fig_pred = plot_predictions(y_true, y_pred)
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Feature importance
                if feature_importance is not None:
                    st.markdown("### 🎯 Feature Importance")
                    fig_importance = px.bar(
                        x=feature_importance.values,
                        y=feature_importance.index,
                        orientation='h',
                        title="Feature Importance for Prediction"
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
    
    with tab4:
        st.markdown('<h2 class="tab-header">Summary Insights & Statistics</h2>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Students", len(filtered_df))
        with col2:
            st.metric("Average Performance", f"{filtered_df['FinalAverage'].mean():.1f}%")
        with col3:
            st.metric("Average Attendance", f"{filtered_df['AttendancePercentage'].mean():.1f}%")
        with col4:
            st.metric("Avg Study Hours/Week", f"{filtered_df['StudyHoursPerWeek'].mean():.1f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top performers
            st.markdown("### 🏆 Top 10 Performers")
            top_students = filtered_df.nlargest(10, 'FinalAverage')[
                ['FirstName', 'LastName', 'FinalAverage', 'PerformanceCategory']
            ]
            st.dataframe(top_students, hide_index=True)
            
            # Subject-wise averages
            st.markdown("### 📚 Subject-wise Class Averages")
            subjects = ['Math', 'Science', 'English', 'History', 'ComputerScience']
            subject_avgs = filtered_df[subjects].mean().round(2)
            
            fig_subjects = px.bar(
                x=subject_avgs.index,
                y=subject_avgs.values,
                title="Average Scores by Subject",
                color=subject_avgs.values,
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_subjects, use_container_width=True)
        
        with col2:
            # Performance category distribution
            st.markdown("### 📊 Performance Category Distribution")
            category_counts = filtered_df['PerformanceCategory'].value_counts()
            
            fig_pie = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Distribution of Performance Categories"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Grade level analysis
            st.markdown("### 🎓 Performance by Grade Level")
            grade_performance = filtered_df.groupby('GradeLevel')['FinalAverage'].mean().round(2)
            
            fig_grade = px.bar(
                x=grade_performance.index,
                y=grade_performance.values,
                title="Average Performance by Grade Level",
                color=grade_performance.values,
                color_continuous_scale='blues'
            )
            st.plotly_chart(fig_grade, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Student Performance DNA Dashboard** | Built with Streamlit & Python | "
        "Data Visualization Project by 2nd Year BE Student"
    )

if __name__ == "__main__":
    main()