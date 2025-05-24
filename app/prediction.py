import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def predict_performance(df, target='FinalAverage', test_size=0.2):
    """
    Predict student performance using multiple regression models.
    
    Args:
        df (pandas.DataFrame): Student dataset
        target (str): Target variable to predict
        test_size (float): Test set size
        
    Returns:
        tuple: (y_true, y_pred, feature_importance)
    """
    
    # Define features for prediction
    feature_columns = ['Math', 'Science', 'English', 'History', 'ComputerScience',
                      'AttendancePercentage', 'StudyHoursPerWeek']
    
    # Prepare the data
    X = df[feature_columns].copy()
    y = df[target].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    # Split the data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    except Exception as e:
        print(f"Error in train_test_split: {e}")
        return None, None, None
    
    # Train multiple models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
        'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10)
    }
    
    best_model = None
    best_score = -np.inf
    best_predictions = None
    best_feature_importance = None
    
    for name, model in models.items():
        try:
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate R² score
            score = r2_score(y_test, y_pred)
            
            if score > best_score:
                best_score = score
                best_model = model
                best_predictions = y_pred
                
                # Get feature importance
                if hasattr(model, 'feature_importances_'):
                    best_feature_importance = pd.Series(
                        model.feature_importances_, 
                        index=feature_columns
                    ).sort_values(ascending=True)
                elif hasattr(model, 'coef_'):
                    best_feature_importance = pd.Series(
                        np.abs(model.coef_), 
                        index=feature_columns
                    ).sort_values(ascending=True)
        
        except Exception as e:
            print(f"Error training {name}: {e}")
            continue
    
    return y_test, best_predictions, best_feature_importance

def plot_predictions(y_true, y_pred):
    """
    Create visualization comparing predicted vs actual values.
    
    Args:
        y_true (array): True values
        y_pred (array): Predicted values
        
    Returns:
        plotly.graph_objects.Figure: Prediction comparison plot
    """
    
    if y_true is None or y_pred is None:
        return go.Figure()
    
    # Create subplot figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Predicted vs Actual', 'Residual Plot'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # Predicted vs Actual scatter plot
    fig.add_trace(
        go.Scatter(
            x=y_true, 
            y=y_pred,
            mode='markers',
            name='Predictions',
            marker=dict(size=8, opacity=0.7, color='blue'),
            hovertemplate='Actual: %{x}<br>Predicted: %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash', width=2)
        ),
        row=1, col=1
    )
    
    # Residual plot
    residuals = y_true - y_pred
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(size=8, opacity=0.7, color='green'),
            hovertemplate='Predicted: %{x}<br>Residual: %{y}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Zero line for residuals
    fig.add_trace(
        go.Scatter(
            x=[min(y_pred), max(y_pred)],
            y=[0, 0],
            mode='lines',
            name='Zero Line',
            line=dict(color='red', dash='dash', width=2)
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Actual Values", row=1, col=1)
    fig.update_yaxes(title_text="Predicted Values", row=1, col=1)
    fig.update_xaxes(title_text="Predicted Values", row=1, col=2)
    fig.update_yaxes(title_text="Residuals", row=1, col=2)
    
    fig.update_layout(
        title="Model Performance Analysis",
        height=400,
        showlegend=True
    )
    
    return fig

def create_model_comparison(df, target='FinalAverage'):
    """
    Compare different regression models for performance prediction.
    
    Args:
        df (pandas.DataFrame): Student dataset
        target (str): Target variable
        
    Returns:
        plotly.graph_objects.Figure: Model comparison chart
    """
    
    feature_columns = ['Math', 'Science', 'English', 'History', 'ComputerScience',
                      'AttendancePercentage', 'StudyHoursPerWeek']
    
    X = df[feature_columns].fillna(df[feature_columns].mean())
    y = df[target].fillna(df[target].mean())
    
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
        'Random Forest': RandomForestRegressor(random_state=42, n_estimators=50, max_depth=10)
    }
    
    results = []
    
    for name, model in models.items():
        try:
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            
            # Train-test split for detailed metrics
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            results.append({
                'Model': name,
                'R² Score': r2,
                'RMSE': rmse,
                'MAE': mae,
                'CV Score (mean)': cv_scores.mean(),
                'CV Score (std)': cv_scores.std()
            })
            
        except Exception as e:
            print(f"Error with {name}: {e}")
            continue
    
    if not results:
        return go.Figure()
    
    results_df = pd.DataFrame(results)
    
    # Create comparison chart
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('R² Score', 'RMSE', 'Cross-Validation Score'),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # R² Score
    fig.add_trace(
        go.Bar(x=results_df['Model'], y=results_df['R² Score'],
               name='R² Score', marker_color=colors[0]),
        row=1, col=1
    )
    
    # RMSE
    fig.add_trace(
        go.Bar(x=results_df['Model'], y=results_df['RMSE'],
               name='RMSE', marker_color=colors[1]),
        row=1, col=2
    )
    
    # CV Score with error bars
    fig.add_trace(
        go.Bar(x=results_df['Model'], y=results_df['CV Score (mean)'],
               error_y=dict(type='data', array=results_df['CV Score (std)']),
               name='CV Score', marker_color=colors[2]),
        row=1, col=3
    )
    
    fig.update_layout(
        title="Model Performance Comparison",
        height=400,
        showlegend=False
    )
    
    return fig

def predict_at_risk_students(df, threshold=60):
    """
    Identify students at risk of poor performance.
    
    Args:
        df (pandas.DataFrame): Student dataset
        threshold (float): Performance threshold for at-risk classification
        
    Returns:
        pandas.DataFrame: At-risk students with risk factors
    """
    
    # Create risk indicators
    risk_factors = []
    
    for _, student in df.iterrows():
        factors = []
        
        if student['FinalAverage'] < threshold:
            factors.append('Low Final Average')
        
        if student['AttendancePercentage'] < 80:
            factors.append('Poor Attendance')
        
        if student['StudyHoursPerWeek'] < 5:
            factors.append('Insufficient Study Time')
        
        # Check for failing grades in any subject
        subjects = ['Math', 'Science', 'English', 'History', 'ComputerScience']
        failing_subjects = [subject for subject in subjects if student[subject] < 50]
        if failing_subjects:
            factors.append(f'Failing in: {", ".join(failing_subjects)}')
        
        risk_factors.append(factors)
    
    df_risk = df.copy()
    df_risk['Risk_Factors'] = risk_factors
    df_risk['Risk_Count'] = [len(factors) for factors in risk_factors]
    df_risk['At_Risk'] = df_risk['Risk_Count'] > 0
    
    # Return only at-risk students
    at_risk_students = df_risk[df_risk['At_Risk']].copy()
    at_risk_students = at_risk_students.sort_values('Risk_Count', ascending=False)
    
    return at_risk_students[['StudentID', 'FirstName', 'LastName', 'FinalAverage', 
                            'AttendancePercentage', 'StudyHoursPerWeek', 
                            'Risk_Factors', 'Risk_Count']]

def create_risk_analysis_chart(df):
    """
    Create visualizations for at-risk student analysis.
    
    Args:
        df (pandas.DataFrame): Student dataset
        
    Returns:
        plotly.graph_objects.Figure: Risk analysis visualization
    """
    
    at_risk_df = predict_at_risk_students(df)
    
    if len(at_risk_df) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No at-risk students identified with current criteria",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Risk Factor Distribution', 'At-Risk by Grade Level',
                       'Performance vs Attendance (At-Risk)', 'Study Hours Distribution'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "histogram"}]]
    )
    
    # Risk factor distribution
    risk_counts = at_risk_df['Risk_Count'].value_counts().sort_index()
    fig.add_trace(
        go.Bar(x=risk_counts.index, y=risk_counts.values,
               name='Risk Factors', marker_color='red'),
        row=1, col=1
    )
    
    # At-risk by grade level
    grade_risk = at_risk_df['GradeLevel'].value_counts()
    fig.add_trace(
        go.Bar(x=grade_risk.index, y=grade_risk.values,
               name='Grade Level', marker_color='orange'),
        row=1, col=2
    )
    
    # Performance vs Attendance scatter
    fig.add_trace(
        go.Scatter(x=at_risk_df['AttendancePercentage'], 
                  y=at_risk_df['FinalAverage'],
                  mode='markers',
                  name='At-Risk Students',
                  marker=dict(size=10, color='red', opacity=0.7)),
        row=2, col=1
    )
    
    # Study hours distribution
    fig.add_trace(
        go.Histogram(x=at_risk_df['StudyHoursPerWeek'],
                    name='Study Hours', marker_color='purple'),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Number of Risk Factors", row=1, col=1)
    fig.update_xaxes(title_text="Grade Level", row=1, col=2)
    fig.update_xaxes(title_text="Attendance (%)", row=2, col=1)
    fig.update_xaxes(title_text="Study Hours per Week", row=2, col=2)
    
    fig.update_yaxes(title_text="Number of Students", row=1, col=1)
    fig.update_yaxes(title_text="Number of Students", row=1, col=2)
    fig.update_yaxes(title_text="Final Average", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    
    fig.update_layout(
        title=f"At-Risk Student Analysis ({len(at_risk_df)} students identified)",
        height=600,
        showlegend=False
    )
    
    return fig

def generate_performance_insights(df):
    """
    Generate insights about factors affecting student performance.
    
    Args:
        df (pandas.DataFrame): Student dataset
        
    Returns:
        dict: Performance insights
    """
    
    insights = {}
    
    # Correlation analysis
    numeric_cols = ['Math', 'Science', 'English', 'History', 'ComputerScience',
                   'AttendancePercentage', 'StudyHoursPerWeek', 'FinalAverage']
    
    correlations = df[numeric_cols].corr()['FinalAverage'].drop('FinalAverage').sort_values(ascending=False)
    
    insights['strongest_correlations'] = {
        'positive': correlations[correlations > 0].head(3).to_dict(),
        'negative': correlations[correlations < 0].tail(2).to_dict()
    }
    
    # Performance by category
    category_stats = df.groupby('PerformanceCategory')['FinalAverage'].agg(['count', 'mean', 'std']).round(2)
    insights['category_breakdown'] = category_stats.to_dict()
    
    # Subject analysis
    subjects = ['Math', 'Science', 'English', 'History', 'ComputerScience']
    subject_performance = df[subjects].mean().sort_values(ascending=False)
    insights['subject_rankings'] = subject_performance.to_dict()
    
    # At-risk analysis
    at_risk_students = predict_at_risk_students(df)
    insights['at_risk_summary'] = {
        'total_at_risk': len(at_risk_students),
        'percentage_at_risk': round(len(at_risk_students) / len(df) * 100, 1),
        'common_risk_factors': at_risk_students['Risk_Count'].value_counts().to_dict()
    }
    
    return insights