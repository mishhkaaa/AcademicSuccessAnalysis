import pandas as pd
import numpy as np

def load_data(filepath="data/students.csv"):
    """
    Load student data from a CSV file.
    Args:
        filepath (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded student dataframe.
    """
    try:
        df = pd.read_csv(filepath)
        # Basic cleaning: fill missing values if needed
        df.fillna({
            'Math': 0, 'Science': 0, 'English': 0, 'History': 0, 'ComputerScience': 0,
            'AttendancePercentage': 0, 'StudyHoursPerWeek': 0, 'FinalAverage': 0
        }, inplace=True)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def generate_recommendations(student_row):
    """
    Generate personalized recommendations for a student.
    Args:
        student_row (pd.Series): Row of student data.
    Returns:
        list: List of recommendation strings.
    """
    recommendations = []
    # Identify weak subjects
    weak_subjects = [subj for subj in ['Math', 'Science', 'English', 'History', 'ComputerScience']
                     if student_row.get(subj, 0) < 60]
    if weak_subjects:
        recommendations.append(f"Focus on improving: {', '.join(weak_subjects)}")
    # Study hours recommendation
    if student_row.get('StudyHoursPerWeek', 0) < 5:
        recommendations.append("Increase study hours to at least 5 per week.")
    # Attendance recommendation
    if student_row.get('AttendancePercentage', 100) < 80:
        recommendations.append("Improve attendance to above 80%.")
    # General performance
    if student_row.get('FinalAverage', 0) < 60:
        recommendations.append("Seek academic support or tutoring.")
    elif student_row.get('FinalAverage', 0) >= 85:
        recommendations.append("Consider advanced/enrichment programs.")
    return recommendations

def get_class_averages(df):
    """
    Calculate class average for each subject.
    Args:
        df (pd.DataFrame): Student dataframe.
    Returns:
        dict: Mapping from subject to average score.
    """
    subjects = ['Math', 'Science', 'English', 'History', 'ComputerScience']
    return {subj: df[subj].mean() for subj in subjects}

def get_student_by_id(df, student_id):
    """
    Retrieve a student row by StudentID.
    Args:
        df (pd.DataFrame): Student dataframe.
        student_id (str or int): Student ID.
    Returns:
        pd.Series: Student row.
    """
    student = df[df['StudentID'] == student_id]
    if not student.empty:
        return student.iloc[0]
    return None
