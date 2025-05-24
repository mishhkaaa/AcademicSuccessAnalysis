import streamlit as st
import pandas as pd
import numpy as np

def create_filters(df):
    """
    Create interactive filters for the dashboard sidebar.
    
    Args:
        df (pandas.DataFrame): Student dataset
        
    Returns:
        tuple: (filtered_df, selected_student, selected_grade, selected_category)
    """
    
    # Initialize session state for filters if not exists
    if 'filter_reset' not in st.session_state:
        st.session_state.filter_reset = False
    
    # Student selection
    st.sidebar.markdown("### 👤 Student Selection")
    student_options = ["All Students"] + [f"{row['StudentID']}" for _, row in df.iterrows()]
    student_labels = ["All Students"] + [f"{row['FirstName']} {row['LastName']} (ID: {row['StudentID']})" 
                     for _, row in df.iterrows()]
    
    selected_student_idx = st.sidebar.selectbox(
        "Select Student:",
        range(len(student_options)),
        format_func=lambda x: student_labels[x],
        key="student_selector"
    )
    
    selected_student = student_options[selected_student_idx]
    
    # Grade Level Filter
    st.sidebar.markdown("### 🎓 Grade Level Filter")
    grade_levels = ["All Grades"] + sorted(df['GradeLevel'].unique().tolist())
    selected_grade = st.sidebar.selectbox(
        "Filter by Grade Level:",
        grade_levels,
        key="grade_filter"
    )
    
    # Performance Category Filter
    st.sidebar.markdown("### 📊 Performance Category Filter")
    performance_categories = ["All Categories"] + sorted(df['PerformanceCategory'].unique().tolist())
    selected_category = st.sidebar.selectbox(
        "Filter by Performance Category:",
        performance_categories,
        key="performance_filter"
    )
    
    # Subject Score Range Filters
    st.sidebar.markdown("### 📚 Subject Score Filters")
    
    # Math filter
    math_range = st.sidebar.slider(
        "Math Score Range:",
        min_value=0.0,
        max_value=100.0,
        value=(0.0, 100.0),
        step=1.0,
        key="math_filter"
    )
    
    # Science filter
    science_range = st.sidebar.slider(
        "Science Score Range:",
        min_value=0.0,
        max_value=100.0,
        value=(0.0, 100.0),
        step=1.0,
        key="science_filter"
    )
    
    # Final Average filter
    st.sidebar.markdown("### 🎯 Overall Performance Filter")
    final_avg_range = st.sidebar.slider(
        "Final Average Range:",
        min_value=0.0,
        max_value=100.0,
        value=(0.0, 100.0),
        step=1.0,
        key="final_avg_filter"
    )
    
    # Attendance filter
    st.sidebar.markdown("### 📅 Attendance Filter")
    attendance_range = st.sidebar.slider(
        "Attendance Percentage:",
        min_value=0.0,
        max_value=100.0,
        value=(0.0, 100.0),
        step=1.0,
        key="attendance_filter"
    )
    
    # Study Hours filter
    st.sidebar.markdown("### ⏰ Study Hours Filter")
    study_hours_range = st.sidebar.slider(
        "Study Hours per Week:",
        min_value=0.0,
        max_value=float(df['StudyHoursPerWeek'].max()),
        value=(0.0, float(df['StudyHoursPerWeek'].max())),
        step=0.5,
        key="study_hours_filter"
    )
    
    # Extracurricular Activities filter
    st.sidebar.markdown("### 🏃 Extracurricular Activities")
    if 'ExtracurricularActivities' in df.columns:
        extracurricular_options = ["All"] + sorted(df['ExtracurricularActivities'].unique().tolist())
        selected_extracurricular = st.sidebar.selectbox(
            "Filter by Activities:",
            extracurricular_options,
            key="extracurricular_filter"
        )
    else:
        selected_extracurricular = "All"
    
    # Advanced Filters Section
    st.sidebar.markdown("### 🔧 Advanced Filters")
    
    # Show only at-risk students
    show_at_risk_only = st.sidebar.checkbox(
        "Show only at-risk students",
        key="at_risk_filter",
        help="Students with Final Average < 60% or Attendance < 80%"
    )
    
    # Show top performers only
    show_top_performers = st.sidebar.checkbox(
        "Show only top performers",
        key="top_performers_filter",
        help="Students with Final Average >= 85%"
    )
    
    # Reset filters button
    if st.sidebar.button("🔄 Reset All Filters", key="reset_filters"):
        st.session_state.filter_reset = True
        st.rerun()
    
    # Apply filters to dataframe
    filtered_df = df.copy()
    
    # Apply grade level filter
    if selected_grade != "All Grades":
        filtered_df = filtered_df[filtered_df['GradeLevel'] == selected_grade]
    
    # Apply performance category filter
    if selected_category != "All Categories":
        filtered_df = filtered_df[filtered_df['PerformanceCategory'] == selected_category]
    
    # Apply subject score filters
    filtered_df = filtered_df[
        (filtered_df['Math'] >= math_range[0]) & 
        (filtered_df['Math'] <= math_range[1])
    ]
    
    filtered_df = filtered_df[
        (filtered_df['Science'] >= science_range[0]) & 
        (filtered_df['Science'] <= science_range[1])
    ]
    
    # Apply final average filter
    filtered_df = filtered_df[
        (filtered_df['FinalAverage'] >= final_avg_range[0]) & 
        (filtered_df['FinalAverage'] <= final_avg_range[1])
    ]
    
    # Apply attendance filter
    filtered_df = filtered_df[
        (filtered_df['AttendancePercentage'] >= attendance_range[0]) & 
        (filtered_df['AttendancePercentage'] <= attendance_range[1])
    ]
    
    # Apply study hours filter
    filtered_df = filtered_df[
        (filtered_df['StudyHoursPerWeek'] >= study_hours_range[0]) & 
        (filtered_df['StudyHoursPerWeek'] <= study_hours_range[1])
    ]
    
    # Apply extracurricular filter
    if selected_extracurricular != "All" and 'ExtracurricularActivities' in df.columns:
        filtered_df = filtered_df[filtered_df['ExtracurricularActivities'] == selected_extracurricular]
    
    # Apply at-risk filter
    if show_at_risk_only:
        at_risk_condition = (
            (filtered_df['FinalAverage'] < 60) | 
            (filtered_df['AttendancePercentage'] < 80)
        )
        filtered_df = filtered_df[at_risk_condition]
    
    # Apply top performers filter
    if show_top_performers:
        filtered_df = filtered_df[filtered_df['FinalAverage'] >= 85]
    
    # Display filter summary
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Filter Summary")
    
    total_students = len(df)
    filtered_students = len(filtered_df)
    percentage_shown = (filtered_students / total_students * 100) if total_students > 0 else 0
    
    st.sidebar.metric(
        "Students Shown",
        f"{filtered_students} / {total_students}",
        f"{percentage_shown:.1f}%"
    )
    
    if filtered_students > 0:
        st.sidebar.metric(
            "Average Performance",
            f"{filtered_df['FinalAverage'].mean():.1f}%"
        )
        
        st.sidebar.metric(
            "Average Attendance",
            f"{filtered_df['AttendancePercentage'].mean():.1f}%"
        )
    
    # Warning if no students match filters
    if len(filtered_df) == 0:
        st.sidebar.warning("⚠️ No students match the current filter criteria. Please adjust your filters.")
    
    return filtered_df, selected_student, selected_grade, selected_category

def create_quick_filters(df):
    """
    Create quick filter buttons for common filter combinations.
    
    Args:
        df (pandas.DataFrame): Student dataset
        
    Returns:
        pandas.DataFrame: Filtered dataframe based on quick filter selection
    """
    
    st.sidebar.markdown("### ⚡ Quick Filters")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🔴 At Risk", key="quick_at_risk"):
            return df[(df['FinalAverage'] < 60) | (df['AttendancePercentage'] < 80)]
    
    with col2:
        if st.button("🟢 High Achievers", key="quick_high_achievers"):
            return df[df['FinalAverage'] >= 85]
    
    col3, col4 = st.sidebar.columns(2)
    
    with col3:
        if st.button("🟡 Average", key="quick_average"):
            return df[(df['FinalAverage'] >= 60) & (df['FinalAverage'] < 85)]
    
    with col4:
        if st.button("📚 Low Attendance", key="quick_low_attendance"):
            return df[df['AttendancePercentage'] < 80]
    
    return df

def get_filter_statistics(original_df, filtered_df):
    """
    Calculate statistics about the applied filters.
    
    Args:
        original_df (pandas.DataFrame): Original unfiltered dataset
        filtered_df (pandas.DataFrame): Filtered dataset
        
    Returns:
        dict: Filter statistics
    """
    
    stats = {
        'original_count': len(original_df),
        'filtered_count': len(filtered_df),
        'percentage_retained': (len(filtered_df) / len(original_df) * 100) if len(original_df) > 0 else 0,
        'students_removed': len(original_df) - len(filtered_df)
    }
    
    if len(filtered_df) > 0:
        stats.update({
            'avg_performance_original': original_df['FinalAverage'].mean(),
            'avg_performance_filtered': filtered_df['FinalAverage'].mean(),
            'performance_change': filtered_df['FinalAverage'].mean() - original_df['FinalAverage'].mean(),
            'grade_distribution': filtered_df['GradeLevel'].value_counts().to_dict(),
            'category_distribution': filtered_df['PerformanceCategory'].value_counts().to_dict()
        })
    
    return stats

def create_dynamic_filters(df, filter_type="performance"):
    """
    Create dynamic filters that update based on data characteristics.
    
    Args:
        df (pandas.DataFrame): Student dataset
        filter_type (str): Type of dynamic filter to create
        
    Returns:
        pandas.DataFrame: Dynamically filtered dataframe
    """
    
    if filter_type == "performance":
        # Create performance-based dynamic filters
        performance_quartiles = df['FinalAverage'].quantile([0.25, 0.5, 0.75])
        
        st.sidebar.markdown("### 📊 Performance Quartiles")
        
        quartile_options = [
            "All Students",
            f"Top Quartile (≥{performance_quartiles[0.75]:.1f}%)",
            f"Upper Middle ({performance_quartiles[0.5]:.1f}% - {performance_quartiles[0.75]:.1f}%)",
            f"Lower Middle ({performance_quartiles[0.25]:.1f}% - {performance_quartiles[0.5]:.1f}%)",
            f"Bottom Quartile (<{performance_quartiles[0.25]:.1f}%)"
        ]
        
        selected_quartile = st.sidebar.selectbox(
            "Performance Quartile:",
            quartile_options,
            key="quartile_filter"
        )
        
        if selected_quartile == quartile_options[1]:  # Top quartile
            return df[df['FinalAverage'] >= performance_quartiles[0.75]]
        elif selected_quartile == quartile_options[2]:  # Upper middle
            return df[(df['FinalAverage'] >= performance_quartiles[0.5]) & 
                     (df['FinalAverage'] < performance_quartiles[0.75])]
        elif selected_quartile == quartile_options[3]:  # Lower middle
            return df[(df['FinalAverage'] >= performance_quartiles[0.25]) & 
                     (df['FinalAverage'] < performance_quartiles[0.5])]
        elif selected_quartile == quartile_options[4]:  # Bottom quartile
            return df[df['FinalAverage'] < performance_quartiles[0.25]]
    
    elif filter_type == "study_habits":
        # Create study habits-based filters
        st.sidebar.markdown("### 📖 Study Habits Filter")
        
        study_median = df['StudyHoursPerWeek'].median()
        attendance_median = df['AttendancePercentage'].median()
        
        habit_options = [
            "All Students",
            "Dedicated Students (High Study Hours & Attendance)",
            "Inconsistent Students (High Study Hours, Low Attendance)",
            "Present but Unprepared (Low Study Hours, High Attendance)",
            "At-Risk Students (Low Study Hours & Attendance)"
        ]
        
        selected_habit = st.sidebar.selectbox(
            "Study Habit Category:",
            habit_options,
            key="study_habit_filter"
        )
        
        if selected_habit == habit_options[1]:  # Dedicated
            return df[(df['StudyHoursPerWeek'] >= study_median) & 
                     (df['AttendancePercentage'] >= attendance_median)]
        elif selected_habit == habit_options[2]:  # Inconsistent
            return df[(df['StudyHoursPerWeek'] >= study_median) & 
                     (df['AttendancePercentage'] < attendance_median)]
        elif selected_habit == habit_options[3]:  # Present but unprepared
            return df[(df['StudyHoursPerWeek'] < study_median) & 
                     (df['AttendancePercentage'] >= attendance_median)]
        elif selected_habit == habit_options[4]:  # At-risk
            return df[(df['StudyHoursPerWeek'] < study_median) & 
                     (df['AttendancePercentage'] < attendance_median)]
    
    return df

def export_filtered_data(filtered_df):
    """
    Create export functionality for filtered data.
    
    Args:
        filtered_df (pandas.DataFrame): Filtered dataset
        
    Returns:
        None (creates download button in sidebar)
    """
    
    if len(filtered_df) > 0:
        st.sidebar.markdown("### 💾 Export Filtered Data")
        
        # Convert to CSV
        csv_data = filtered_df.to_csv(index=False)
        
        st.sidebar.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"filtered_student_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="download_csv"
        )
        
        st.sidebar.info(f"📊 {len(filtered_df)} students in current filter")
    else:
        st.sidebar.warning("No data to export with current filters.")