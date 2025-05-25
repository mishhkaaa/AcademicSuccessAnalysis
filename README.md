# Student Performance DNA: Visual Analytics for Academic Success

## 📊 Project Overview
Student Performance DNA is an interactive data visualization dashboard for analyzing academic performance, inspired by recent research in educational data mining and predictive analytics. Built with Python, Streamlit, and modern visualization libraries, it helps uncover hidden patterns in student data and provides actionable insights for academic success.

## 🚀 Features
- **Radar Chart**: Visualize individual student strengths and weaknesses
- **Parallel Coordinates Plot**: Explore class-wide trends across subjects
- **Cluster-Based Grouping**: Group students using K-Means and visualize with PCA/t-SNE
- **Correlation Heatmap**: Analyze relationships between study habits, attendance, and performance
- **Time Series Simulation**: Simulate and visualize academic progress over multiple terms
- **Predictive Panel**: Predict final averages using regression models
- **Smart Filters**: Filter by grade, subject, cluster, and performance category
- **Student Recommendation Panel**: Get personalized improvement suggestions
- **Multi-Tab Dashboard**: Seamlessly switch between individual, class, clustering, and summary views

## 🗂️ Folder Structure
```
StudentPerformanceDNA/
│
├── data/
│   └── students.csv
├── app/
│   ├── app.py                # Streamlit entry point
│   ├── radar_chart.py        # Radar chart logic
│   ├── clustering.py         # K-Means, PCA, visualization
│   ├── prediction.py         # Regression and plots
│   ├── filters.py            # Sidebar and filtering logic
│   ├── utils.py              # Helpers
├── assets/
│   └── logo.png              # Optional branding
├── requirements.txt
└── README.md
```

## ⚙️ Installation & Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/mishhkaaa/AcademicSuccessAnalysis
   cd StudentPerformanceDNA
   ```
2. **Set up Python environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # (use source venv/bin/activate on Unix/Mac)
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Add your data**
   - Place your `students.csv` file in the `data/` folder.

5. **Run the app**
   ```bash
   streamlit run app/app.py
   ```

## 🖼️ Screenshots & Insights
- (Add screenshots of each dashboard tab and a summary of key findings here)

## 📚 References
- Ahmed Kord et al. (2025): Academic course planning recommendation and students' performance prediction
- Yaodong Yuan et al. (2024): Visualization analysis of educational data statistics based on big data mining

## 🛠️ Tools & Libraries
- pandas, plotly, seaborn, matplotlib, scikit-learn, streamlit

## ✨ Credits
Developed by Mishka for the Data Visualization with Python term project.
