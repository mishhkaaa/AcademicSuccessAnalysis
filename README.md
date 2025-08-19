# Student Performance DNA: Visual Analytics for Academic Success

## 📊 Project Overview

**Student Performance DNA** is an interactive data visualization dashboard for analyzing academic performance, inspired by cutting-edge research in educational data mining and predictive analytics. Built with Python, Streamlit, and modern visualization libraries, it uncovers hidden patterns in student data and delivers actionable insights to foster academic success.

---

## 🚀 Features

- **Radar Chart:** Visualize individual student strengths and weaknesses across subjects.
- **Parallel Coordinates Plot:** Explore class-wide trends and multidimensional subject relationships.
- **Cluster-Based Grouping:** Group students using K-Means and visualize with PCA/t-SNE for cohort analysis.
- **Correlation Heatmap:** Analyze relationships between study habits, attendance, and academic outcomes.
- **Time Series Simulation:** Simulate and visualize academic progress over multiple terms.
- **Predictive Panel:** Predict final averages using advanced regression models.
- **Smart Filters:** Filter by grade, subject, cluster, and performance category for targeted analysis.
- **Student Recommendation Panel:** Get personalized improvement suggestions for each student.
- **Multi-Tab Dashboard:** Seamlessly switch between individual, class, clustering, and summary views for comprehensive exploration.

---

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
│   ├── utils.py              # Helpers and recommendations
│   ├── timeseries.py         # Time series simulation and trend charts
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**
    ```bash
    git clone https://github.com/mishhkaaa/AcademicSuccessAnalysis
    cd AcademicSuccessAnalysis
    ```

2. **Set up Python environment**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On Unix/Mac:
    source venv/bin/activate
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

---

## 📚 References

- Ahmed Kord et al. (2025): Academic course planning recommendation and students' performance prediction
- Yaodong Yuan et al. (2024): Visualization analysis of educational data statistics based on big data mining

---

## 🛠️ Tools & Libraries

- pandas
- plotly
- seaborn
- matplotlib
- scikit-learn
- streamlit

---

> For questions or contributions, please open an issue or submit a pull request.
