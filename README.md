Overview
The Enhanced KMeans Clustering App is an interactive web application built using Streamlit. It allows users to upload datasets, preprocess them, and perform clustering using the KMeans algorithm. The app also provides interactive visualizations and options to download the clustered dataset.
App Url:
https://kmeansanalyzer.streamlit.app/

Features
Upload Dataset:

Accepts CSV files and displays a preview of the data.
Preprocessing Options:

Drop unnecessary columns.
Scale the data using StandardScaler or MinMaxScaler.
Select specific features for clustering.
KMeans Clustering:

Choose the number of clusters (k).
Calculate metrics like inertia and silhouette score.
Dimensionality Reduction:

Uses PCA for 2D or 3D visualization.
Automatically adjusts components based on dataset size.
Interactive Visualizations:

2D and 3D scatter plots using Matplotlib and Plotly.
Cluster differentiation with color-coded points.
Download Clustered Dataset:

Export the clustered data as a CSV file.
Installation
Follow these steps to set up and run the app:

Prerequisites
Python 3.8 or later
pip (Python package manager)
Steps
Clone this repository:

git clone <repository_url>
cd <repository_directory>
Install dependencies:

pip install -r requirements.txt
Run the Streamlit app:

streamlit run app.py
Open the app in your browser:

The app will usually open at http://localhost:8501.
Usage
Upload Data:

Click on the "Upload a CSV file" button to load your dataset.
Preprocess Data:

Drop unnecessary columns using the sidebar.
Scale the data and select features for clustering.
Perform Clustering:

Set the number of clusters (k) using the slider.
View clustering metrics like inertia and silhouette score.
Visualize Clusters:

Choose between 2D or 3D scatter plots.
Interact with the Plotly visualizations.
Download Results:

Click the "Download as CSV" button to save the clustered data.
Dependencies
The following Python libraries are used in this project:

Streamlit: For building the web application.
Pandas: For data manipulation.
scikit-learn: For clustering and preprocessing.
Matplotlib: For 2D plotting.
Plotly: For interactive 3D visualization.
