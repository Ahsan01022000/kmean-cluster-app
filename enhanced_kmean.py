import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
from plotly import express as px
from sklearn.decomposition import PCA


#title 
st.title("KMeans Clustering App")
st.write("Upload Your CSV dataset,Configure Clustering Parameters and Visualise the result")


#upload the dataset
uploaded_file=st.file_uploader("Upload the Csv file",type="csv")
if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    st.write("###dataset preview: ")
    st.dataframe(df)
   
    #Data Summary
    st.write("###data summary:")
    st.write(df.describe())
    st.write("Missing values: ",df.isnull().sum())
    
    #Dropping unnecessary columns
    st.sidebar.subheader("Data Preprocessing")
    unecessary_columns=st.sidebar.multiselect("Columns to Drop",options=df.columns)
    if unecessary_columns:
        df=df.drop(columns=unecessary_columns)
        st.write(f"Updated dataset(dropped Columns:{','.join(unecessary_columns)})")
        st.dataframe(df)
    
    
    #Data Preprocessing 
    scale_data=st.sidebar.checkbox("Scale Data",value=True)
    scaler_options=st.sidebar.radio("Scaler Type",options=["StandardScaler","MinMaxScaler"],index=0)
    
    
    #feature Selection
    features=st.multiselect("Select the features for Clustering",options=df.columns)
    if features:
        X=df[features]
        if scale_data:
            scaler= StandardScaler() if scaler_options == "StandardScaler" else MinMaxScaler()
            X=scaler.fit_transform(X)
        
        
        #Choosing k-Value
        n_clusters=st.slider("Select the Number of Clusters(K-value)",min_value=2,max_value=10,value=3)
       
        
        #Applying KMeans
        kmeans=KMeans(n_clusters=n_clusters,random_state=42)
        labels=kmeans.fit_predict(X)
        df["cluster"]= labels

        
        #cluster metrics
        inertia=kmeans.inertia_
        silhouette=silhouette_score(X,labels)

        st.write("### Cluster Metrics: ")
        st.write(f"inertia :{inertia: .2f}")
        st.write(f"Silhouette Score: {silhouette: .2f}")


        #Visualising Option
        st.sidebar.subheader("Visualising Options")
        vis_type=st.sidebar.radio("select your Visualising type",options=["2D plot","3D plot"])
        color_palette=st.sidebar.selectbox("Select the colors",options=["viridis", "plasma", "cividis", "cool"])


        # Step 7: Dimensionality Reduction for Visualization
        max_components = min(X.shape[0], X.shape[1])  # Maximum valid components
        n_components = min(3, max_components)  # Adjust for 3D visualization or less
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        if n_components < 2:
            st.warning("Only 1 feature selected so no visualisation possible,Please select 2 or more features.")
        else:
            if vis_type=="2D plot":

                fig, ax = plt.subplots()
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap=color_palette, alpha=0.7)
                legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
                ax.add_artist(legend1)
                st.pyplot(fig)  # Display in Streamlit
            
     
        
        if n_components < 3:
            st.warning("Less than 3 features are selected,3D visualisation cannot be done")
        
        elif vis_type=="3D plot":
                fig = px.scatter_3d(
                x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
                color=labels.astype(str), title="3D Cluster Visualization", color_discrete_sequence=px.colors.qualitative.G10)
                st.plotly_chart(fig)
           
        
        # Step 9: Download Clustered Data
        st.write("### Download Clustered Dataset")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="clustered_data.csv",
            mime="text/csv"
        )

# Footer
st.write("### Created with Streamlit")



         

       