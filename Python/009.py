import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Streamlit app title
st.title("CSV Data Analyzer")

# File uploader for CSV files
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Check if a file is uploaded
if uploaded_file is not None:
    # Load the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())

    # Column selection for analysis
    columns = st.multiselect("Select columns for analysis:", options=df.columns)

    if columns:
        # Display summary statistics for selected columns
        st.subheader("Summary Statistics")
        st.write(df[columns].describe())

        # Visualization options
        st.subheader("Visualizations")
        plot_type = st.selectbox("Select plot type:", ["Histogram", "Box Plot"])

        for column in columns:
            st.write(f"**{plot_type} for {column}**")

            # Generate a histogram
            if plot_type == "Histogram":
                plt.figure(figsize=(10, 4))
                plt.hist(df[column].dropna(), bins=30, edgecolor="black")
                plt.title(f"Histogram of {column}")
                plt.xlabel(column)
                plt.ylabel("Frequency")
                st.pyplot(plt)
                plt.clf()

            # Generate a box plot
            elif plot_type == "Box Plot":
                plt.figure(figsize=(10, 4))
                plt.boxplot(df[column].dropna(), vert=False)
                plt.title(f"Box Plot of {column}")
                plt.xlabel(column)
                st.pyplot(plt)
                plt.clf()
else:
    st.write("Please upload a CSV file to begin.")
