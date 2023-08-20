!pip install stramlit, pandas, seaborn, matplotlib, pandasai 
import streamlit as st
import pandas as pd
import seaborn as sns


import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI


llm = OpenAI(api_token="your-openapi-key")
pandas_ai = PandasAI(llm=llm)


st.title("DataSet(.csv) Analyzer")
st.subheader("Effortlessly Explore and Visualize Your CSV Dataset")

hide_menu = """
<style>
#MainMenu{
visibility: hidden
}
.css-cio0dv{
visibility: hidden
}
</style>"""
st.markdown(hide_menu, unsafe_allow_html=True)


uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    with st.sidebar:
        st.title("Navigation")
        section = st.radio(
            "Go to",
            [
                "Give Prompt",
                "Preview",
                "Basic Information",
                "Dataset Info",
                "Basic Statistics",
                "Data Visualization",
            ],
        )

    if section == "Give Prompt":
        st.subheader("Give prompt to analyze your data")
        prompt = st.text_area("Enter your prompt:")

        # Generate output
        if st.button("Generate"):
            if prompt:
                with st.spinner("Generating response..."):
                    st.write(pandas_ai.run(df, prompt))
            else:
                st.warning("Please enter a prompt.")

    elif section == "Preview":
        st.write(
            '<p style="font-size:30px;">Preview of the dataset</p>',
            unsafe_allow_html=True,
        )
        st.dataframe(df.head())

    elif section == "Basic Information":
        st.write(
            '<p style="font-size:30px;">Basic Information</p>', unsafe_allow_html=True
        )
        st.text(f"Number of Rows: {df.shape[0]}")
        st.text(f"Number of Columns: {df.shape[1]}")

    elif section == "Dataset Info":
        st.write('<p style="font-size:30px;">Dataset Info</p>', unsafe_allow_html=True)

        # Create a custom table for DataFrame info with an additional null count column
        info_dict = {
            "Non-Null Count": df.count(),
            "Data Type": df.dtypes,
            "Null Count": df.isnull().sum(),  # Calculate null counts for each column
        }
        info_df = pd.DataFrame(info_dict)
        info_df.index.name = "Column"

        st.table(info_df)

    elif section == "Basic Statistics":
        st.write(
            '<p style="font-size:30px;">Basic Stastics</p>', unsafe_allow_html=True
        )
        st.write(df.describe())

    elif section == "Data Visualization":
        st.write(
            '<p style="font-size:30px;">Data Visualization</p>', unsafe_allow_html=True
        )

        columns = df.columns
        all_option = st.checkbox("Select All Columns")
        if all_option:
            selected_columns = columns
        else:
            selected_columns = st.multiselect(
                "Select columns for visualization", columns
            )

        if len(selected_columns) >= 2:
            plot_type = st.selectbox("Select plot type", ["Heatmap", "Pair Plot"])

            if plot_type == "Heatmap":
                corr_matrix = df[selected_columns].corr()
                st.write(
                    "A heatmap is a graphical representation of data where values are visualized using a color scale to indicate their magnitude."
                )
                st.write(
                    "It is particularly useful for displaying patterns, correlations, or distributions within a dataset. Heatmaps are commonly used in data analysis to visually explore relationships between variables"
                )
                st.write("Correlation Heatmap:")
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
                plt.tight_layout()
                st.pyplot()

            elif plot_type == "Pair Plot":
                st.write("Pair Plot:")
                st.write(
                    "Pair plots are helpful for identifying potential correlations and distributions between variables, aiding in understanding the relationships and patterns within the data."
                )
                pair_plot = sns.pairplot(df[selected_columns])
                plt.tight_layout()
                st.pyplot()
