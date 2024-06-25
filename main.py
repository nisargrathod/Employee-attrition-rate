# Importing ToolKits
import re
import vizualizations
import prediction

from time import sleep
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix

import streamlit as st
from streamlit_option_menu import option_menu
import warnings

def run():
    st.set_page_config(
        page_title="Employee Attrition Rate",
        page_icon="📊",
        layout="wide"
    )

    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Function To Load Our Dataset
    @st.cache_data
    def load_data(the_file_path):
        df = pd.read_csv(the_file_path)
        df.columns = df.columns.str.replace(" ", "_").str.replace(".", "")
        df.drop_duplicates(inplace=True)
        df.reset_index(inplace=True, drop=True)
        return df

    # Function To Load Our Model
    @st.cache_resource
    def load_the_model(model_path):
        return pd.read_pickle(model_path)

    df = load_data("HR_comma_sep.csv")
    model = load_the_model("Employee Attrition Rate(Nisarg).pkl")

    # Function To Validate Input Data
    @st.cache_data
    def is_valid_data(d):
        letters = list("qwertyuiopasdfghjklzxcvbnm@!#$%^&*-+~")
        return len(d) >= 2 and not any([i in letters for i in list(d)])

    @st.cache_data
    def validate_test_file(test_file_columns):
        pa = """satisfaction_level
last_evaluation
average_montly_hours
time_spend_company
"""
        col = "\n".join(test_file_columns).lower()
        pattern = re.compile(pa)
        matches = pattern.findall(col)
        return len("\n".join(matches).split("\n")) == 4

    st.markdown(
        """
    <style>
         .main {
            text-align: center; 
         }
         .st-emotion-cache-16txtl3 h1 {
         font: bold 29px arial;
         text-align: center;
         margin-bottom: 15px;
         }
         div[data-testid=stSidebarContent] {
         background-color: #111;
         border-right: 4px solid #222;
         padding: 8px!important;
         }
         div.block-container{
            padding-top: 0.5rem;
         }
         .st-emotion-cache-z5fcl4{
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 1.1rem;
            padding-right: 2.2rem;
            overflow-x: hidden;
         }
         .st-emotion-cache-16txtl3{
            padding: 2.7rem 0.6rem;
         }
         .plot-container.plotly{
            border: 1px solid #333;
            border-radius: 6px;
         }
         div.st-emotion-cache-1r6slb0 span.st-emotion-cache-10trblm{
            font: bold 24px tahoma;
         }
         div[data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
        div[data-baseweb=select]>div{
            cursor: pointer;
            background-color: #111;
            border: 2px solid #17B794;
        }
        div[data-baseweb=base-input]{
            background-color: #111;
            border: 4px solid #444;
            border-radius: 5px;
            padding: 5px;
        }
        div[data-testid=stFormSubmitButton]> button{
            width: 40%;
            background-color: #111;
            border: 2px solid #17B794;
            padding: 18px;
            border-radius: 30px;
            opacity: 0.8;
        }
        div[data-testid=stFormSubmitButton]  p{
            font-weight: bold;
            font-size: 20px;
        }
        div[data-testid=stFormSubmitButton]> button:hover{
            opacity: 1;
            border: 2px solid #17B794;
            color: #fff;
        }
    </style>
    """,
        unsafe_allow_html=True
    )

    side_bar_options_style = {
        "container": {"padding": "0!important", "background-color": 'transparent'},
        "icon": {"color": "white", "font-size": "16px"},
        "nav-link": {"color": "#fff", "font-size": "18px", "text-align": "left", "margin": "0px", "margin-bottom": "15px"},
        "nav-link-selected": {"background-color": "#17B794", "font-size": "15px"},
    }

    sub_options_style = {
        "container": {"padding": "3!important", "background-color": '#101010', "border": "2px solid #0000"},
        "nav-link": {"color": "white", "padding": "12px", "font-size": "18px", "text-align": "center", "margin": "0px", },
        "nav-link-selected": {"background-color": "#17B794"},
    }

    header = st.container()
    content = st.container()

    with st.sidebar:
        st.title(":green[Forecasting] Employee Attrition Rate")
        st.title("Developed by- Nisarg Rathod")
        st.image("imgs/division.png", caption="", width=90)
        page = option_menu(
            menu_title=None,
            options=['Home', 'Visualizations', 'Prediction'],
            icons=['diagram-3-fill', 'bar-chart-line-fill', "graph-up-arrow"],
            menu_icon="cast",
            default_index=0,
            styles=side_bar_options_style
        )
        st.write("***")

        data_file = st.file_uploader("Upload Your Dataset 📂", type="csv")

        if data_file is not None:
            if data_file.name.split(".")[-1].lower() != "csv":
                st.error("Please, Upload CSV FILE ONLY")
            else:
                df = pd.read_csv(data_file)

        # Home Page
        if page == "Home":
            with header:
                st.header('Employee Retention Classification 👨‍💼')

            with content:
                st.dataframe(df.sample(frac=0.25, random_state=35).reset_index(drop=True),
                             use_container_width=True)
                st.write("***")
                st.subheader("Data Summary Overview")

                len_numerical_data = df.select_dtypes(include="number").shape[1]
                len_string_data = df.select_dtypes(include="object").shape[1]

                if len_numerical_data > 0:
                    st.subheader("Numerical Data [123]")
                    data_stats = df.describe().T
                    st.table(data_stats)

                if len_string_data > 0:
                    st.subheader("String Data [𝓗]")
                    data_stats = df.select_dtypes(include="object").describe().T
                    st.table(data_stats)

        # Visualizations
        if page == "visualizations":
            with header:
                st.header("Data Visualizations 📉")

            with content:
                # Numerical Columns
                visualizations.create_visualization(
                    df, viz_type="box", data_type="number")

                # Categorical Columns
                visualizations.create_visualization(
                    df, viz_type="bar", data_type="object")

                # Less Than 4 Values
                visualizations.create_visualization(
                    df, viz_type="pie")

                st.plotly_chart(visualizations.create_heat_map(df),
                                use_container_width=True)

        # Prediction Model
        if page == "Prediction":
            with header:
                st.header("🏦 Prediction Model 👨‍💼")
                prediction_option = option_menu(menu_title=None, options=["Predict Employee Expected To STAY/LEAVE"],
                                                icons=[" "], menu_icon="cast", default_index=0,
                                                orientation="horizontal", styles=sub_options_style)

            with content:
                if prediction_option == "Predict Employee Expected To STAY/LEAVE":
                    with st.form("Predict_value"):
                        c1, c2 = st.columns(2)
                        with c1:
                            satisfaction_level = np.round(st.number_input(
                                'Satisfaction Level', min_value=0.05, max_value=1.0, value=0.66), 2)
                            last_evaluation = np.round(st.number_input(
                                'Last Evaluation', min_value=0.05, max_value=1.0, value=0.54), 2)

                        with c2:
                            avg_monthly_hours = st.number_input(
                                'Average Monthly Hours', min_value=50, max_value=320, step=1, value=120)
                            time_in_company = st.number_input(
                                'Time In Company', min_value=1, max_value=20, step=1, value=5)

                        salary_category = st.selectbox(
                            "Salary", options=["low", "medium", "high"])

                        st.write("")  # Space

                        predict_button = st.form_submit_button(
                            label='Predict', use_container_width=True)
                        st.write("***")  # Space

                        if predict_button:
                            salary = [0, 0]  # High Salary
                            if salary_category == "low":
                                salary = [1, 0]
                            elif salary_category == "medium":
                                salary = [0, 1]

                            with st.spinner(text='Predicting the Value..'):
                                new_data = [satisfaction_level, last_evaluation, avg_monthly_hours, time_in_company]
                                new_data.extend(salary)
                                predicted_value = model.predict([new_data])[0]
                                sleep(1.2)
                                prediction_prop = np.round(
                                    model.predict_proba([new_data]) * 100)

                                predicted_col, leave_prop, stay_prop = st.columns(
                                    3)

                                with predicted_col:
                                    if predicted_value == 0:
                                        st.image("imgs/manager.png",
                                                 caption="", width=70)
                                        st.subheader("Employee Expected To")
                                        st.subheader(":green[STAY]")

                                    else:
                                        st.image("imgs/turnover.png",
                                                 caption="", width=70)
                                        st.subheader(f"Employee Expected To")
                                        st.subheader(":red[LEAVE]")

                                with leave_prop:
                                    st.image("imgs/discount.png",
                                             caption="", width=70)
                                    st.subheader("Probability To Stay")
                                    st.subheader(f"{prediction_prop[0, 0]}%")

                                with stay_prop:
                                    st.image("imgs/discount.png",
                                             caption="", width=70)
                                    st.subheader("Probability To Leave")
                                    st.subheader(f"{prediction_prop[0, 1]}%")

run()
