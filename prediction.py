# Importing ToolKits
import pandas as pd
import numpy as np
import plotly.express as px

import streamlit as st
import warnings

# Function to create matrix score cards
def create_matrix_score_cards(card_image="", card_title="Card Title", card_value=None, percent=False):
    st.image(card_image, caption="", width=70)
    st.subheader(card_title)
    
    if percent:
        st.subheader(f"{card_value}%")
    else:
        st.subheader(f"{card_value}")

# Function to create a comparison DataFrame
def create_comparison_df(y_actual, y_pred):
    predicted_df = pd.DataFrame()
    predicted_df["Actual Spent Values"] = y_actual
    predicted_df.reset_index(drop=True, inplace=True)
    predicted_df["Predicted Spent Values"] = y_pred
    return predicted_df

# Function to create a confusion matrix plot
def create_confusion_plot(cm):
    fig = px.imshow(cm, aspect=True, text_auto="0.0f", template="plotly_dark",
                    color_continuous_scale="greens", x=["Stay", "Left"], y=["Stay", "Left"], height=550)
    fig.update_traces(
        textfont={
            "size": 15,
            "family": "consolas"
        }
    )
    return fig
