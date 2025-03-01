import numpy as np
import pandas as pd
import streamlit as st
import joblib

model = joblib.load("newxG_modeldistance.pkl")

def calculate_distance(x, y, goal_x=120, goal_y=40):
    return np.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)

# Streamlit app
def main():
    st.title("Expected Goals (xG) Model")

    # User input
    x = st.number_input("Enter x coordinate (0-120):", min_value=0, max_value=120)
    y = st.number_input("Enter y coordinate (0-80):", min_value=0, max_value=80)

    # Predict xG
    if st.button("Predict xG"):
        distance = calculate_distance(x, y)
        xg = model.predict_proba([[distance]])[0, 1]
        st.write(f"Predicted xG: {xg:.3f}")


if __name__ == "__main__":
    main()


st.write("""How It Works

Welcome to our Expected Goals (xG) model! xG is a statistical measure used in football (soccer) to evaluate the quality of scoring chances and the likelihood of a shot resulting in a goal. This model predicts the likelihood of a goal being scored based on the position of the shot on the field.

What Does the Model Do?
         
When you input the distance from the goal (both horizontal and vertical) where the shot was taken, the model uses this information to calculate the expected goals (xG) for that shot. The xG value represents the probability that the shot will result in a goal based on historical data from past football matches. A higher xG indicates a greater likelihood of scoring, while a lower xG suggests a more challenging shot.

How Was the Model Created?
         
This model was built using historical football data from StatBomb. The dataset includes details about shots taken during matches, such as their position on the field, the outcome of the shot, and other relevant factors. Here’s a quick overview of the process used to build the model:


Data Collection: The model uses event data from StatBomb, which includes shot locations on the field (x/y coordinates).


Feature Engineering: We focused on the distance from the goal as a key feature for predicting xG. For simplicity, we normalized the shot coordinates to a scale based on the size of a football field (120 meters long and 80 meters wide).

Model Training: The model was trained using a logistic regression approach, which learns from the historical data to predict the probability of a shot resulting in a goal based on its distance from the goal.

What the output means:         
The xG prediction is a value between 0 and 1 that represents the likelihood of a goal being scored from a specific shot location. For example:

An xG of 0.75 means the shot has a 75% chance of resulting in a goal.
An xG of 0.05 means the shot has only a 5% chance of going in.
The model is based on data from thousands of past football shots, so it provides a statistically informed prediction based on the position of the shot.

Limitations

While the model provides an estimate of the likelihood of a goal, it is important to understand that this is a simplified version. The xG value does not take into account every factor that can influence the outcome of a shot, such as the angle, speed, or whether the shot is blocked or deflected. It only uses the shot's distance from the goal. Therefore, while it gives a good overall idea of shot quality, it’s not a perfect predictor of every individual shot outcome.""")
