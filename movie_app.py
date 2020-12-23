import sys

import streamlit as st
import joblib


def argmax(data: list):
    f = lambda i: data[i]
    return max(range(len(data)), key=f)


def main():
    mlb = joblib.load("movies_mlb.pkl")
    model = joblib.load("movies_model.pkl")
    class_name = [str(genre) for genre in mlb.classes_]

    st.title("Movie demo application")
    text = "Somehow during the tour she came into possession of a prototype transmitting device. We don't know how."
    text = st.text_area("Please enter a phrase", text)

    prediction_prob = model.predict_proba([text])
    pred_prob = []
    for pred_genre in prediction_prob:
        pred_prob.append(pred_genre[:, 1].tolist())

    result = class_name[argmax(pred_prob)]
    st.write("The likely movie genre is: ", result)


if __name__ == "__main__":
    main()
