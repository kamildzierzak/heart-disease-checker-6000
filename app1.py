# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
startTime = datetime.now()

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

filename = "model.sv"
model = pickle.load(open(filename,'rb'))

sex_d = {0:"Kobieta", 1:"Mężczyzna"}
exercise_induced_angina_d = {0:"Nie", 1:"Tak"}
chest_pain_type_d = {0:"Typowa angina", 1:"Atypowa angina", 2:"Ból inny niż angina", 3:"Bezobjawowy"}
fasting_blood_sugar_d = {0:"Nie", 1:"Tak"}
resting_electrocardiographic_d = {0:"Normalne", 1:"Nieprawidłowości w załamkach ST-T (np. odwrócenie załamka T, podniesienie lub obniżenie ST > 0,05 mV)", 2:"Przerost lewej komory serca (zgodnie z kryteriami Estesa)"}
st_slope_d = {0:"W dół", 1:"Płaski", 2:"W górę"}

def main():

	st.set_page_config(page_title="Heart Disease Checker 6000")
	overview = st.container()
	left, right = st.columns(2)
	prediction = st.container()

	st.image("https://images.unsplash.com/photo-1623134915837-d2fdb4f59035?q=80&w=1171&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")

	with overview:
		st.title("Heart Disease Checker 6000")

	with left:
		sex_radio = st.radio( "Płeć", list(sex_d.keys()), format_func=lambda x : sex_d[x] )
		exercise_induced_angina_radio = st.radio("Angina wywołana wysiłkiem", list(exercise_induced_angina_d.keys()), format_func=lambda x : exercise_induced_angina_d[x] )
		chest_pain_type_radio = st.radio( "Rodzaj bólu w klatce piersiowej", list(chest_pain_type_d.keys()), format_func=lambda x : chest_pain_type_d[x] )
		fasting_blood_sugar_radio = st.radio( "Cukier na czczo> 120 mg/dl", list(fasting_blood_sugar_d.keys()), format_func=lambda x : fasting_blood_sugar_d[x] )
		resting_electrocardiographic_radio = st.radio( "Wyniki EKG w spoczynku", list(resting_electrocardiographic_d.keys()), format_func=lambda x : resting_electrocardiographic_d[x] )
		st_slope_radio = st.radio( "Nachylenia odcinka ST", list(st_slope_d .keys()), format_func=lambda x : st_slope_d [x] )

	with right:
		age_slider = st.slider("Wiek", value=50, min_value=28, max_value=77)
		resting_blood_pressure_slider = st.slider("Ciśnienie krwi w spoczynku (mm Hg)", value=120, min_value=0, max_value=200)
		cholesterol_slider = st.slider("Cholesterol", value=200, min_value=0, max_value=603)
		maximum_heart_rate = st.slider("Maksymalne tętno", value=140, min_value=60, max_value=202)
		oldpeak_slider = st.slider("Poprzedni szczyt", value=0.0, min_value=-2.5, max_value=6.2)

	feature_names = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
	data = pd.DataFrame([[age_slider, sex_radio, chest_pain_type_radio, resting_blood_pressure_slider, cholesterol_slider, fasting_blood_sugar_radio, resting_electrocardiographic_radio, maximum_heart_rate, exercise_induced_angina_radio, oldpeak_slider, st_slope_radio]], columns=feature_names)
	survival = model.predict(data)
	s_confidence = model.predict_proba(data)

	with prediction:
		st.subheader("Czy wystąpi choroba serca?")
		st.subheader(("Tak" if survival[0] == 1 else "Nie"))
		st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))

if __name__ == "__main__":
    main()
