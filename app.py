import streamlit as st
import numpy as np
import pandas as pd
import pickle as pk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

with open("sentiment_pred.pkl", 'rb') as file:
    sentiment_model = pk.load(file)

with open("type_pred.pkl", 'rb') as file:
    type_model = pk.load(file)

def predictSentiment(text):
	text = [text]
	prediction = sentiment_model.predict(text)
	return prediction

def predictType(text):
	text = [text]
	prediction = type_model.predict(text)
	return prediction

def main():
	st.title("Predicting Response Sentiment")

	text = st.text_area("Text", height=500)

	st.title("Results")

	result = predictSentiment(text)
	typeText = predictType(text)

	for i in result:
		# st.write(i)
		st.write('Type of the caption: '+typeText[0])
		if i == 0:
			st.write('Sentiment of the replies: Negative')
		elif i == 2:
			st.write('Sentiment of the replies: Neutral')
		elif i == 4:
			st.write('Sentiment of the replies: Positive')

main()

