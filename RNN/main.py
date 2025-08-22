
import numpy as np 
import tensorflow as tf
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st



index=imdb.get_word_index()

reverseOrder=dict([(value,key) for (key,value) in index.items()])


model=load_model('rnn_model.h5')

#decode review
def decode(encoded):
    return ' '.join([reverseOrder.get(i -3, '?')for i in encoded])

#preprocessing 
def preprocess(text):
    words=text.lower().split()
    encoded =[index.get(word ,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded],maxlen=500)
    return padded_review


### Prediction  function

def predict_sentiment(review):
    preprocessed_input=preprocess(review)

    prediction=model.predict(preprocessed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    return sentiment, prediction[0][0]



st.title(" Review Sentiment Analysis")
st.write("Enter a movie review below to predict its sentiment (Positive or Negative).")

user_input=st.text_area("Movie Review","Type Here")

if st.button('Classify'):
    

    preprocesseed_input=preprocess(user_input)


    prediction=model.predict(preprocesseed_input)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'

    ## to display the result of prediction
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction score: {prediction[0][0]} ')
else:
    st.write("please enter a movie review ")


    
