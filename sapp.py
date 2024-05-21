import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

st.title('NER model')
pipe = pipeline("token-classification", model = "./Model1", tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased"))
x = st.text_input('Enter a custom message:', 'Hello, Streamlit!')
if x:
    prediction = pipe(x)
    data = str(prediction)
    st.write(data)
