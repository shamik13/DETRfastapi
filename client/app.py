import urllib.request

import requests
import streamlit as st

from .utils import display_image_from_response, response_from_server

st.header("WebApp with FastAPI")

user_input = st.text_input("Input Image URL", "")

st.header("Input Image")
if user_input:
    st.image(user_input, use_column_width=True)
    with open("./input_images/1.jpg", "wb") as f:
        f.write(urllib.request.urlopen(user_input).read())
    with open("./input_images/1.jpg", "rb") as image_file:
    prediction = response_from_server(api_url, image_file)
display_image_from_response(prediction)
