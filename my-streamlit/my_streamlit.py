import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Sample App", layout="wide")

st.title("Welcome to My Streamlit App")

st.write("This is a sample Streamlit application.")

# Sidebar
st.sidebar.header("Options")
name = st.sidebar.text_input("Enter your name:")
age = st.sidebar.slider("Select your age:", 0, 100, 25)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Section")
    if name:
        st.write(f"Hello, {name}!")
    st.write(f"Your age: {age}")

with col2:
    st.subheader("Data Display")
    st.metric("Age Value", age)

# Button interaction
if st.button("Click me!"):
    st.success("Button clicked!")

# Simple chart
st.subheader("Sample Chart")

data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['A', 'B', 'C']
)
st.line_chart(data)