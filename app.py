import streamlit as st

# Page setup (title and icon you see in the browser tab)
st.set_page_config(page_title="Adder", page_icon="➕", layout="centered")

st.title("Add Two Numbers")

# A simple form so calculation runs only when you click the button
with st.form("adder_form"):
    # Ask for the two numbers
    number1 = st.number_input("Number-1", value=0.0, step=1.0, format="%.6f")
    number2 = st.number_input("Number-2", value=0.0, step=1.0, format="%.6f")

    # The button that triggers the addition
    submitted = st.form_submit_button("Calculate")

# After clicking, show the result
if submitted:
    result = number1 + number2
    st.success(f"Result: {result}")
