import streamlit as st

st.set_page_config(page_title="Adder", page_icon="➕", layout="centre")
st.title("Add Two Numbers")

with st.form("adder_form"):
    number1 = st.number_input("Number-1", value=0.0, step=1.0, format="%.0f")
    number2 = st.number_input("Number-2", value=0.0, step=1.0, format="%.0f")
    submitted = st.form_submit_button("Calculate")

if submitted:
    result = number1 + number2
    st.success(f"Result: {result}")
