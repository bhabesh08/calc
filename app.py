import streamlit as st

st.set_page_config(page_title="Adder", page_icon="➕", layout="left")
st.title("Add Two Numbers")

# Use a form so the calculation happens only when you click the button
with st.form("adder_form"):
    number1 = st.number_input("Number-1", value=0.0, step=1.0, format="%.6f")
    number2 = st.number_input("Number-2", value=0.0, step=1.0, format="%.6f")
    submitted = st.form_submit_button("Calculate")

if submitted:
    result = number1 + number2
    st.success(f"Result: {result}")
