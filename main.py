import LangChain_Document as LC_Doc
import streamlit as st
import textwrap


st.title("PDF Helper")

with st.sidebar:
    with st.form(key="my_form"):
        uploaded_file = st.file_uploader("Choose a file")
        query = st.sidebar.text_area(label="What is your question", max_chars=50, key="query")


        submit_button = st.form_submit_button(label='Submut')



if uploaded_file and query:
    file_path = uploaded_file.name
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    db = LC_Doc.document_loader(file_path) 
    response = LC_Doc.get_response_from_query(db,query)
    st.subheader("Answer: ")
    st.text(textwrap.fill(response, width=80))