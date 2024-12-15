import requests
import streamlit as st

st.title("Hybrid Text Summarizer")
st.write("Upload a text file. and get a summary combining extractive and abstractive techniques.")

#File upload
uploaded_file = st.file_uploader("Choose a text file", type = ["txt"])
if uploaded_file is not None:
    #read the file
    content = uploaded_file.read().decode("utf-8")
    st.text_area("Uploaded Content", content, height = 300)

    #Call Flask API
    if st.button("Summarize"):
        with st.spinner("Summarizing the text..."):
            #prepare the data to be sent to the flask api as JSON
            payload = {"text": content}
            headers = {"Content-Type": "application/json"}
        
            response = requests.post("http://127.0.0.1:5000/summarize", json = payload, headers = headers)
            if response.status_code == 200:
                summary = response.json().get('summary')
                st.subheader("Summary")
                st.write(summary)
            else:
                st.error("Error: Unable to summarize the text.")