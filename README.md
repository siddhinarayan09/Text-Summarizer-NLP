# Hybrid Text Summarizer with Flask and Streamlit

This project is a hybrid text summarization application combining extractive (TF-IDF) and abstractive (Hugging Face Transformers) techniques. It features a Flask backend for API support and a Streamlit frontend for an interactive user experience. Users can upload text files and receive a concise summary using advanced NLP methods.


---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [How It Works](#how-it-works)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Output](#output)
- [Future Enhancements](#future-enhancements)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project provides a hybrid approach to summarization by combining:

Extractive Summarization: Selects key sentences using TF-IDF scores.
Abstractive Summarization: Generates summaries using Hugging Face’s facebook/bart-large-cnn model.
Hybrid Summarization: Refines extractive summaries through abstractive techniques for improved results.

---

## Features

-**File Upload Support:** Accepts text files for summarization.
-**Hybrid Summarization:** Combines extractive and abstractive methods.
-**Interactive UI:** Streamlit frontend for user-friendly interactions.
-**API Integration:** Flask backend for scalable summarization services.

---

## Technologies Used

- **Backend**: Flask
- **Frontend**: Streamlit
- **NLP Model**:
    -SpaCy for sentence tokenization.
    -Hugging Face Transformers (facebook/bart-large-cnn) for abstractive summarization.
    -Scikit-learn's TfidfVectorizer for extractive summarization.
- **Libraries**:
    - Requests for API communication.
    -Streamlit for frontend development.

---

## How It Works

1. **Input**:
   - User uploads a text file via the Streamlit interface.
     
2. **Summarization**:
   - The text is processed and tokenized using SpaCy.
   -Extractive summarization ranks sentences by relevance using TF-IDF scores.
   -Abstractive summarization generates refined summaries using Hugging Face's model.
   -The hybrid approach combines both methods for optimal results.

4. **Output**:
   -The summary is displayed in the Streamlit app.

---

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- Pipenv or pip for Python dependency management

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone <https://github.com/siddhinarayan09/Text-Summarizer-NLP>
   cd Text-Summarizer-NLP
2. **Set Up the Backend**:
  Create a virtual environment and install dependencies:
   ```bash
    pip install -r requirements.txt

3. **Start the Flask App**:
   ```bash
     python app.py

3. **Run the Streamlit Frontend**:
   ```bash
     streamlit run streamlit_app.py
   
## Usage

Launch the Streamlit app.
Upload a text file for summarization.
Click "Summarize" to generate a hybrid summary.
View the results dynamically in the app.

## Output

The application produces:



## Future Enhancements

**Support for Multiple File Formats:**
Add support for PDFs and Word documents.
**Advanced Visualizations:**
Use Plotly for interactive visual outputs.
**Authentication:**
Allow user accounts to save and access summaries.
**Language Support:**
Extend the model to handle multilingual text.

## Acknowledgments

**Hugging Face:** For providing the facebook/bart-large-cnn model.
**SpaCy:** For efficient text preprocessing.
**Streamlit:** For building an interactive frontend.
**Flask:** For lightweight backend API development.

Inspiration from NLP projects and article summarizers.

*Thank you for exploring the project:)*




