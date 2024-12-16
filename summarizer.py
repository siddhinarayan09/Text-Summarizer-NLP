import spacy
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

#loading the spacy model
nlp = spacy.load("en_core_web_sm")

#initialise the Hugging Face summarizer
summarizer = pipeline("summarization", model = "facebook/bart-large-cnn")


def extractive_summary(text, n_sentences = 3):
    """Extractive summarization using TF-IDF to rank sentences.
    Select the top sentences based on their relevence to the text.
    """

    doc = nlp(text)
    sentences = list(doc.sents)
    sentence_texts = [str(s) for s in sentences]

    #Compute TF-IDF scores for sentences
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentence_texts)
    scores = tfidf_matrix.sum(axis = 1).A1 #sum the TF-IDF scores fro each sentence

    #rank the sentences by their scores
    ranked_sentences = [
        sent for _, sent in sorted(zip(scores, sentences), reverse = True)
    ]
    return " ".join(str(sent) for sent in ranked_sentences[:n_sentences])

def chunk_text(text, max_length = 1024):
    """chunk the text to avoid exceeding the token limits."""
    words = text.split()
    for i in range(0, len(words), max_length):
        yield " ".join(words[i:i + max_length])


def abstractive_summary(text):
    """Generate abstractive summary using Hugging Face Transformers."""

    max_input_tokens = 1024 

    #chunk the text if it exceeds the token limit
    chunks = list(chunk_text(text, max_length = max_input_tokens))
    summaries = []

    for chunk in chunks:
        summary = summarizer(chunk, max_length = 130, min_length = 30, do_sample = False)
        summaries.append(summary[0]["summary_text"])

    return " ".join(summaries)


def hybrid_summary(text, n_sentences = 3):
    """Combine extractive and abstractive summaries.
    first generates an extractive summary which then is refined with the abstractive summary."""
    extractive = extractive_summary(text, n_sentences = n_sentences)
    abstractive = abstractive_summary(extractive)
    return abstractive 
 