import spacy
from transformers import pipeline

#loading the spacy model
nlp = spacy.load("en_core_web_sm")

#initialise the Hugging Face summarizer
summarizer = pipeline("summarization", model = "facebook/bart-large-cnn")

def extractive_summary(text, n_sentences = 3):
    """Extract the key sentences using SpaCy."""
    doc = nlp(text)
    sentences = list(doc.sents)
    #sort sentences by length (as a basic heuristic)
    sorted_sentences = sorted(sentences, key = lambda s: len(s), reverse = True)
    return " ".join(str(s) for s in sorted_sentences[:n_sentences])

def abstractive_summary(text):
    """Generate abstractive summary using Hugging Face Transformers."""
    summary = summarizer(text, max_length = 130, min_length = 30, do_sample = False)
    return summary[0]['summary_text']

def hybrid_summary(text):
    """Combine extractive and abstractive summaries."""
    extractive = extractive_summary(text)
    combined_text = extractive + " " + text
    abstractive = abstractive_summary(combined_text)
    return abstractive 
 