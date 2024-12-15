from flask import Flask, request, jsonify
from summarizer import hybrid_summary

app = Flask(__name__)

@app.route('/')
def summary_form():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Summarize Text</title>
    </head>
    <body>
        <h1>Welcome to the Hybrid Text Summarizer</h1>
        <p>Upload your text file and summarize it using the hybrid method:</p>

        <!-- Form to upload the text file -->
        <form action="http://127.0.0.1:8501" method="get" enctype="multipart/form-data">
            <label for="textFile">Choose a text file:</label>
            <input type="file" id="textFile" name="textFile" accept=".txt" required><br><br>
            
            <button type="submit">Upload and Summarize</button>
        </form>
        
        <p>After uploading, the summary will be shown in the Streamlit app.</p>

    </body>
    </html>

    '''

@app.route('/summarize', methods=['POST'])
def summarize():
    """API endpoint for summarization."""
    data = request.get_json()  # Use get_json() for JSON data
    text = data.get('text', '')  # Get the text from the JSON
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    summary = hybrid_summary(text)
    return jsonify({'summary': summary})

if __name__ == "__main__":
    app.run(debug=True)
