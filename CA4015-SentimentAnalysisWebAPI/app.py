from flask import Flask, Response
from flask_cors import CORS
from helpers.sentiment_analyser import SentimentAnalyser

# Initialise the app.
app = Flask(__name__)

# Enable CORS.
CORS(app)

@app.route('/<relevant_text>', methods=['GET'])
def get_sentiment(relevant_text: str):
    """
    This endpoint allows ones to perform a sentiment analysis on a specified piece of text.
    :param relevant_text: The text to be analysed.
    :type relevant_text: str
    :return: A string that represents the sentiment class into which the given text fell.
    :rtype: str
    """
    # Carry out a sentiment analysis on the text in question.
    sentiment_category = SentimentAnalyser.get_sentiment(relevant_text)

    # Return the result of the sentiment analysis.
    return Response(f"'{relevant_text}' was categorised as '{sentiment_category}'.", status=200)


if __name__ == '__main__':
    app.run(threaded=True)
