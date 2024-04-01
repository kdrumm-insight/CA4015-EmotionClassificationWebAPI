from flask import Flask, Response
from flask_cors import CORS
from helpers.emotion_classifier import EmotionClassifier

# Initialise the app.
app = Flask(__name__)

# Enable CORS.
CORS(app)

@app.route('/<relevant_text>', methods=['GET'])
def classify_emotion(relevant_text: str):
    """
    This endpoint allows ones to perform an emotion classification on a specified piece of text.
    :param relevant_text: The text to be classified.
    :type relevant_text: str
    :return: A string that represents the emotion class into which the given text fell.
    :rtype: str
    """
    # Carry out an emotion classification on the text in question.
    emotion = EmotionClassifier.classify_emotion(relevant_text)

    # Return the result of the classification.
    return Response(f"'{relevant_text}' was categorised as '{emotion}'.", status=200)


if __name__ == '__main__':
    app.run(threaded=True)
