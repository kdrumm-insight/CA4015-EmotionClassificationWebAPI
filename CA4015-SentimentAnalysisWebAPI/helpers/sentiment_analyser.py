from pathlib import Path
from transformers import pipeline
import configparser

# Create a reference to the config relative to this file.
config_path = Path(__file__).parent.parent / "configuration/config.ini"


class SentimentAnalyser:
    """
    This utility class can be used to perform a sentiment analysis on a specified piece of text.
    """

    @staticmethod
    def get_sentiment(relevant_text: str):
        """
        This function allows ones to perform a sentiment analysis on a specified piece of text.
        :param relevant_text: The text to be analysed.
        :type relevant_text: str
        :return: A string that represents the sentiment class into which the given text fell.
        :rtype: str
        """

        # Initialise a sentiment map for later use.
        sentiment_map = {
            "POS": "positive",
            "NEG": "negative",
            "NEU": "neutral"
        }

        # Retrieve the URL of the model on HuggingFace with which the analysis shall be carried out.
        config = configparser.ConfigParser()
        config.read(config_path)
        sentiment_analysis_model_name = config.get("ModelNames", "SentimentAnalysisModelName")

        # Carry out a sentiment analysis on the specified piece of text.
        sentiment_analysis_model = pipeline(model=sentiment_analysis_model_name)
        result = sentiment_analysis_model([relevant_text])

        # Return the sentiment under which said text was categorised.
        return sentiment_map.get(result[0]["label"], "Unknown")
