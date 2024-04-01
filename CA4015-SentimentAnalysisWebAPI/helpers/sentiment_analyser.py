from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import configparser
import torch

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
        sentiment_map = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]

        # Retrieve the URL of the model on HuggingFace with which the analysis shall be carried out.
        config = configparser.ConfigParser()
        config.read(config_path)
        sentiment_analysis_model_name = config.get("ModelNames", "SentimentAnalysisModelName")

        # Carry out a sentiment analysis on the specified piece of text.
        sentiment_analysis_model = AutoModelForSequenceClassification.from_pretrained(sentiment_analysis_model_name)
        sentiment_analysis_tokenizer = AutoTokenizer.from_pretrained(sentiment_analysis_model_name)
        input_ids = sentiment_analysis_tokenizer(relevant_text, return_tensors="pt")
        output = sentiment_analysis_model(**input_ids)

        # Identify the class with the highest probability, as this will be our predicted sentiment.
        sentiment_prediction = torch.argmax(output.logits, dim=1)

        # Return the sentiment under which said text was categorised.
        return sentiment_map[sentiment_prediction]
