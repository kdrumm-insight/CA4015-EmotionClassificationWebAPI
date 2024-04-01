from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import configparser
import torch

# Create a reference to the config relative to this file.
config_path = Path(__file__).parent.parent / "configuration/config.ini"


class EmotionClassifier:
    """
    This utility class can be used to perform an emotion classification on a specified piece of text.
    """

    @staticmethod
    def classify_emotion(relevant_text: str):
        """
        This function allows ones to perform an emotion classification on a specified piece of text.
        :param relevant_text: The text to be analysed.
        :type relevant_text: str
        :return: A string that represents the emotion class into which the given text fell.
        :rtype: str
        """

        # Initialise a list of possible classes for later use.
        possible_classes = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]

        # Retrieve the URL of the model on HuggingFace with which the analysis shall be carried out.
        config = configparser.ConfigParser()
        config.read(config_path)
        emotion_classiciation_model_name = config.get("ModelNames", "EmotionClassificationModelName")

        # Carry out an emotion classification on the specified piece of text.
        emotion_classification_model = AutoModelForSequenceClassification.from_pretrained(emotion_classiciation_model_name)
        emotion_classification_tokenizer = AutoTokenizer.from_pretrained(emotion_classiciation_model_name)
        input_ids = emotion_classification_tokenizer(relevant_text, return_tensors="pt")
        output = emotion_classification_model(**input_ids)

        # Identify the class with the highest probability, as this will be our predicted emotion.
        predicted_emotion_index = torch.argmax(output.logits, dim=1)

        # Return the emotion under which said text was categorised.
        return possible_classes[predicted_emotion_index]
