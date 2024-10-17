# config/config.py
import os
import openai



class config:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

    def setup_openai(self):
        openai.api_key = self.openai_api_key
        return openai