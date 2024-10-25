# config/config.py
import os
import openai
import dotenv


class config:
    def __init__(self):
        self.openai_api_key = "Your API key"

    def setup_openai(self):
        openai.api_key = self.openai_api_key
        return openai