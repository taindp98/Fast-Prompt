import base64
from dotenv import load_dotenv
import os
import requests
import json
import re
from collections import defaultdict
import multiprocessing as mp
from tqdm import tqdm
from IPython.display import Image, display
from fast_prompt.chat.utils import unit_price, encode_image
import uuid

load_dotenv()

class ChatGemini:
    """
    A class to interact with Gemini's language model for generating chat responses.

    Attributes:
        client (Gemini): The Gemini client initialized with the API key.

    Methods:
        request(prompt: str, question: str, temperature: float, top_p: float, max_tokens: int, **kwargs):
            Sends a request to the Gemini API with the provided prompt and question, and returns the response.
    """
    def __init__(
        self,
        llm_model: str = "gemini-1.5-pro",
    ) -> None:
        supported_llm_models = list(unit_price.keys())
        assert (
            llm_model in supported_llm_models
        ), f"`llm_model` should be in {supported_llm_models}"
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{llm_model}:generateContent?key={os.getenv('GEMINI_API_KEY')}"
        self.headers = {
            "Content-Type": "application/json"
        }
    
    def request(
            self,
            system_prompt: str,
            user_prompt: str,
            max_tokens: int = 1024,
            temperature: float = 1e-4,
    ):

        contents = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": user_prompt
                        }
                    ]
                }
            ],
            "systemInstruction": {
                "parts": [
                    {
                        "text": system_prompt
                    }
                ]
            },
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
                "topP": 0.95
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }
        payload = json.dumps(contents)
        response = requests.post(
            url=self.api_url,
            headers=self.headers,
            data=payload,
        )
        response = response.json()
        usage = response["usageMetadata"]
        output = response["candidates"][0]["content"]["parts"][0]["text"]
        result = {
            "request_id": str(uuid.uuid4()),
            "llm_model": response["modelVersion"],
            "input": contents, 
            "output": output,
            "complete_tokens": usage["candidatesTokenCount"],
            "prompt_tokens": usage["totalTokenCount"],
        }
        return result
    
    def batch_request(self, image_paths: list):
        with mp.Pool(processes=8) as pool:
            results = list(pool.map(self.request, image_paths))
        return results

class ChatVisionGemini:
    """
    A class to interact with Gemini's language model for generating chat responses.

    Attributes:
        client (Gemini): The Gemini client initialized with the API key.

    Methods:
        request(prompt: str, question: str, temperature: float, top_p: float, max_tokens: int, **kwargs):
            Sends a request to the Gemini API with the provided prompt and question, and returns the response.
    """
    def __init__(
        self,
        llm_model: str = "gemini-1.5-pro",
    ) -> None:
        supported_llm_models = list(unit_price.keys())
        assert (
            llm_model in supported_llm_models
        ), f"`llm_model` should be in {supported_llm_models}"
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{llm_model}:generateContent?key={os.getenv('GEMINI_API_KEY')}"
        self.headers = {
            "Content-Type": "application/json"
        }
    
    def request(
            self, image_path: str,
            user_prompt: str,
            max_tokens: int = 1024,
            temperature: float = 1e-4,
            show_preview: bool = False,
    ):
        """
        Sends a request to the Gemini API with the provided prompt and image URL or file path, and returns the response.

        Args:
            image_path (str): The URL of the image or the local file path to the image.
        Returns:
            dict: A dictionary containing the request ID, output, completion tokens, prompt tokens, and total tokens.
        """
        if show_preview:
            display(Image(filename=image_path))

        base64_image = encode_image(image_path)
        contents = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": base64_image
                            }
                        },
                        {
                            "text": user_prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
                "topP": 0.95
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }
        payload = json.dumps(contents)
        response = requests.post(
            url=self.api_url,
            headers=self.headers,
            data=payload,
        )
        response = response.json()
        usage = response["usageMetadata"]
        output = response["candidates"][0]["content"]["parts"][0]["text"]
        result = {
            "request_id": str(uuid.uuid4()),
            "llm_model": response["modelVersion"],
            "input": contents, 
            "output": output,
            "complete_tokens": usage["candidatesTokenCount"],
            "prompt_tokens": usage["totalTokenCount"],
        }
        return result
    
    def batch_request(self, image_paths: list):
        with mp.Pool(processes=8) as pool:
            results = list(pool.map(self.request, image_paths))
        return results