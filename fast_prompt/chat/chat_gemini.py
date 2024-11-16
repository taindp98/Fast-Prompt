import base64
from dotenv import load_dotenv
import os
import requests
import json
import re
from collections import defaultdict
import multiprocessing as mp
from tqdm import tqdm

class ChatGemini:
    """
    A class to interact with Gemini's language model for generating chat responses.

    Attributes:
        llm_model (str): The language model to use, default is "gpt-4o-mini".
        client (Gemini): The Gemini client initialized with the API key.

    Methods:
        request(prompt: str, question: str, temperature: float, top_p: float, max_tokens: int, **kwargs):
            Sends a request to the Gemini API with the provided prompt and question, and returns the response.
    """
    def __init__(
        self,
        max_tokens: int = 4096,
        temperature: float = 1e-4
    ) -> None:

        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={}"
        self.api_url = self.api_url.format(os.getenv("GEMINI_API_KEY"))
        self.headers = {
            "Content-Type": "application/json"
        }
        
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
    def encode_image(self, image_fname):
        """
        Encodes an image file to a base64 string.

        Args:
            image_fname (str): The path to the image file.

        Returns:
            str: The base64-encoded string of the image.
        """
        with open(image_fname, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _clean(self, text: str):
        text = text.replace("\n", " ")
        return text
    
    def _post_processing(self, scene_texts: list):
        # Initialize a defaultdict to hold grouped texts
        grouped_texts = defaultdict(list)

        # Iterate through the list of scene texts
        for item in scene_texts:
            position = item['position']
            text = item['text']
            grouped_texts[position].append(text)

        # Convert back to regular dict if needed
        grouped_texts = dict(grouped_texts)
        scene_sentence = []
        for k, v in grouped_texts.items():
            scene_sentence.append(
                self.descriptions[k].format(
                    json.dumps(v, ensure_ascii=False).replace(r"[", "").replace(r"]", "")
                )
            )
        scene_sentence = ". ".join(scene_sentence)
        return scene_sentence
    
    def predict(self, image_fname: str):
        """
        Sends a request to the Gemini API with the provided prompt and image URL or file path, and returns the response.

        Args:
            image_fname (str): The URL of the image or the local file path to the image.
        Returns:
            dict: A dictionary containing the request ID, output, completion tokens, prompt tokens, and total tokens.
        """

        base64_image = self.encode_image(image_fname)
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
                            "text": self.system_prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": self.max_tokens,
                "temperature": self.temperature,
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
        try:
            response = response.json()
            output = response["candidates"][0]["content"]["parts"][0]["text"]
            output = json.loads(output)
            for item in output["scene_texts"]:
                item["text"] = self._clean(item["text"])
            sentence = self._post_processing(output["scene_texts"])
        

            usage = response["usageMetadata"]
            result = {
                "output": output["scene_texts"],
                "sentence": sentence,
                "usage": usage,
            }
        except Exception as e:
            result = {
                "output": [{"position": "", "text": ""}],
                "sentence": "",
                "usage": {},
            }
        return result
    
    def batch_predict(self, image_fnames: list):
        with mp.Pool(processes=8) as pool:
            results = list(pool.map(self.predict, image_fnames))
        return results

if __name__ == "__main__":
    load_dotenv()
    image_fnames = ["data/016.jpg"]
    model = SceneTextInfer()
    contents = model.batch_predict(
        image_fnames=image_fnames
    )
    print(f"contents:{contents}")