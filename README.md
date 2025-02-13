<h1>
  <!-- <img alt="lograg_bg" src="./docs/images/lograg_bg.png" style="width: 100%;"> -->
  <p align="center">
	Fast-Prompt
  </p>
</h1>

<p align="center">
    <img alt="OpenAI" src="https://img.shields.io/badge/OpenAI-000000?logo=openai&logoColor=white&style=flat" />
<a href="./notebooks/example.ipynb">
        <img alt="Jupyter Notebook" src="https://img.shields.io/badge/Jupyter_Notebook-F37626?logo=jupyter&logoColor=white&style=flat" />
    </a>
</p>

Fast-Prompt is a simple module designed to capture and manage API requests for large language models.

## Table of Contents

- [Update](#update)
- [Quickstart](#quickstart)
- [Contributing](#contributing)

## Update
- [TBD]: Support more LLM APIs providers.
- **Nov 16th, 2024**: Support Gemini Pro API for both text and vision.
- **August 12th, 2024**: Support for the Batch API to OpenAI inference, reducing costs by half.
- **August 1st, 2024**: Support OpenAI API usage for both text [`ChatOpenAI`](/fast_prompt/chat/chat_openai.py) and vision [`ChatVisionOpenAI`](/fast_prompt/chat/chat_openai.py) requests.

## Quickstart

The structure of source code:

```
├── LICENSE             <- Open-source license if one is chosen
├── Makefile            <- Makefile with convenience commands like `make data` or `make train`
├── README.md           <- The top-level README for developers using this project.
├── notebooks           <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml      <- Project configuration file with package metadata for fast_prompt
├── requirements.txt    <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg           <- Configuration file for flake8
├── .env                <- LLM provider API key.
└── fast_prompt         <- Source code for use in this project.
```

Let’s install the library

```bash
git clone https://github.com/taindp98/Fast-Prompt.git
cd Fast-Prompt
pip install -r requirements.txt
```

alternative installation from a specific commit hash, use the following command:

```bash
pip install git+https://github.com/taindp98/Fast-Prompt.git@commit-hash
```

Generate an API key from your chosen LLM provider and put the API key in the `.env` file located in the root path of this repository:

```
OPENAI_API_KEY=sk-***
```

An example for logging a certain LLM request:

```python
from fast_prompt.chat.chat_openai import ChatOpenAI

llm = ChatOpenAI()

system_prompt = """
    You possess in-depth knowledge of natural images and can describe them with ease. \
    From the given input text indicating the category name of a certain object, your task involves the following steps:
    1-Imagine a scene containing the input object.
    2-Generate 4 descriptions about different key appearance features of the input object from the imagined scene, with each description having a maximum of 16 words.
    3-Output a JSON object containing the following key:
        "description": <list of 4 descriptions>

    Use the following examples:
        Input text: "sea lion"
        Answer: "description": ["A round-bellied seal sits on a rock, looking intently at something off-camera.", "The seal lies with flippers tucked, sleek body well-maintained.", "The seal's thick, smooth fur and large dark eyes show alertness and curiosity.", "Turquoise water contrasts with the seal's brown fur and grey rock, highlighting its natural environment."]
"""
response = llm.request(system_prompt=system_prompt, user_prompt="British shorthair")
print(response)
```

the output follows:

```bash
{
  "request_id": "chatcmpl-9t5nxsAIVMEcbsT4yrd3B1Y7iIpLG",
  "llm_model": "gpt-3.5-turbo-instruct",
  "input": [
    {
      "role": "system",
      "content": "\n    You possess in-depth knowledge of natural images and can describe them with ease. From the given input text indicating the category name of a certain object, your task involves the following steps:\n    1-Imagine a scene containing the input object.\n    2-Generate 4 descriptions about different key appearance features of the input object from the imagined scene, with each description having a maximum of 16 words.\n    3-Output a JSON object containing the following key:\n        \"description\": <list of 4 descriptions>\n\n    Use the following examples:\n        Input text: \"sea lion\"\n        Answer: \"description\": [\"A round-bellied seal sits on a rock, looking intently at something off-camera.\", \"The seal lies with flippers tucked, sleek body well-maintained.\", \"The seal's thick, smooth fur and large dark eyes show alertness and curiosity.\", \"Turquoise water contrasts with the seal's brown fur and grey rock, highlighting its natural environment.\"]\n"
    },
    {
      "role": "user",
      "content": "British shorthair"
    }
  ],
  "output": {
    "description": [
      "A fluffy British Shorthair cat lounges on a cozy armchair, eyes half-closed in contentment.",
      "Its round face and chubby cheeks give it an adorable, teddy bear-like appearance.",
      "The cat's dense, plush coat is a striking blue-gray color, adding to its charm.",
      "Large, round eyes in shades of copper or gold exude a calm and gentle expression."
    ]
  },
  "completion_tokens": 87,
  "prompt_tokens": 217,
  "total_tokens": 304
}
```

## Contributing

All contributions are welcome.




