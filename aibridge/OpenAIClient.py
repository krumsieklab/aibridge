import os
import time
from openai import OpenAI

from aibridge.llm import LLM
openai_models = {
    "gpt-3.5-turbo": {
        "model_name": "gpt-3.5-turbo",
        "cost_structure": {
            "cost_per_1M_tokens_input": 0.5,
            "cost_per_1M_tokens_output": 1.5
        }
    },
    "gpt-4-turbo": {
        "model_name": "gpt-4-turbo",
        "cost_structure": {
            "cost_per_1M_tokens_input": 10.00,
            "cost_per_1M_tokens_output": 30.00
        }
    },
    "gpt-4o": {
        "model_name": "gpt-4o",
        "cost_structure": {
            "cost_per_1M_tokens_input": 2.5,
            "cost_per_1M_tokens_output": 10.00
        }
    },
    "gpt-4o-mini": {
        "model_name": "gpt-4o-mini",
        "cost_structure": {
            "cost_per_1M_tokens_input": 0.15,
            "cost_per_1M_tokens_output": 0.60
        }
    },
    "o1": {
        "model_name": "o1",
        "cost_structure": {
            "cost_per_1M_tokens_input": 15,
            "cost_per_1M_tokens_output": 60
        }
    },
    "o1-mini": {
        "model_name": "o1-mini",
        "cost_structure": {
            "cost_per_1M_tokens_input": 3,
            "cost_per_1M_tokens_output": 12
        }
    },
    "o3-mini": {
        "model_name": "o1-mini",
        "cost_structure": {
            "cost_per_1M_tokens_input": 3,
            "cost_per_1M_tokens_output": 12
        }
    }
}

class OpenAIClient(LLM):

    def __init__(self, api_key: str, model_name: str, cost_structure: dict = None, openai_args: dict = None,
                 system_prompt: str = "You are a helpful AI assistant.", custom_url: str = None,
                 reasoning_effort: str = "medium"):
        """
        Initialize the OpenAIClient with the API key, model name, optional cost structure, and OpenAI API arguments.

        Args:
            api_key (str): OpenAI API key.
            model_name (str): The name of the OpenAI model to use.
            cost_structure (dict, optional): The cost structure of the model.
            openai_args (dict, optional): Additional arguments for the OpenAI API call.

        Example:
            openai_args = {
                "temperature": 0.7,
                "top_k": 40,
                "top_n": 1
            }
        """
        # Initialize superclass with cost structure
        super().__init__(cost_structure)
        # Store model name and OpenAI API arguments
        self.model_name = model_name
        self.openai_args = openai_args if openai_args else {}
        self.openai_args["model"] = model_name
        # Store system prompt
        self.system_prompt = system_prompt
        # Initialize OpenAI client
        os.environ["OPENAI_API_KEY"] = api_key
        # initialize OpenAI client
        if custom_url:
            self.client = OpenAI(base_url=custom_url)
        else:
            self.client = OpenAI()
        # if reasoning_effort was changed from default, and this is NOT a model starting with "o", throw an error
        if reasoning_effort != "medium" and not model_name.startswith("o"):
            raise ValueError("Reasoning effort can only be set for OpenAI models starting with 'o'.")
        # if this IS a model starting with "o", set reasoning effort
        if model_name.startswith("o"):
            self.openai_args["reasoning_effort"] = reasoning_effort

    def get_completion(self, prompt, max_retries=3):
        """
        Get a completion from the OpenAI API.

        Args:
            prompt (str): The prompt to send to the OpenAI API.
            max_retries (int, optional): Maximum number of retries in case of failure. Default is 3.

        Returns:
            str: The completion text from the OpenAI API.
        """


        # build messages dict
        # if this is one of the "o1..." models, then we cannot give a system role
        if self.model_name.startswith("o1"):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]

        # Retry loop
        for i in range(max_retries):
            try:
                # Run prompt with OpenAI API
                response = self.client.chat.completions.create(
                    messages=messages, **self.openai_args
                )
                # Update internal token counters
                self.update_token_counters(response.usage.prompt_tokens, response.usage.completion_tokens)
                # Extract and return message text
                text = response.choices[0].message.content
                return text

            except Exception as e:
                print("  OpenAI error: " + str(e))
                print("  Retrying...")
                time.sleep(1)

        # Raise exception after max retries
        raise Exception("Failed to get response from OpenAI after " + str(max_retries) + " retries")
