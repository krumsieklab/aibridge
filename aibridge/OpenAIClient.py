import os
import time
from openai import OpenAI

from aibridge.llm import LLM
openai_models = {
    # --- legacy ---
    "gpt-3.5-turbo": {
        "model_name": "gpt-3.5-turbo",
        "cost_structure": {
            "cost_per_1M_tokens_input": 0.50,
            "cost_per_1M_tokens_output": 1.50
        }
    },

    # --- GPT-4.1 family ---
    "gpt-4.1": {
        "model_name": "gpt-4.1",
        "cost_structure": {
            "cost_per_1M_tokens_input": 2.00,
            "cost_per_1M_tokens_output": 8.00
        }
    },
    "gpt-4.1-mini": {
        "model_name": "gpt-4.1-mini",
        "cost_structure": {
            "cost_per_1M_tokens_input": 0.40,
            "cost_per_1M_tokens_output": 1.60
        }
    },
    "gpt-4.1-nano": {
        "model_name": "gpt-4.1-nano",
        "cost_structure": {
            "cost_per_1M_tokens_input": 0.10,
            "cost_per_1M_tokens_output": 0.40
        }
    },

    # --- GPT-4.5 preview ---
    "gpt-4.5-preview": {
        "model_name": "gpt-4.5-preview",
        "cost_structure": {
            "cost_per_1M_tokens_input": 75.00,
            "cost_per_1M_tokens_output": 150.00
        }
    },

    # --- GPT-4o family ---
    "gpt-4o": {
        "model_name": "gpt-4o",
        "cost_structure": {
            "cost_per_1M_tokens_input": 5.00,
            "cost_per_1M_tokens_output": 20.00
        }
    },
    "gpt-4o-mini": {
        "model_name": "gpt-4o-mini",
        "cost_structure": {
            "cost_per_1M_tokens_input": 0.60,
            "cost_per_1M_tokens_output": 2.40
        }
    },

    # --- O-series ---
    "o1": {
        "model_name": "o1",
        "cost_structure": {
            "cost_per_1M_tokens_input": 15.00,
            "cost_per_1M_tokens_output": 60.00
        }
    },
    "o1-pro": {
        "model_name": "o1-pro",
        "cost_structure": {
            "cost_per_1M_tokens_input": 150.00,
            "cost_per_1M_tokens_output": 600.00
        }
    },
    "o1-mini": {
        "model_name": "o1-mini",
        "cost_structure": {
            "cost_per_1M_tokens_input": 1.10,
            "cost_per_1M_tokens_output": 4.40
        }
    },

    # --- O-3 family ---
    "o3": {
        "model_name": "o3",
        "cost_structure": {
            "cost_per_1M_tokens_input": 2.00,
            "cost_per_1M_tokens_output": 8.00
        }
    },
    "o3-pro": {
        "model_name": "o3-pro",
        "cost_structure": {
            "cost_per_1M_tokens_input": 20.00,
            "cost_per_1M_tokens_output": 80.00
        }
    },
    "o3-mini": {
        "model_name": "o3-mini",
        "cost_structure": {
            "cost_per_1M_tokens_input": 1.10,
            "cost_per_1M_tokens_output": 4.40
        }
    },

    "o4-mini": {
        "model_name": "o4-mini",
        "cost_structure": {
            "cost_per_1M_tokens_input": 1.10,
            "cost_per_1M_tokens_output": 4.40
        }
    },

    # --- GPT-5 family (verified pricing) ---
    "gpt-5": {
        "model_name": "gpt-5",
        "cost_structure": {
            "cost_per_1M_tokens_input": 1.25,
            "cost_per_1M_tokens_output": 10.00
        }
    },
    "gpt-5-mini": {
        "model_name": "gpt-5-mini",
        "cost_structure": {
            "cost_per_1M_tokens_input": 0.25,
            "cost_per_1M_tokens_output": 2.00
        }
    },
    "gpt-5-nano": {
        "model_name": "gpt-5-nano",
        "cost_structure": {
            "cost_per_1M_tokens_input": 0.05,
            "cost_per_1M_tokens_output": 0.40
        }
    },
    "gpt-5-chat-latest": {
        "model_name": "gpt-5-chat-latest",
        "cost_structure": {
            "cost_per_1M_tokens_input": 1.25,
            "cost_per_1M_tokens_output": 10.00
        }  # matches base gpt-5 pricing
    }
}


class OpenAIClient(LLM):

    def __init__(self, api_key: str, model_name: str, cost_structure: dict = None, openai_args: dict = None,
                 system_prompt: str = "You are a helpful AI assistant.", custom_url: str = None,
                 reasoning_effort: str = None, max_retries: int = 3, wait_time: int = 1):
        """
        Initialize the OpenAIClient with the API key, model name, optional cost structure, and OpenAI API arguments.

        Args:
            api_key (str): OpenAI API key.
            model_name (str): The name of the OpenAI model to use.
            cost_structure (dict, optional): The cost structure of the model.
            openai_args (dict, optional): Additional arguments for the OpenAI API call.
            system_prompt (str, optional): The system prompt to use. Defaults to "You are a helpful AI assistant."
            custom_url (str, optional): Custom URL for the OpenAI API.
            reasoning_effort (str, optional): Reasoning effort level for O-series models.
            max_retries (int, optional): Maximum number of retries for API calls. Defaults to 3.
            wait_time (int, optional): Time to wait between retries in seconds. Defaults to 1.

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
        # Store retry settings
        self.max_retries = max_retries
        self.wait_time = wait_time
        # Initialize OpenAI client
        os.environ["OPENAI_API_KEY"] = api_key
        # initialize OpenAI client
        if custom_url:
            self.client = OpenAI(base_url=custom_url)
        else:
            self.client = OpenAI()

        # if reasoning effort is given, this must be a model starting with "o"
        if reasoning_effort and not (model_name.startswith("o") or model_name.startswith("gpt-5")):
            raise ValueError("Reasoning effort is only supported for models starting with 'o'")
        # but specifically, it cannot be given for "o1-mini"
        if reasoning_effort and model_name == "o1-mini":
            raise ValueError("Reasoning effort is not supported for 'o1-mini'")
        # if reasoning effort is given, it must be "low", "medium", or "high"
        if reasoning_effort and reasoning_effort not in ["low", "medium", "high"]:
            raise ValueError("Reasoning effort must be 'low', 'medium', or 'high'")
        # if it is an "o" model EXCEPT "o1-mini", a reasoning effort MUST be given
        if model_name.startswith("o") and model_name != "o1-mini" and not reasoning_effort:
            raise ValueError("Reasoning effort must be given for models starting with 'o', except 'o1-mini'")
        
        # add the reasoning effort to the openai_args if it is given
        if reasoning_effort:
            self.openai_args["reasoning_effort"] = reasoning_effort

    def get_completion(self, prompt):
        """
        Get a completion from the OpenAI API.

        Args:
            prompt (str): The prompt to send to the OpenAI API.

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
        for i in range(self.max_retries):
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
                time.sleep(self.wait_time)

        # Raise exception after max retries
        raise Exception("Failed to get response from OpenAI after " + str(self.max_retries) + " retries")

    def identify(self):
        if "reasoning_effort" in self.openai_args:
            return f"{self.__class__.__name__}({self.model_name} || reasoning={self.openai_args['reasoning_effort']})"
        else:
            return f"{self.__class__.__name__}({self.model_name})"