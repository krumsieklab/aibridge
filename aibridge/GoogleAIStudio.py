import time
from google import genai
from google.genai import types
from aibridge.llm import LLM
from aibridge.llm_commons import count_tokens
from google.genai.types import ThinkingConfig
from google.genai.types import (
    GenerateContentConfig,
    HarmCategory,
    HarmBlockThreshold,
    SafetySetting,
)

class GoogleAIStudioClient(LLM):
    """
    GoogleAIStudioClient class to interact with Google AI Studio using the genai SDK.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        cost_structure: dict = None,
        system_prompt: str = "You are a helpful AI assistant.",
        generation_args: dict = None,
        max_retries: int = 3,
        retry_wait_time: int = 1
    ):
        """
        Initialize the GoogleAIStudioClient with the API key, model name,
        optional cost structure, and optional generation arguments.

        Args:
            api_key (str): Google AI Studio API key.
            model_name (str): The name of the model (e.g., "gemini-2.0-flash-exp", "gemini-3-pro-preview").
            cost_structure (dict, optional): The cost structure of the model. Must contain:
                {
                  "cost_per_1M_tokens_input": float,
                  "cost_per_1M_tokens_output": float
                }
                Defaults to None.
            system_prompt (str, optional): A system-level prompt to prepend to user prompts.
                                           Defaults to a simple helper message.
            generation_args (dict, optional): Additional arguments for generation.
                                            For example:
                {
                    "generation_config": {...},
                    "safety_settings": {...},
                    "thinking_config": {...}
                }
                Defaults to None.
            max_retries (int, optional): Maximum number of retries in case of API call failure.
                                         Defaults to 3.
            retry_wait_time (int, optional): Number of seconds to wait between retries.
                                             Defaults to 1.
        """
        super().__init__(cost_structure)

        # Initialize Google AI Studio client
        self.client = genai.Client(api_key=api_key)
        
        # Save parameters
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        self.retry_wait_time = retry_wait_time

        # Merge or define default generation arguments
        self.generation_args = generation_args if generation_args else {}
        self.generation_config = self.generation_args.get("generation_config",{
            "max_output_tokens": 8000,
            "temperature": 0.9,
            "top_p": 0.95,
        })
        
        # Extract thinking_config or use default
        thinking_config_params = self.generation_args.get("thinking_config", {})
        self.thinking_config = ThinkingConfig(
            thinking_budget=thinking_config_params.get("thinking_budget")
        )

        self.safety_settings = [
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            ),
        ]

    def get_completion(self, prompt: str) -> str:
        """
        Get a completion (generated text) from Google AI Studio.

        Args:
            prompt (str): The user prompt to send to Google AI Studio.

        Returns:
            str: The generated text from Google AI Studio.
        """
        final_prompt = f"{self.system_prompt}\n{prompt}" if self.system_prompt else prompt

        for attempt in range(self.max_retries):
            try:
                # Generate content using Google AI Studio
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=final_prompt,
                    config=GenerateContentConfig(
                        temperature=self.generation_config.get("temperature", 0.9),
                        top_p=self.generation_config.get("top_p", 0.95),
                        max_output_tokens=self.generation_config.get("max_output_tokens", 8000),
                        thinking_config=self.thinking_config
                    )
                )
                total_text = response.text if hasattr(response, "text") else ""

                # Update internal token counters from response metadata
                if hasattr(response, "usage_metadata"):
                    input_tokens = response.usage_metadata.prompt_token_count
                    output_tokens = response.usage_metadata.candidates_token_count
                else:
                    # Fallback to tiktoken approximation if metadata not available
                    input_tokens = count_tokens(final_prompt)
                    output_tokens = count_tokens(total_text)
                self.update_token_counters(input_tokens, output_tokens)

                return total_text.strip()

            except Exception as e:
                print(f"  Google AI Studio error: {e}")
                print("  Retrying...")
                time.sleep(self.retry_wait_time)

        raise Exception(
            f"Failed to get response from Google AI Studio after {self.max_retries} retries"
        )

