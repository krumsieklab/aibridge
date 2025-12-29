import time
import warnings
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

class GoogleVertexGenAIClient(LLM):
    """
    GoogleVertexGenAIClient class to interact with Google Cloud's Vertex AI using the genai SDK.
    """

    def __init__(
        self,
        project: str,
        location: str,
        model_name: str,
        cost_structure: dict = None,
        system_prompt: str = "You are a helpful AI assistant.",
        vertexai_args: dict = None,
        max_retries: int = 3,
        retry_wait_time: int = 1,
        suppress_warnings: bool = False
    ):
        """
        Initialize the GoogleVertexGenAIClient with the project ID, location, model name,
        optional cost structure, and optional Vertex AI arguments.

        Args:
            project (str): GCP project ID.
            location (str): GCP region (e.g., "us-central1").
            model_name (str): The name of the Vertex AI Generative Model (e.g., "gemini-1.5-pro").
            cost_structure (dict, optional): The cost structure of the model. Must contain:
                {
                  "cost_per_1M_tokens_input": float,
                  "cost_per_1M_tokens_output": float
                }
                Defaults to None.
            system_prompt (str, optional): A system-level prompt to prepend to user prompts.
                                           Defaults to a simple helper message.
            vertexai_args (dict, optional): Additional arguments for Vertex AI generation.
                                            For example:
                {
                    "generation_config": {...},
                    "safety_settings": {...},
                    "stream": True/False,
                    "thinking_config": {...}
                }
                Defaults to None.
            max_retries (int, optional): Maximum number of retries in case of API call failure.
                                         Defaults to 3.
            retry_wait_time (int, optional): Number of seconds to wait between retries.
                                             Defaults to 1.
            suppress_warnings (bool, optional): If True, suppress Google Cloud SDK user credential warnings. Defaults to False.
        """
        super().__init__(cost_structure)

        self.suppress_warnings = suppress_warnings
        # Suppress credential warnings if requested
        if self.suppress_warnings:
            warnings.filterwarnings(
                "ignore",
                message="Your application has authenticated using end user credentials from Google Cloud SDK without a quota project.*",
                category=UserWarning,
                module=r"google\.auth\._default",
            )

        # Initialize Vertex AI
        self.client = genai.Client(
            vertexai=True, project=project, location=location,
        )
        # Save parameters
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        self.retry_wait_time = retry_wait_time

        # Merge or define default Vertex AI arguments
        self.vertexai_args = vertexai_args if vertexai_args else {}
        self.generation_config = self.vertexai_args.get("generation_config",{
            "max_output_tokens": 8000,
            "temperature": 0.9,
            "top_p": 0.95,
        })
        
        # Extract thinking_config or use default
        thinking_config_params = self.vertexai_args.get("thinking_config", {})
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
        Get a completion (generated text) from the Vertex AI Generative Model.

        Args:
            prompt (str): The user prompt to send to Vertex AI.

        Returns:
            str: The generated text from Vertex AI.
        """
        final_prompt = f"{self.system_prompt}\n{prompt}" if self.system_prompt else prompt

        for attempt in range(self.max_retries):
            try:
                # In non-streaming mode, generate_content returns a single GenerationResponse object
                responses = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[final_prompt],
                    config=GenerateContentConfig(
                        temperature=self.generation_config.get("temperature", 0.9),
                        top_p=self.generation_config.get("top_p", 0.95),
                        max_output_tokens=self.generation_config.get("max_output_tokens", 8000),
                        thinking_config=self.thinking_config
                    )
                )
                total_text = responses.text if hasattr(responses, "text") else ""

                # Update internal token counters from response metadata
                if hasattr(responses, "usage_metadata"):
                    input_tokens = responses.usage_metadata.prompt_token_count
                    output_tokens = responses.usage_metadata.candidates_token_count
                else:
                    # Fallback to tiktoken approximation if metadata not available
                    input_tokens = count_tokens(final_prompt)
                    output_tokens = count_tokens(total_text)
                self.update_token_counters(input_tokens, output_tokens)

                return total_text.strip()

            except Exception as e:
                print(f"  Vertex AI error: {e}")
                print("  Retrying...")
                time.sleep(self.retry_wait_time)

        raise Exception(
            f"Failed to get response from Vertex AI after {self.max_retries} retries"
        )

