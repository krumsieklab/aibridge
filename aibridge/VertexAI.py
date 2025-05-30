import time
import vertexai
import vertexai.preview.generative_models as generative_models
from aibridge.llm import LLM  # Adjust import path as needed


class VertexAIClient(LLM):
    """
    VertexAIClient class to interact with Google Cloud's Vertex AI Generative Models.
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
        retry_wait_time: int = 1
    ):
        """
        Initialize the VertexAIClient with the project ID, location, model name,
        optional cost structure, and optional Vertex AI arguments.

        Args:
            project (str): GCP project ID.
            location (str): GCP region (e.g., "us-central1").
            model_name (str): The name of the Vertex AI Generative Model (e.g., "gemini-1.0-pro-001").
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
                    "stream": True/False
                }
                Defaults to None.
            max_retries (int, optional): Maximum number of retries in case of API call failure.
                                         Defaults to 3.
            retry_wait_time (int, optional): Number of seconds to wait between retries.
                                             Defaults to 1.
        """
        super().__init__(cost_structure)

        # Initialize Vertex AI
        vertexai.init(project=project, location=location)

        # Save parameters
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        self.retry_wait_time = retry_wait_time

        # Merge or define default Vertex AI arguments
        self.vertexai_args = vertexai_args if vertexai_args else {}
        self.generation_config = self.vertexai_args.get("generation_config", {
            "max_output_tokens": 8000,
            "temperature": 0.9,
            "top_p": 0.95,
        })
        self.safety_settings = self.vertexai_args.get("safety_settings", {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH:
                generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
                generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT:
                generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT:
                generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        })
        # Whether we want streaming responses or not
        self.stream = self.vertexai_args.get("stream", False)

        # Create a Vertex AI Generative Model instance
        self.model = generative_models.GenerativeModel(self.model_name)

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
                if self.stream:
                    # In streaming mode, generate_content returns an iterator of GenerationResponse
                    responses = self.model.generate_content(
                        [final_prompt],
                        generation_config=self.generation_config,
                        safety_settings=self.safety_settings,
                        stream=True,
                    )
                    total_text = ""
                    for response in responses:
                        if hasattr(response, "text"):
                            total_text += response.text
                else:
                    # In non-streaming mode, generate_content returns a single GenerationResponse object
                    response = self.model.generate_content(
                        [final_prompt],
                        generation_config=self.generation_config,
                        safety_settings=self.safety_settings,
                        stream=False,
                    )
                    total_text = response.text if hasattr(response, "text") else ""

                # Update internal token counters (approximation)
                input_tokens = len(final_prompt.split())
                output_tokens = len(total_text.split())
                self.update_token_counters(input_tokens, output_tokens)

                return total_text.strip()

            except Exception as e:
                print(f"  Vertex AI error: {e}")
                print("  Retrying...")
                time.sleep(self.retry_wait_time)

        raise Exception(
            f"Failed to get response from Vertex AI after {self.max_retries} retries"
        )
