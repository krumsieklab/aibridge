import time
import anthropic  # Assuming anthropic is a mock or actual package for API handling

from aibridge.llm import LLM

# Dictionary of models with model name as key and cost structure as value (per million tokens)
anthropic_models = {
    "claude-3-opus": {
        "model_name": "claude-3-opus",
        "cost_structure": {
            "cost_per_1M_tokens_input": 15.0,
            "cost_per_1M_tokens_output": 75.0
        }
    },
    "claude-3.5-sonnet": {
        "model_name": "claude-3-5-sonnet-20241022",
        "cost_structure": {
            "cost_per_1M_tokens_input": 3.0,
            "cost_per_1M_tokens_output": 15
        }
    },
    "claude-3.5-haiku": {
        "model_name": "claude-3.5-haiku",
        "cost_structure": {
            "cost_per_1M_tokens_input": 0.8,
            "cost_per_1M_tokens_output": 4
        }
    }
}


class AnthropicClient(LLM):
    """
    AnthropicClient class to interact with the Claude API.
    """

    def __init__(self, api_key: str, model_name: str, cost_structure: dict = None, anthropic_args: dict = None,
                 system_prompt: str = "You are a helpful AI assistant."):
        """
        Initialize the AnthropicClient with the API key, model name, optional cost structure, and Anthropic API arguments.

        Args:
            api_key (str): Anthropic API key.
            model_name (str): The name of the Anthropic model to use.
            cost_structure (dict, optional): The cost structure of the model.
            anthropic_args (dict, optional): Additional arguments for the Anthropic API call.
        """
        # Initialize superclass with cost structure
        super().__init__(cost_structure)
        # Store model name and Anthropic API arguments
        self.model_name = model_name
        self.anthropic_args = anthropic_args if anthropic_args else {}
        self.anthropic_args["model"] = model_name
        # Store system prompt
        self.system_prompt = system_prompt
        # default settings for max_tokens, because it is required
        if "max_tokens" not in self.anthropic_args:
            self.anthropic_args["max_tokens"] = 1024
        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=api_key)
        self.call_timestamps = []  # Keep track of call timestamps for rate limiting

    def get_completion(self, prompt, max_retries=3):
        """
        Get a completion from the Anthropic API.

        Args:
            prompt (str): The prompt to send to the Anthropic API.
            max_retries (int, optional): Maximum number of retries in case of failure. Default is 3.

        Returns:
            str: The completion text from the Anthropic API.
        """
        # Retry loop
        for i in range(max_retries):
            try:
                # Check rate limit before proceeding
                self._check_rate_limit()

                # Call the API
                response = self.client.messages.create(
                    system=self.system_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    **self.anthropic_args
                )

                # Update internal token counters
                self.update_token_counters(response.usage.input_tokens, response.usage.output_tokens)
                # Extract and return message text
                return response.content.pop().text

            except anthropic.RateLimitError as e:
                print(f"Anthropic rate limit reached. Waiting a full minute to reset. (Full error: {e})")
                # Reset the call timestamps here
                self.call_timestamps.clear()
                # wait
                time.sleep(60)

            except Exception as e:
                print("  Anthropic error: " + str(e))
                print("  Retrying...")
                time.sleep(1)

        # Raise exception after max retries
        raise Exception("Failed to get response from Anthropic after " + str(max_retries) + " retries")

    def _check_rate_limit(self) -> None:
        """
        Check the rate limit and wait if necessary.
        """
        max_per_minute = self.anthropic_args.get('max_per_minute', 100000000)
        # If there are already max_per_minute calls, check if we should wait
        if len(self.call_timestamps) >= max_per_minute:
            first_call_timestamp = self.call_timestamps[0]
            current_time = time.time()
            time_since_first_call = current_time - first_call_timestamp

            # If the first call in the list was made less than a minute ago, wait
            if time_since_first_call < 60:
                time_to_wait = 60 - time_since_first_call
                print(f"Rate limit reached. Waiting for {time_to_wait:.2f} seconds.")
                time.sleep(time_to_wait)

            # Remove calls that are outside of the current minute window
            self.call_timestamps = [timestamp for timestamp in self.call_timestamps if current_time - timestamp < 60]

        # Record the timestamp of this call
        self.call_timestamps.append(time.time())
