"""
Interface to the Claude API. This class is responsible for handling the API calls to Claude, rate limiting, and
keeping track of usage statistics.

Contains a dictionary of models with model name as key and cost per token as value called "claude_models".
"""

import time
import anthropic  # Assuming anthropic is a mock or actual package for API handling

from aibridge.llm import LLM

# dictionary of models with model name as key and cost per token as value
antropic_models = {
    "claude-3-opus" : {
        "model_name": "claude-3-opus-20240229",
        "cost_per_1k_tokens_input": 0.015,
        "cost_per_1k_tokens_output": 0.075
    },
    "claude-3-sonnet" : {
        "model_name": "claude-3-sonnet-20240229",
        "cost_per_1k_tokens_input": 0.003,
        "cost_per_1k_tokens_output": 0.0015
    },
    "claude-3-haiku" : {
        "model_name": "claude-3-haiku-20240307",
        "cost_per_1k_tokens_input": 0.00025,
        "cost_per_1k_tokens_output": 0.00125
    }
}


class AnthropicClient(LLM):
    """
    ClaudeClient class to interact with the Claude API.
    """

    def __init__(self, api_key: str, model_name: str, cost_per_1k_tokens_input: float, cost_per_1k_tokens_output: float,
                 verbose=False, max_per_minute: int = 100000000, temperature: float = 0.8):
        """
        Initialize the ClaudeClient with the API key, model name, cost per 1k tokens for input and output, and other
        parameters.
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name
        self.cost_per_1k_tokens_input = cost_per_1k_tokens_input
        self.cost_per_1k_tokens_output = cost_per_1k_tokens_output
        self.max_per_minute = max_per_minute
        self.temperature = temperature
        self.token_counter_input = 0
        self.token_counter_output = 0
        self.verbose = verbose
        self.call_timestamps = []  # Keep track of call timestamps within the current minute


    def get_completion(self, prompt: str) -> str:
        """
        Get a completion from the Claude API given a prompt.
        """

        # Check rate limit before proceeding
        self._check_rate_limit()

        repeat = True
        while repeat:
            # Call the API
            try:
                message = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=1024,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                repeat = False

            except anthropic.RateLimitError as e:
                if self.verbose:
                    print(f"Claude rate limit reached. Waiting a full minute to reset.")
                # Reset the call timestamps here
                self.call_timestamps.clear()
                # wait
                time.sleep(60)

            except Exception:
                # Re-raise the caught exception without handling it
                raise


        # Update token counters with hypothetical or actual usage
        self.token_counter_input += message.usage.input_tokens
        self.token_counter_output += message.usage.output_tokens

        return message.content.pop().text

    def _check_rate_limit(self) -> None:
        """
        Check the rate limit and wait if necessary.
        """
        # If there are already max_per_minute calls, check if we should wait
        if len(self.call_timestamps) >= self.max_per_minute:
            first_call_timestamp = self.call_timestamps[0]
            current_time = time.time()
            time_since_first_call = current_time - first_call_timestamp

            # If the first call in the list was made less than a minute ago, wait
            if time_since_first_call < 60:
                time_to_wait = 60 - time_since_first_call
                if self.verbose:
                    print(f"Rate limit reached. Waiting for {time_to_wait:.2f} seconds.")
                time.sleep(time_to_wait)

            # Remove calls that are outside of the current minute window
            self.call_timestamps = [timestamp for timestamp in self.call_timestamps if current_time - timestamp < 60]

        # Record the timestamp of this call
        self.call_timestamps.append(time.time())


    # getter function for token counter
    def get_token_counter(self):
        # generate dictionary
        return {
            "input": self.token_counter_input,
            "output": self.token_counter_output,
            "total": self.token_counter_input + self.token_counter_output
        }

    # getter function for cost
    def get_cost(self):
        # calculate cost
        cost = (self.token_counter_input * self.cost_per_1k_tokens_input / 1000) + (
                    self.token_counter_output * self.cost_per_1k_tokens_output / 1000)
        return cost


