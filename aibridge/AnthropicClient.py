import time
import anthropic  # Assuming anthropic is a mock or actual package for API handling

from aibridge.llm import LLM

# Dictionary of models with model name as key and cost structure as value (per million tokens)
anthropic_models = {
    # ── Latest generation ──────────────────────────────────────────────
    "claude-opus-4-8": {
        "model_name": "claude-opus-4-8",
        "cost_structure": {
            "cost_per_1M_tokens_input": 5.0,
            "cost_per_1M_tokens_output": 25.0
        }
    },
    "claude-opus-4-7": {
        "model_name": "claude-opus-4-7",
        "cost_structure": {
            "cost_per_1M_tokens_input": 5.0,
            "cost_per_1M_tokens_output": 25.0
        }
    },
    "claude-opus-4-6": {
        "model_name": "claude-opus-4-6",
        "cost_structure": {
            "cost_per_1M_tokens_input": 5.0,
            "cost_per_1M_tokens_output": 25.0
        }
    },
    "claude-sonnet-4-6": {
        "model_name": "claude-sonnet-4-6",
        "cost_structure": {
            "cost_per_1M_tokens_input": 3.0,
            "cost_per_1M_tokens_output": 15.0
        }
    },
    "claude-haiku-4-5": {
        "model_name": "claude-haiku-4-5",
        "cost_structure": {
            "cost_per_1M_tokens_input": 1.0,
            "cost_per_1M_tokens_output": 5.0
        }
    },

    # ── Previous generation (still available) ──────────────────────────
    "claude-opus-4-5": {
        "model_name": "claude-opus-4-5",
        "cost_structure": {
            "cost_per_1M_tokens_input": 5.0,
            "cost_per_1M_tokens_output": 25.0
        }
    },
    "claude-sonnet-4-5": {
        "model_name": "claude-sonnet-4-5",
        "cost_structure": {
            "cost_per_1M_tokens_input": 3.0,
            "cost_per_1M_tokens_output": 15.0
        }
    },
    "claude-opus-4-1": {
        "model_name": "claude-opus-4-1",
        "cost_structure": {
            "cost_per_1M_tokens_input": 15.0,
            "cost_per_1M_tokens_output": 75.0
        }
    },
    "claude-sonnet-4-0": {
        "model_name": "claude-sonnet-4-0",
        "cost_structure": {
            "cost_per_1M_tokens_input": 3.0,
            "cost_per_1M_tokens_output": 15.0
        }
    },
    "claude-opus-4-0": {
        "model_name": "claude-opus-4-0",
        "cost_structure": {
            "cost_per_1M_tokens_input": 15.0,
            "cost_per_1M_tokens_output": 75.0
        }
    },
    "claude-3.5-haiku": {
        "model_name": "claude-3-5-haiku-latest",
        "cost_structure": {
            "cost_per_1M_tokens_input": 0.8,
            "cost_per_1M_tokens_output": 4.0
        }
    }
}

_ADAPTIVE_THINKING_MODELS = {"claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6"}
_EFFORT_MODELS = {"claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-4-5"}


class AnthropicClient(LLM):
    """
    AnthropicClient class to interact with the Claude API.
    """

    def __init__(self, api_key: str, model_name: str, cost_structure: dict = None, anthropic_args: dict = None,
                 system_prompt: str = "You are a helpful AI assistant.", thinking_budget: int = None,
                 effort: str = None, max_retries: int = 3, wait_time: int = 1):
        """
        Initialize the AnthropicClient with the API key, model name, optional cost structure, and Anthropic API arguments.

        Args:
            api_key (str): Anthropic API key.
            model_name (str): The name of the Anthropic model to use.
            cost_structure (dict, optional): The cost structure of the model.
            anthropic_args (dict, optional): Additional arguments for the Anthropic API call.
            system_prompt (str, optional): The system prompt to use. Defaults to "You are a helpful AI assistant."
            thinking_budget (int, optional): Token budget for extended thinking (deprecated for 4.6 models).
                Minimum 1024 tokens. For 4.6 models, use effort instead.
            effort (str, optional): Controls how many tokens Claude spends responding.
                One of "low", "medium", "high", "max". "max" is Opus 4.6 only.
                On 4.6 models this also enables adaptive thinking automatically.
                Supported on: Opus 4.6, Sonnet 4.6, Opus 4.5.
            max_retries (int, optional): Maximum number of retries in case of failure. Defaults to 3.
            wait_time (int, optional): Time to wait between retries in seconds. Defaults to 1.
        """
        super().__init__(cost_structure)
        self.model_name = model_name
        self.anthropic_args = anthropic_args if anthropic_args else {}
        self.anthropic_args["model"] = model_name
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        self.wait_time = wait_time
        if "max_tokens" not in self.anthropic_args:
            self.anthropic_args["max_tokens"] = 2048

        # ── Effort parameter ──────────────────────────────────────────────
        if effort is not None:
            valid_efforts = {"low", "medium", "high", "max"}
            if effort not in valid_efforts:
                raise ValueError(f"effort must be one of {valid_efforts}")
            if effort == "max" and model_name != "claude-opus-4-6":
                raise ValueError("effort='max' is only supported for claude-opus-4-6")
        self.effort = effort

        # ── Thinking configuration ────────────────────────────────────────
        if thinking_budget is not None and model_name in _ADAPTIVE_THINKING_MODELS:
            print(f"Warning: thinking_budget is deprecated for {model_name}. "
                  f"Use effort parameter instead. Ignoring thinking_budget.")
            self.thinking_budget = None
        elif thinking_budget is not None and model_name == "claude-3-5-haiku-latest":
            print(f"Warning: Model {model_name} does not support thinking. Ignoring thinking_budget.")
            self.thinking_budget = None
        elif thinking_budget is not None:
            if thinking_budget < 1024:
                raise ValueError("thinking_budget must be at least 1024 tokens")
            if thinking_budget >= self.anthropic_args["max_tokens"]:
                raise ValueError("thinking_budget must be less than max_tokens")
            self.thinking_budget = thinking_budget
        else:
            self.thinking_budget = None

        self.client = anthropic.Anthropic(api_key=api_key)
        self.call_timestamps = []

        # Anthropic-specific cache counters (separate from base class cached counter
        # because Anthropic has two tiers: creation at 1.25x and read at 0.1x)
        self.token_counter_cache_creation = 0
        self.token_counter_cache_read = 0

    def get_completion(self, prompt):
        """
        Get a completion from the Anthropic API.

        Args:
            prompt (str): The prompt to send to the Anthropic API.

        Returns:
            str: The completion text from the Anthropic API.
        """
        # Retry loop
        for i in range(self.max_retries):
            try:
                # Check rate limit before proceeding
                self._check_rate_limit()

                api_args = self.anthropic_args.copy()

                if self.effort is not None:
                    api_args["output_config"] = {"effort": self.effort}

                if self.model_name in _ADAPTIVE_THINKING_MODELS and self.effort is not None:
                    api_args["thinking"] = {"type": "adaptive"}
                elif self.thinking_budget is not None:
                    api_args["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": self.thinking_budget
                    }

                response = self.client.messages.create(
                    system=self.system_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    **api_args
                )

                # Extract cache and thinking token breakdowns
                usage = response.usage
                cache_creation = getattr(usage, 'cache_creation_input_tokens', 0) or 0
                cache_read = getattr(usage, 'cache_read_input_tokens', 0) or 0
                self.token_counter_cache_creation += cache_creation
                self.token_counter_cache_read += cache_read

                # Total input includes cache tokens (Anthropic reports them separately)
                total_input = usage.input_tokens + cache_creation + cache_read

                # Estimate thinking tokens from response content blocks.
                # Anthropic doesn't report thinking tokens separately in usage,
                # so we approximate by tokenizing the text response and subtracting
                # from the total output token count.
                thinking_tokens = 0
                response_text = None
                has_thinking = False
                for block in response.content:
                    if getattr(block, 'type', None) == 'thinking':
                        has_thinking = True
                    elif getattr(block, 'type', None) == 'text':
                        response_text = block.text
                if has_thinking and response_text is not None:
                    from aibridge.llm_commons import count_tokens
                    text_tokens = count_tokens(response_text)
                    thinking_tokens = max(0, usage.output_tokens - text_tokens)
                elif response_text is None:
                    response_text = response.content[-1].text if response.content else ""

                self.update_token_counters(
                    total_input, usage.output_tokens,
                    thinking_tokens=thinking_tokens,
                    cached_input_tokens=cache_read,
                )
                return response_text

            except anthropic.RateLimitError as e:
                print(f"Anthropic rate limit reached. Waiting a full minute to reset. (Full error: {e})")
                # Reset the call timestamps here
                self.call_timestamps.clear()
                # wait
                time.sleep(60)

            except Exception as e:
                print("  Anthropic error: " + str(e))
                print("  Retrying...")
                time.sleep(self.wait_time)

        # Raise exception after max retries
        raise Exception("Failed to get response from Anthropic after " + str(self.max_retries) + " retries")

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

    def get_token_counter(self):
        result = super().get_token_counter()
        if self.token_counter_cache_creation > 0:
            result["cache_creation"] = self.token_counter_cache_creation
        if self.token_counter_cache_read > 0:
            result["cache_read"] = self.token_counter_cache_read
        return result

    def get_cost(self):
        """Anthropic cache pricing: creation at 1.25x, read at 0.1x base input rate."""
        if self.cost_per_1M_tokens_input is None or self.cost_per_1M_tokens_output is None:
            return 0.0
        rate_in = self.cost_per_1M_tokens_input / 1_000_000
        rate_out = self.cost_per_1M_tokens_output / 1_000_000
        regular_input = self.token_counter_input - self.token_counter_input_cached \
                        - self.token_counter_cache_creation
        input_cost = (regular_input * rate_in) + \
                     (self.token_counter_cache_creation * rate_in * 1.25) + \
                     (self.token_counter_cache_read * rate_in * 0.10)
        output_cost = self.token_counter_output * rate_out
        return input_cost + output_cost

    def identify(self):
        extras = []
        if self.effort is not None:
            extras.append(f"effort={self.effort}")
        if self.thinking_budget is not None:
            extras.append(f"thinking_budget={self.thinking_budget}")
        if extras:
            return f"{self.__class__.__name__}({self.model_name} || {', '.join(extras)})"
        return f"{self.__class__.__name__}({self.model_name})"
