from abc import ABC, abstractmethod

# LLM interface class
class LLM(ABC):

    def __init__(self, cost_structure: dict = None):
        # Validate and initialize cost structure
        if cost_structure:
            if not isinstance(cost_structure, dict) or \
               "cost_per_1M_tokens_input" not in cost_structure or \
               "cost_per_1M_tokens_output" not in cost_structure:
                raise ValueError("Cost structure must be a dictionary with 'cost_per_1M_tokens_input' and 'cost_per_1M_tokens_output' keys.")
            self.cost_per_1M_tokens_input = cost_structure["cost_per_1M_tokens_input"]
            self.cost_per_1M_tokens_output = cost_structure["cost_per_1M_tokens_output"]
        else:
            self.cost_per_1M_tokens_input = None
            self.cost_per_1M_tokens_output = None

        # Initialize token counters
        self.token_counter_input = 0
        self.token_counter_output = 0
        self.token_counter_thinking = 0
        self.token_counter_input_cached = 0

        # Subclasses can override to apply a discount for cached input tokens.
        # OpenAI uses 0.5 (50% discount). Anthropic overrides get_cost() directly
        # because it has two cache tiers (creation at 1.25x, read at 0.1x).
        self._cached_input_cost_multiplier = 1.0

    @abstractmethod
    def get_completion(self, prompt):
        """
        Get a completion from the LLM given a prompt.
        """
        pass

    def get_completion_structured(self, prompt, structure):
        """
        Must be implemented by subclasses, throws NotImplementedError if not implemented.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def update_token_counters(self, input_tokens: int, output_tokens: int,
                              thinking_tokens: int = 0, cached_input_tokens: int = 0):
        self.token_counter_input += input_tokens
        self.token_counter_output += output_tokens
        self.token_counter_thinking += thinking_tokens
        self.token_counter_input_cached += cached_input_tokens

    def get_token_counter(self):
        result = {"input": self.token_counter_input}
        if self.token_counter_input_cached > 0:
            result["input_cached"] = self.token_counter_input_cached
        if self.token_counter_thinking > 0:
            result["output_thinking"] = self.token_counter_thinking
            result["output_response"] = self.token_counter_output - self.token_counter_thinking
            result["output_total"] = self.token_counter_output
        else:
            result["output"] = self.token_counter_output
        result["total"] = self.token_counter_input + self.token_counter_output
        return result

    def get_cost(self):
        if self.cost_per_1M_tokens_input is None or self.cost_per_1M_tokens_output is None:
            return 0.0
        non_cached = self.token_counter_input - self.token_counter_input_cached
        input_cost = (non_cached * self.cost_per_1M_tokens_input / 1_000_000) + \
                     (self.token_counter_input_cached * self.cost_per_1M_tokens_input
                      * self._cached_input_cost_multiplier / 1_000_000)
        output_cost = self.token_counter_output * self.cost_per_1M_tokens_output / 1_000_000
        return input_cost + output_cost

    def get_cost_str(self):
        cost = self.get_cost()
        return "${:,.5f}".format(cost)

    def print_cost(self):
        """
        Print token counter and cost, formatted.
        """
        counters = self.get_token_counter()
        parts = [f"input={counters['input']}"]
        if "input_cached" in counters:
            parts.append(f"input_cached={counters['input_cached']}")
        if "output_thinking" in counters:
            parts.append(f"output_thinking={counters['output_thinking']}")
            parts.append(f"output_response={counters['output_response']}")
            parts.append(f"output_total={counters['output_total']}")
        else:
            parts.append(f"output={counters['output']}")
        parts.append(f"total={counters['total']}")
        print(f"Tokens: {', '.join(parts)}")
        print(f"Cost: {self.get_cost_str()}")

    def identify(self):
        """
        By default, just return the class name.
        Add the model name in parentheses, is the class has a 'model_name' attribute.
        """
        if hasattr(self, "model_name"):
            return f"{self.__class__.__name__}({self.model_name})"
        else:
            return self.__class__.__name__



