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

    @abstractmethod
    def get_completion(self, prompt):
        """
        Get a completion from the LLM given a prompt.
        """
        pass

    def update_token_counters(self, input_tokens: int, output_tokens: int):
        # Update internal token counters
        self.token_counter_input += input_tokens
        self.token_counter_output += output_tokens

    def get_token_counter(self):
        # Generate dictionary with token counts
        return {
            "input": self.token_counter_input,
            "output": self.token_counter_output,
            "total": self.token_counter_input + self.token_counter_output
        }

    def get_cost(self):
        # Calculate cost in dollars
        if self.cost_per_1M_tokens_input is not None and self.cost_per_1M_tokens_output is not None:
            cost = (self.token_counter_input * self.cost_per_1M_tokens_input / 1000000) + \
                   (self.token_counter_output * self.cost_per_1M_tokens_output / 1000000)
            return cost
        else:
            return 0.0

    def get_cost_str(self):
        # Return formatted cost string
        cost = self.get_cost()
        return "${:,.5f}".format(cost)

    def print_cost(self):
        """
        Print token counter and cost, formatted.
        """
        print("Tokens: {}".format(self.get_token_counter()))
        print("Cost: {}".format(self.get_cost_str()))

    def identify(self):
        """
        By default, just return the class name.
        """
        return self.__class__.__name__


