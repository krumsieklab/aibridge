from abc import ABC, abstractmethod

# LLM interface class
class LLM(ABC):

    @abstractmethod
    def get_completion(self, prompt):
        """
        Get a completion from the LLM given a prompt.
        """
        pass

    @abstractmethod
    def get_token_counter(self):
        """
        Get the current token counter. Must be a dictionary with fields "input", "output", and "total".
        """
        pass

    @abstractmethod
    def get_cost(self):
        """
        Return cost in dollars.
        """
        pass

    # print token counter and cost, formatted
    def print_cost(self):
        """
        Print token counter and cost, formatted.
        """

        # print token counter and cost, formatted
        print("Tokens: {}".format(self.get_token_counter()))
        print("Cost: ${:,.5f}".format(self.get_cost()))

