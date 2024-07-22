'''
Dummy LLM class that just returns a fixed string for testing purposes.
'''

from aibridge.llm import LLM  # Assuming LLM is in aibridge.llm based on your code structure

class DummyLLM(LLM):
    def __init__(self, fixed_response: str, cost_structure: dict = None):
        """
        Initializes DummyLLM with a fixed response and an optional cost structure.
        Args:
            fixed_response (str): The fixed string to return for every completion.
            cost_structure (dict, optional): Model's cost structure.
        """
        super().__init__(cost_structure)
        self.fixed_response = fixed_response

    def get_completion(self, prompt):
        """
        Returns a fixed response regardless of the prompt.
        Args:
            prompt (str): Ignored in this dummy implementation.
        Returns:
            str: The fixed response.
        """
        return self.fixed_response
