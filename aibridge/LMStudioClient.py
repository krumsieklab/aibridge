import os
import time
from openai import OpenAI
from aibridge.llm import LLM

class LMStudioClient(LLM):
    """
    Accesses an LM Studio server via the OpenAI API.
    """

    def __init__(self, base_url: str, system_prompt: str = "You are a helpful AI assistant.", openai_args: dict = None, verbose: bool = False):
        """
        Initialize the LMStudioClient with the base URL, model name, system prompt, and other parameters.

        Args:
            base_url (str): Base URL of the LM Studio server.
            system_prompt (str): System prompt to use for the assistant.
            openai_args (dict, optional): Additional arguments for the OpenAI API call.
            verbose (bool, optional): If True, print additional information.
        """
        # Initialize superclass
        super().__init__()
        # Store parameters
        self.api_key = "lm-studio"
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.openai_args = openai_args if openai_args else {}
        self.openai_args["model"] = "irrelevant_string"
        # Initialize OpenAI client
        os.environ["OPENAI_API_KEY"] = self.api_key
        self.client = OpenAI(base_url=base_url)

    def get_completion(self, prompt: str, max_retries: int = 3) -> str:
        """
        Get a completion from the LM Studio server.

        Args:
            prompt (str): The prompt to send to the LM Studio server.
            max_retries (int, optional): Maximum number of retries in case of failure. Default is 3.

        Returns:
            str: The completion text from the LM Studio server.
        """
        if self.verbose:
            print("Sending request to server...")

        # Retry loop
        for i in range(max_retries):
            try:
                # Send request to the server
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    **self.openai_args
                )

                # Update internal token counters
                self.update_token_counters(response.usage.prompt_tokens, response.usage.completion_tokens)

                # Extract and return message text
                text = response.choices[0].message.content
                return text

            except Exception as e:
                if self.verbose:
                    print(f"Error: {str(e)}")
                    print("Retrying...")
                time.sleep(1)

        # Raise exception after max retries
        raise Exception(f"Failed to get response from LM Studio server after {max_retries} retries")

# test code
if __name__ == "__main__":
    llm = LMStudioClient(
        base_url="http://localhost:1234/v1",
        system_prompt="You are a helpful AI assistant that will take the user's instructions very seriously and output nothing else except what was asked for.",
        verbose=True
    )
    print(llm.get_completion("Cite the entire declaration of independence. "))
