import time

import requests
from aibridge.llm import LLM

class OllamaClientHTTP(LLM):
    """
    Accesses an Ollama server via HTTP.
    Best practice is to use the dockerized version of the server.
    """

    def __init__(self, model_name: str, url: str = "http://localhost:11434/api/generate", verbose: bool = False, ollama_args: dict = None,
                 system_prompt: str = "", max_retries: int = 3, wait_time: int = 1):
        """
        Initialize the OllamaClientHTTP with the URL of the server, the model name, and other parameters.

        Args:
            url (str): Location of the Ollama server, e.g. http://localhost:11434/api/generate
            model_name (str): Name of the Ollama model, https://ollama.com/library
            verbose (bool, optional): If True, print additional information
            ollama_args (dict, optional): Additional arguments for the Ollama API call
            system_prompt (str, optional): System prompt to prepend to all requests
            max_retries (int, optional): Maximum number of retries in case of failure. Default is 3.
            wait_time (int, optional): Time to wait between retries in seconds. Default is 1.
        """
        # Initialize superclass with no cost structure
        super().__init__()
        # Store parameters
        self.url = url
        self.model_name = model_name
        self.verbose = verbose
        self.ollama_args = ollama_args if ollama_args else {}
        self.ollama_args["model"] = model_name
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        self.wait_time = wait_time

    @staticmethod
    def _remove_braille_characters(input_string):
        # Unicode range for Braille patterns: U+2800 to U+28FF
        return ''.join(char for char in input_string if not 0x2800 <= ord(char) <= 0x28FF)

    def get_completion(self, prompt: str) -> str:
        """
        Get a completion from the Ollama server.

        Args:
            prompt (str): The prompt to send to the Ollama server.

        Returns:
            str: The completion text from the Ollama server.
        """
        if self.verbose:
            print("Sending request to server...")

        # if there is a system prompt, add it to the beginning of the prompt, followed by a newline
        if self.system_prompt:
            prompt = self.system_prompt + "\n" + prompt

        # Retry loop
        for i in range(self.max_retries):
            try:
                # Send a POST request to the server
                response = requests.post(self.url, json={'prompt': prompt, 'model': self.model_name, 'stream': False, **self.ollama_args})

                # If the request was not successful, raise an exception
                if response.status_code != 200:
                    raise Exception(f"Error: {response.status_code} - {response.text}")

                # Parse JSON from response.text
                response_parsed = response.json()

                # Return the completion text
                return self._remove_braille_characters(response_parsed['response']).strip()

            except Exception as e:
                if self.verbose:
                    print(f"Error: {str(e)}")
                    print("Retrying...")
                time.sleep(self.wait_time)

        # Raise exception after max retries
        raise Exception(f"Failed to get response from Ollama server after {self.max_retries} retries")



# test code
if __name__ == "__main__":

    llm = OllamaClientHTTP(url="http://localhost:11434/api/generate", model_name="llama3:70b", verbose=True)
    print(llm.get_completion("How many planets are there in the solar system?"))
