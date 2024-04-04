# Client implementation to server-hosted, dockerized llama service
import requests
from aibridge import llm_commons
from aibridge.llm import LLM


class OllamaClientHTTP(LLM):
    """
    Accesses an Ollama server via HTTP.
    Best practice is use the dockerized version of the server.
    """

    def __init__(self, url, model, verbose=False):
        """
        Initialize the OllamaClientHTTP with the URL of the server, the model name, and other parameters.
        :param url:      Location of the Ollama server, e.g. http://localhost:11434/api/generate
        :param model:    Name of the ollama model, https://ollama.com/library
        :param verbose:  If True, print additional information
        """
        # store all parameters
        self.url = url
        self.model = model
        self.verbose = verbose


    # helper method
    @staticmethod
    def _remove_braille_characters(input_string):
        # Unicode range for Braille patterns: U+2800 to U+28FF
        return ''.join(char for char in input_string if not 0x2800 <= ord(char) <= 0x28FF)


    def get_completion(self, prompt: str) -> str:
        if self.verbose:
            print("Sending request to server...")

        # send a POST request to the server
        response = requests.post(self.url, json={'prompt': prompt, 'model': self.model, 'stream': False})

        # if the request was not successful, raise an exception
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")

        # parse JSON from response.text
        response_parsed = response.json()

        # return the completion text
        return self._remove_braille_characters(response_parsed['response']).strip()


    def get_token_counter(self):
        return 0

    def get_cost(self):
        return 0






