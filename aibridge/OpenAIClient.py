# OpenAI API calls
import os
import time
from openai import OpenAI

from aibridge.llm import LLM

# dictionary of models with model name as key and cost per token as value
openai_models = {
    "gpt-3.5-turbo" : {
        "model_name": "gpt-3.5-turbo-1106",
        "cost_per_1k_tokens_input": 0.0010,
        "cost_per_1k_tokens_output": 0.0020
    },
    "gpt-4-turbo" : {
        "model_name": "gpt-4-turbo-2024-04-09",
        "cost_per_1k_tokens_input": 0.01,
        "cost_per_1k_tokens_output": 0.03
    },
    "gpt-4o" : {
        "model_name": "gpt-4o-2024-05-13",
        "cost_per_1k_tokens_input": 0.005,
        "cost_per_1k_tokens_output": 0.015
    },
}


class OpenAIClient(LLM):

    def __init__(self, api_key: str, model_name: str, cost_per_1k_tokens_input: float, cost_per_1k_tokens_output, temperature: float = 0.8):
        # store parameters
        self.model_name = model_name
        self.cost_per_1k_tokens_input = cost_per_1k_tokens_input
        self.cost_per_1k_tokens_output = cost_per_1k_tokens_output
        self.temperature = temperature
        # initialize token counter
        self.token_counter_input=0
        self.token_counter_output=0

        # initialize openai
        os.environ["OPENAI_API_KEY"] = api_key
        self.client = OpenAI()

    # function that executes a prompt and returns the response
    # also keeps track of the total number of tokens used
    def get_completion(self, prompt, max_retries=3):

        # retry loop
        for i in range(max_retries):
            try:
                # run prompt
                response = self.client.chat.completions.create(
                  model=self.model_name,
                  messages=[
                    #{"role": "system", "content": "You are a helpful assistant. You must always read your chat history very carefully and follow the instructions exactly."},
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt}
                  ]
                )
                # increase internal token counter
                self.token_counter_input += response.usage.prompt_tokens
                self.token_counter_output += response.usage.completion_tokens
                # extract and return message text
                text = response.choices[0].message.content
                return text

            except Exception as e:
                print("  OpenAI error: " + str(e))
                print("  Retrying...")
                time.sleep(1)

        # if we get here, we've tried max_retries times and failed
        raise Exception("Failed to get response from OpenAI after " + str(max_retries) + " retries")

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
        # calculate cost in dollars
        cost = (self.token_counter_input * self.cost_per_1k_tokens_input/1000) + (self.token_counter_output * self.cost_per_1k_tokens_output/1000)
        return cost


