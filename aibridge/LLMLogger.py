from datetime import datetime
import os
import shutil

from aibridge.llm import LLM


class LLMLogger(LLM):
    """
    Wraps an existing LLM object and logs every single prompt/completion pair into a directory.
    """

    def __init__(self, llm: LLM, log_dir: str):
        """
        Initialize the LLMSimpleLogger with an existing LLM object and a directory to log prompts and completions.
        Deletes the directory first if it already exists.
        """

        # store parameters
        self.llm = llm
        self.log_dir = log_dir
        # delete directory if it already exists
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        # create directory
        os.makedirs(log_dir)


    def get_completion(self, prompt):
        # get completion from the wrapped LLM object
        completion = self.llm.get_completion(prompt)

        # generate timestamp as yyyy_mm_dd_hh_mm_ss_ffffff
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        # write prompt and completion to two separate files in the log directory
        with open(os.path.join(self.log_dir, timestamp + "_a_prompt.txt"), "w") as f:
            f.write(prompt)
        with open(os.path.join(self.log_dir, timestamp + "_b_completion.txt"), "w") as f:
            f.write(completion)

        # return completion
        return completion


    def get_token_counter(self):
        # just call the wrapped LLM object
        return self.llm.get_token_counter()

    def get_cost(self):
        # just call the wrapped LLM object
        return self.llm.get_cost()

