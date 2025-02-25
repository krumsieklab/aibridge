from datetime import datetime
import os
import shutil

from aibridge.llm import LLM


class LLMLogger(LLM):
    """
    Wraps an existing LLM object and logs every single prompt/completion pair into a directory.
    """

    def __init__(self, llm: LLM, log_dir: str, file_prefix : str = "", delete_existing_dir: bool = False):
        """
        Initialize the LLMSimpleLogger with an existing LLM object and a directory to log prompts and completions.
        By default, the directory is not deleted if it already exists.

        :param llm:                 The LLM object to wrap
        :param log_dir:             The directory to log prompts and completions
        :param file_prefix:         Optional suffix to append to the file names
        :param delete_existing_dir: If True, delete the directory if it already exists
        """

        # store parameters
        self.llm = llm
        self.log_dir = log_dir
        self.file_prefix = file_prefix
        # delete directory if it already exists
        if delete_existing_dir and os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        # create directory, if it does not exist
        os.makedirs(log_dir, exist_ok=True)


    def get_completion(self, prompt):
        # get completion from the wrapped LLM object
        completion = self.llm.get_completion(prompt)

        # generate timestamp as yyyy_mm_dd_hh_mm_ss_ffffff
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        # write prompt and completion to two separate files in the log directory
        with open(os.path.join(self.log_dir, f"{self.file_prefix}{timestamp}_a_prompt.txt"), "w") as f:
            f.write(prompt)
        with open(os.path.join(self.log_dir, f"{self.file_prefix}{timestamp}_b_completion.txt"), "w") as f:
            f.write(completion)

        # return completion
        return completion

    # function to change the prefix without needing to re-create the object
    def set_prefix(self, prefix: str):
        self.file_prefix = prefix

    # delegate attribute access to the wrapped LLM object
    def __getattr__(self, attr):
        return getattr(self.llm, attr)

    # identify should say Logger(identify() of inner LLM)
    def identify(self):
        # logger needs to be invisible, call identify on the wrapped LLM object
        return self.llm.identify()

