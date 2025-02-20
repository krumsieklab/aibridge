import json

from aibridge.llm import LLM
from aibridge import llm_structured_helper


class ExampleMirrorLLM(LLM):

    def get_completion(self, prompt):
        # this is a very specific hack made to work with the complete_and_validate_autoschema function of aibridge
        # it will find that example that is given, and return that exact example

        # we're accessing the inner workings of aibridge, this is not sustainable
        prefix = llm_structured_helper.promptstr_pre_example
        suffix = llm_structured_helper.promptstr_json_only

        # find what's exactly in between the prefix and suffix
        start = prompt.find(prefix)
        end = prompt.find(suffix)
        # if not found: this LLM is not compatible with the current prompt
        if start == -1 or end == -1:
            raise ValueError(f"The {self.__class__.__name__} LLM only works with the complete_and_validate_autoschema function.")
        # extract the example
        example = prompt[start + len(prefix):end].strip()

        # parse from JSON
        result = json.loads(example)
        # if it's a list and any of the entries are the verbatim string "...", remove those entries
        if isinstance(result, list):
            result = [x for x in result if x != "..."]

        # convert back to JSON string and return
        return json.dumps(result)

