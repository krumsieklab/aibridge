import json
import jsonschema
import os
import re
import aibridge.llm_commons as llm_commons
from aibridge.OpenAIClient import OpenAIClient, openai_models
from aibridge.llm import LLM

promptstr_pre_example = "Here is an example output in JSON format:"
promptstr_json_only = "Output nothing but JSON, no extra whitespaces, no extra characters, no extra lines, no commentary:"

def complete_and_validate(llm, prompt_basepath, variable_dict):

    '''
    Complete the prompt with the variables in variable_dict and validate the output against
    a .schema file. prompt_basebath will be appended with .txt (for the prompt template) and .schema (for the JSON schema)
    '''

    # throw meaningful errors if either the prompt .txt file or the schema .schema file is missing
    if not os.path.exists(prompt_basepath + ".txt"):
        raise ValueError(f"Prompt file {prompt_basepath}.txt not found")
    if not os.path.exists(prompt_basepath + ".schema"):
        raise ValueError(f"Schema file {prompt_basepath}.schema not found")

    # load the prompt template
    with open(prompt_basepath + ".txt", "r") as f:
        prompt_template = f.read()
    # load the validation .schema
    with open(prompt_basepath + ".schema", "r") as f:
        schema = json.load(f)

    # build prompt
    try:
        prompt = prompt_template.format(**variable_dict)
    except KeyError as e:
        raise ValueError(f"KeyError: {e}\nThis either means a mismatch between the provided variable dictionary and the placeholders in the template, or it means that a curly bracket was not escaped properly in the template ({{ vs. {{{{).")
    # get response
    response = llm.get_completion(prompt)
    # clean JSON (GPT formatting, and invalid escape characters)
    response = llm_commons.clean_gpt_json_output(response)
    response = llm_commons.clean_json(response)
    # convert to python object
    response = json.loads(response)


    # validate response using jsonschema.validate
    try:
        jsonschema.validate(response, schema)
    except jsonschema.exceptions.ValidationError as err:
        raise ValueError(f"LLM JSON output does not adhere to the schema: {err.message}")

    return response


def generate_json_schema(example_json_str):
    def infer_schema(value):
        if isinstance(value, dict):
            return {
                "type": "object",
                "properties": {k: infer_schema(v) for k, v in value.items()},
                "required": list(value.keys())
            }
        elif isinstance(value, list):
            if len(value) > 0 and value[-1] == "...":
                if len(value) > 1:
                    return {
                        "type": "array",
                        "items": infer_schema(value[0])
                    }
                else:
                    return {
                        "type": "array",
                        "items": {"type": "any"}
                    }
            else:
                # For any array without "...", treat it as if it had "..."
                if len(value) > 0:
                    return {
                        "type": "array",
                        "items": infer_schema(value[0])
                    }
                else:
                    return {
                        "type": "array",
                        "items": {"type": "any"}
                    }
        elif isinstance(value, bool):
            return {"type": "boolean"}
        elif isinstance(value, str):
            return {"type": "string"}
        elif isinstance(value, int):
            return {"type": "integer"}
        elif isinstance(value, float):
            return {"type": "number"}
        elif value is None:
            return {"type": "null"}
        else:
            return {"type": "any"}

    example_data = json.loads(example_json_str)
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        **infer_schema(example_data)
    }

def complete_and_validate_autoschema(
    llm, prompt_template, variable_dict
):
    '''
    Expects the prompt to end with a <example></example> block, which will then be used to generate a schema.
    Furthermore, this function will automatically add prompt text around the example to form a complete prompt.
    '''

    # throw an error if llm is not of type LLM
    if not isinstance(llm, LLM):
        raise ValueError("llm must be an instance of LLM")

    # trim
    prompt_template = prompt_template.strip()
    # validate that it ends with <example>.*</example>

    if not re.search(r'<example>.*</example>$', prompt_template, re.DOTALL):
        raise ValueError("Prompt template must end with a <example>...</example> block")
    # extract inside of <example>...</example>, and remove entire block from prompt_template
    example = re.search(r'<example>(.*)</example>$', prompt_template, re.DOTALL).group(1).strip()
    prompt_template = re.sub(r'<example>.*</example>$', '', prompt_template, flags=re.DOTALL).strip()
    # convert example to schema
    schema = generate_json_schema(example)

    # build prompt
    try:
        prompt = prompt_template.format(**variable_dict)
    except KeyError as e:
        raise ValueError(
            f"KeyError: {e}\nThis either means a mismatch between the provided variable dictionary and the placeholders in the template, or it means that a curly bracket was not escaped properly in the template ({{ vs. {{{{).")

    # add JSON example with surrounding prompt text additions
    prompt = f"{prompt}\n\n{promptstr_pre_example}\n{example}\n{promptstr_json_only}"

    # get response
    response = llm.get_completion(prompt)
    # clean JSON (GPT formatting, and invalid escape characters)
    response = llm_commons.clean_gpt_json_output(response)
    response = llm_commons.clean_json(response)
    # convert to python object
    response = json.loads(response)

    # validate response using jsonschema.validate
    try:
        jsonschema.validate(response, schema)
    except jsonschema.exceptions.ValidationError as err:
        raise ValueError(f"LLM JSON output does not adhere to the schema: {err.message}")

    return response

