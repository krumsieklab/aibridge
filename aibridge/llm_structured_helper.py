import json
import jsonschema
import os
import re
import aibridge.llm_commons as llm_commons
from aibridge.OpenAIClient import OpenAIClient, openai_models
from aibridge.llm import LLM

promptstr_pre_example = "Here is an example output in JSON format:"
promptstr_instructions = "Invalid JSON characters must be escaped properly, including newlines and quotes within strings. Output nothing but JSON, no extra whitespaces, no extra characters, no extra lines, no commentary:"

promptstr_safe_pre_example = "Here is an example output:"
promptstr_safe_instructions = "Your output must follow EXACTLY that XML like block from above. Output nothing else:"

def complete_and_validate(llm, prompt_basepath, variable_dict, fix_json=False):

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
    # fix json?
    if fix_json:
        response = llm_commons.fix_llm_json(response)
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

    try:
        example_data = json.loads(example_json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"The example JSON string was not valid: {e}")

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        **infer_schema(example_data)
    }

def complete_and_validate_autoschema(
    llm, prompt_template, variable_dict, fix_json=False
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
    prompt = f"{prompt}\n\n{promptstr_pre_example}\n{example}\n{promptstr_instructions}"

    # get response
    response = llm.get_completion(prompt)
    # clean JSON (GPT formatting, and invalid escape characters)
    response = llm_commons.clean_gpt_json_output(response)
    response = llm_commons.clean_json(response)
    # fix json?
    if fix_json:
        response = llm_commons.fix_llm_json(response)
    # convert to python object
    response = json.loads(response)

    # validate response using jsonschema.validate
    try:
        jsonschema.validate(response, schema)
    except jsonschema.exceptions.ValidationError as err:
        raise ValueError(f"LLM JSON output does not adhere to the schema: {err.message}")

    return response


def ensure_flat_string_schema(schema):
    """
    Validates that 'schema' is a dictionary with top-level 'type'='object',
    contains a 'properties' dictionary, and each property is { "type": "string" }
    with no nesting. Everything else is ignored.

    Raises:
        ValueError if any violation is found.
    """
    # Must be a dictionary
    if not isinstance(schema, dict):
        raise ValueError("Schema must be a dictionary (top-level).")

    # Must have "type": "object"
    top_level_type = schema.get("type")
    if top_level_type != "object":
        raise ValueError(
            f"Top-level 'type' must be 'object'; found '{top_level_type}'."
        )

    # Must have a 'properties' key that is a dictionary
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("Schema must have a 'properties' key containing a dictionary.")

    # Check each property in 'properties'
    for prop_name, prop_def in properties.items():
        if not isinstance(prop_def, dict):
            raise ValueError(f"Property '{prop_name}' must be a dictionary.")

        # Only "type" : "string" is allowed
        prop_type = prop_def.get("type")
        if prop_type != "string":
            raise ValueError(
                f"Property '{prop_name}' must have 'type'='string'; found '{prop_type}'."
            )

        # Disallow nesting under each property (ignore everything else if it's not "properties")
        if "properties" in prop_def:
            raise ValueError(
                f"Nesting is not allowed in property '{prop_name}' (found nested 'properties')."
            )

    return True


def complete_and_validate_autoschema_textsafe(
    llm, prompt_template, variable_dict, fix_json=False
):
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
    # validate that the schema is flat and only contains strings
    ensure_flat_string_schema(schema)

    # since we ensure its structure, it's parsable, generate structure out of it
    struct = json.loads(example)

    # build a string where each key of the dictionary is an <key>...</key> block with the content inside
    xml_example = ""
    for key, value in struct.items():
        # if value is a list, join it with commas
        if isinstance(value, list):
            value = ", ".join(value)
        # add to xml_example
        xml_example += f"<{key}>{value}</{key}>\n"
    # wrap in <output>...</output>
    xml_example = f"<output>\n{xml_example}\n</output>"

    # build prompt
    try:
        prompt = prompt_template.format(**variable_dict)
    except KeyError as e:
        raise ValueError(
            f"KeyError: {e}\nThis either means a mismatch between the provided variable dictionary and the placeholders in the template, or it means that a curly bracket was not escaped properly in the template ({{ vs. {{{{).")

    # add JSON example with surrounding prompt text additions
    prompt = f"{prompt}\n\n{promptstr_safe_pre_example}\n{xml_example}\n{promptstr_safe_instructions}"

    # get response
    response = llm.get_completion(prompt)

    # clean JSON (GPT formatting, and invalid escape characters)
    response = llm_commons.clean_gpt_json_output(response)
    response = response.strip()
    # if this still starts with <example> AND ends with </example>, remove those
    if response.startswith("<output>") and response.endswith("</output>"):
        response = response[8:-9].strip()

    # go through the keys again and reg-exp the response to get the values
    results = {}
    for key, value in struct.items():
        # if value is a list, join it with commas
        if isinstance(value, list):
            value = ", ".join(value)
        match = re.search(f"<{key}>(.*?)</{key}>", response, re.DOTALL)
        if match:
            results[key] = match.group(1).strip()
        else:
            results[key] = ""

    # return
    return results






