import json
import jsonschema

from aibridge.llm_commons import clean_json_output


def complete_and_validate(llm, prompt_basepath, variable_dict):

    '''
    Complete the prompt with the variables in variable_dict and validate the output against
    a .schema file. prompt_basebath will be appended with .txt (for the prompt template) and .schema (for the JSON schema)
    '''

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
    # clean JSON
    response = clean_json_output(response)
    # convert to python object
    response = json.loads(response)


    # validate response using jsonschema.validate
    try:
        jsonschema.validate(response, schema)
    except jsonschema.exceptions.ValidationError as err:
        raise ValueError(f"LLM JSON output does not adhere to the template_A3_followup.schema: {err.message}")

    return response

