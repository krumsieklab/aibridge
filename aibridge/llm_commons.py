import os
import re
import warnings

#from transformers import GPT2TokenizerFast
import tiktoken


def count_tokens(txt):
    """
    Count the number of tokens in the given text using GPT-4's tokenizer.

    Args:
        txt (str): The input text to tokenize.

    Returns:
        int: The number of tokens.
    """
    encoding = tiktoken.encoding_for_model('gpt-4o')
    return len(encoding.encode(txt))



def truncate_text_by_tokens(text, max_tokens):
    """
    Truncate the input text to a maximum number of tokens using GPT-4's tokenizer.

    Args:
        text (str): The input text to tokenize and truncate.
        max_tokens (int): The maximum number of tokens allowed.

    Returns:
        str: The truncated text if the token count exceeds max_tokens; otherwise, the original text.
    """
    # Initialize the tokenizer once and reuse it to avoid reinitialization overhead
    if not hasattr(truncate_text_by_tokens, "encoding"):
        truncate_text_by_tokens.encoding = tiktoken.encoding_for_model('gpt-4o')

    encoding = truncate_text_by_tokens.encoding

    # Encode the input text
    tokens = encoding.encode(text)

    # Truncate the tokens if their count exceeds max_tokens
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        # Decode the truncated tokens back to text
        truncated_text = encoding.decode(truncated_tokens)
        return truncated_text
    else:
        # Return the original text if token count is within the limit
        return text




def clean_gpt_json_output(str):
    '''
    Especially OpenAI's gpt4 sometimes adds ``` around the output.
    Remove all lines that start with ``` (plus the rest of those lines if there are additional characters)
    '''
    lines = str.split('\n')
    cleaned_lines = [line for line in lines if not line.startswith('```')]
    return '\n'.join(cleaned_lines)


def clean_json(json_string):
    """
    Cleans and standardizes the "text" fields within a JSON string representing a list of dictionaries.

    Functionality includes:

    1. **Fixes Invalid Escape Sequences**: Escapes any backslashes not followed by a valid escape character in the entire JSON string.

    2. **Replacement of Non-standard Characters**: Converts non-ASCII symbols, such as en dashes (–) to regular hyphens, and removes special characters like asterisks (*), daggers (†), double daggers (‡), and section symbols (§).

    3. **Retention of Valid Escape Sequences**: Preserves legitimate escape sequences (e.g., \n for newlines, \t for tabs) while escaping only those backslashes not followed by valid escape characters.

    4. **Removal of Invisible Unicode Characters**: Removes invisible or zero-width Unicode characters (e.g., \u200B).

    5. **Filtering and Escaping of Control Characters**: Strips out control characters outside the standard ASCII range and properly escapes control characters within the standard ASCII range (e.g., \x00-\x1F), ensuring the JSON string remains valid.

    The output is a JSON string with cleaned "text" fields.
    """

    # Define patterns and replacements
    invalid_escape_in_json = re.compile(r'\\(?![\"\\/bfnrtu])')
    text_field_pattern = re.compile(
        r'("text"\s*:\s*")'            # Match the "text" field name and the opening quote
        r'('
            r'(?:'
                r'[^\\"]'              # Non-backslash, non-quote characters
                r'|\\.'                # Escaped characters
            r')*?'
        r')'                           # Capture the text content
        r'"'                           # Closing quote of the text value
        , re.DOTALL                    # Add re.DOTALL to match multiline strings
    )
    replacements = {'–': '-', '*': '', '†': '', '‡': '', '§': ''}
    invisible_characters = re.compile(r'[\u200B-\u200F\u202A-\u202E\u2060]')
    non_ascii_control_characters = re.compile(r'[\x80-\x9F\xA0-\xFF]')
    backslash_not_escape_text = re.compile(r'\\(?![\\\"/bfnrtu])')

    # First, fix invalid escape sequences in the entire JSON string
    fixed_json_string = invalid_escape_in_json.sub(r'\\\\', json_string)

    # Function to escape control characters in a string
    def escape_control_characters(s):
        def replace(match):
            ch = match.group()
            return '\\u{:04x}'.format(ord(ch))
        return re.sub(r'[\x00-\x1F]', replace, s)

    # Function to process and clean the text content
    def process_text_content(match):
        prefix = match.group(1)  # The "text": " part
        content = match.group(2)  # The actual text content

        # Decode escape sequences in the content
        try:
            decoded_content = bytes(content, "utf-8").decode("unicode_escape")
        except UnicodeDecodeError:
            decoded_content = content  # Fallback if decoding fails

        # Apply replacements
        for char, replacement in replacements.items():
            decoded_content = decoded_content.replace(char, replacement)

        # Remove invisible Unicode characters
        decoded_content = invisible_characters.sub('', decoded_content)

        # Remove non-ASCII control characters
        decoded_content = non_ascii_control_characters.sub('', decoded_content)

        # Escape backslashes not followed by valid escape sequences
        decoded_content = backslash_not_escape_text.sub(r'\\\\', decoded_content)

        # Escape control characters (U+0000 through U+001F)
        decoded_content = escape_control_characters(decoded_content)

        # Re-escape backslashes and quotes for JSON
        decoded_content = decoded_content.replace('\\', '\\\\')
        decoded_content = decoded_content.replace('"', '\\"')

        # Return the reconstructed "text" field
        return f'{prefix}{decoded_content}"'

    # Apply the processing function to each "text" field
    cleaned_json_string = text_field_pattern.sub(process_text_content, fixed_json_string)

    return cleaned_json_string