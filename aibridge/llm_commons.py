import os
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
        truncate_text_by_tokens.encoding = tiktoken.encoding_for_model('gpt-4')

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




def clean_json_output(str):
    '''
    Especially OpenAI's gpt4 sometimes adds ``` around the output.
    Remove all lines that start with ``` (plus the rest of those lines if there are additional characters)
    '''
    lines = str.split('\n')
    cleaned_lines = [line for line in lines if not line.startswith('```')]
    return '\n'.join(cleaned_lines)
