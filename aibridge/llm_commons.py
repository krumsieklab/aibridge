import os
import warnings

from transformers import GPT2TokenizerFast


def count_tokens(text):
    # "static" variable via function attribute
    if not hasattr(count_tokens, "tokenizer"):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        count_tokens.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return len(count_tokens.tokenizer(text)['input_ids'])


def truncate_text_by_tokens(text, max_tokens):
    # Initialize the tokenizer once and reuse it to avoid reinitialization overhead
    if not hasattr(truncate_text_by_tokens, "tokenizer"):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        truncate_text_by_tokens.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Tokenize the input text and suppress any warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tokens = truncate_text_by_tokens.tokenizer(text)['input_ids']

    # Truncate the tokens if their count exceeds max_tokens
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        # Decode the truncated tokens back to text
        truncated_text = truncate_text_by_tokens.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        return truncated_text
    else:
        # Return the original text if token count is within the limit
        return text


