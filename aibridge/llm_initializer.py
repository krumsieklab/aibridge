from aibridge.AnthropicClient import antropic_models, AnthropicClient
from aibridge.OllamaClient import OllamaClientHTTP
from aibridge.OpenAIClient import openai_models, OpenAIClient


# LLM factory with hardcoded class choice
def init_llm(model_name, api_key=None):

    # if model name starts with "gpt-", use OpenAIClient
    if model_name.startswith("gpt-"):
        # API key must be provided
        if api_key is None:
            raise Exception("API key must be provided for OpenAI models")
        # make sure the model name is in the dictionary
        if model_name not in openai_models:
            raise Exception(f"Model {model_name} not found in openai_models dictionary")
        return OpenAIClient(api_key=api_key, **openai_models[model_name])

    # if model name starts with "llama", use LlamaClientHTTP
    elif model_name.startswith("llama") or model_name.startswith("mistral"):
        return OllamaClientHTTP("http://localhost:11434/api/generate", model=model_name, verbose=True)

    # if model name starts with "claude", use ClaudeClient
    elif model_name.startswith("claude"):
        # API key must be provided
        if api_key is None:
            raise Exception("API key must be provided for Claude models")
        # make sure the model name is in the dictionary
        if model_name not in antropic_models:
            raise Exception(f"Model {model_name} not found in claude_models dictionary")
        return AnthropicClient(api_key=api_key, **antropic_models[model_name], max_per_minute=5, verbose=True)

    # if model name is not recognized, raise an exception
    else:
        raise Exception(f"Model {model_name} not recognized")
