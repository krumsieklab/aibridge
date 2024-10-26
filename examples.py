"""
Example codes for the usage of the aibridge package.
"""

# All examples will initialize an object called "llm". (and overwrite each other here in this script)
# The easiest way to test it is to just call
# llm.get_completion("Who are you?")


##% Define API keys
openai_key = "sk-..."
anthropic_key = "sk-..."


#%% Quickstart
from aibridge.OpenAIClient import openai_models, OpenAIClient

# generate object
llm = OpenAIClient(api_key=openai_key, **openai_models["gpt-3.5-turbo"])
# run prompt
print(llm.get_completion("Who are you?"))
# print cost
llm.print_cost()


#%% OpenAI
from aibridge.OpenAIClient import openai_models, OpenAIClient
# Initialize OpenAI models using predefined model name and cost
from aibridge.OpenAIClient import openai_models, OpenAIClient
llm = OpenAIClient(api_key=openai_key, **openai_models["gpt-3.5-turbo"])
llm = OpenAIClient(api_key=openai_key, **openai_models["gpt-4-turbo"])
# Note: The ** mechanism is used to unpack the dictionary into keyword arguments

# Initialize OpenAI model by specifying all parameters
llm = OpenAIClient(api_key=openai_key, model_name="gpt-3.5-turbo-1106", cost_structure={"cost_per_1M_tokens_input":1.0, "cost_per_1M_tokens_output":2.0})

# There is also a "temperature" parameter that can be set, see documentation

print(llm.get_completion("Who are you?"))


#%% Anthropic

# Initialize Anthropic model using predefined model name and cost
from aibridge.AnthropicClient import anthropic_models, AnthropicClient
llm = AnthropicClient(api_key=anthropic_key, **anthropic_models["claude-3-haiku"])
llm = AnthropicClient(api_key=anthropic_key, **anthropic_models["claude-3-sonnet"])
# Note: The ** mechanism is used to unpack the dictionary into keyword arguments

# Initialize Anthropic model by specifying all parameters
llm = AnthropicClient(api_key=anthropic_key, model_name="claude-3-opus-20240229", cost_structure={"cost_per_1M_tokens_input":15.0, "cost_per_1M_tokens_output":75.0})

print(llm.get_completion("Who are you?"))


#%% Llama

# Initialize HTTP client to access an Ollama server
from aibridge.OllamaClient import OllamaClientHTTP
llm = OllamaClientHTTP(url="http://localhost:11434/api/generate", model_name="llama3:70b", verbose=True)
# Note: This particular example assumes that the docker runs locally or has been port-forwarded to localhost
print(llm.get_completion("Who are you?"))


#%% LM Studio

# Initialize LM Studio model (which internally uses the openai API)
# The model cannot be parameterized here, it'll connect to whatever server is running in LM Studio
from aibridge.LMStudioClient import LMStudioClient
llm = LMStudioClient(url="http://localhost:1234/v1")
print(llm.get_completion("Who are you?"))


#%% Google Client

# Setup is a bit more elaborate, since we have to communicate through the Google Cloud API
# It'll require prior authentication and setup of the Google Cloud SDK via 'gcloud auth ...'

# Create an instance of GoogleClient
from aibridge.GoogleClient import GoogleClient
llm = GoogleClient(
    api_endpoint="us-central1-aiplatform.googleapis.com",
    project_id="studious-pulsar-397018",
    location="us-central1",
    model_id="medlm-large",
    parameters_dict={
      "candidateCount": 1,
      "maxOutputTokens": 500,
      "temperature": 0.2,
      "topP": 0.8,
      "topK": 40
    }
)

print(llm.get_completion("Who are you?"))




#%% Logging wrapper

# This provides a mechanism which wraps an LLM object and logs every single prompt and completion
from aibridge.LLMLogger import LLMLogger
from aibridge.OpenAIClient import openai_models, OpenAIClient
# generate object
llm = LLMLogger(
    llm=OpenAIClient(api_key=openai_key, **openai_models["gpt-3.5-turbo"]),
    log_dir="logs"
)
# run two prompts
print(llm.get_completion("How many planets are there in the solar system?"))
print(llm.get_completion("What is the capital of France?"))
# Now check the logs directory for the files


#%% Simple chatbot
# Stores the entire conversation in a single string
from aibridge.OpenAIClient import openai_models, OpenAIClient
llm=OpenAIClient(api_key=openai_key, **openai_models["gpt-3.5-turbo"])
history = ""
# main loop
print("Chatbot ready. Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit': break
    history += f"You: {user_input}\n"
    response = llm.get_completion(history)
    history += f"Bot: {response}\n"
    print("Bot:", response)
# once the user typed "quit", print the entire conversation
print("Conversation:")
print(history)