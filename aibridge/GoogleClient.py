from abc import ABC, abstractmethod
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

from aibridge.llm import LLM


class GoogleClient(LLM):
    def __init__(self, api_endpoint: str, project_id: str, location: str, model_id: str, parameters_dict=None):
        self.client_options = {"api_endpoint": api_endpoint}
        self.client = aiplatform.gapic.PredictionServiceClient(client_options=self.client_options)
        self.endpoint = f"projects/{project_id}/locations/{location}/publishers/google/models/{model_id}"
        self.parameters_dict = parameters_dict or {
            "candidateCount": 1,
            "maxOutputTokens": 500,
            "temperature": 0.2,
            "topP": 0.8,
            "topK": 40
        }

    def get_completion(self, prompt):
        instance_dict = {"content": prompt}
        instance = json_format.ParseDict(instance_dict, Value())
        instances = [instance]
        parameters = json_format.ParseDict(self.parameters_dict, Value())
        response = self.client.predict(endpoint=self.endpoint, instances=instances, parameters=parameters)
        predictions = response.predictions
        if predictions:
            result = dict(predictions[0]).get('content', '').strip()
            return result
        return ""

    def get_token_counter(self):
        return {
            "input": -1,
            "output": -1,
            "total": --1,
        }

    def get_cost(self):
        # we don't know the cost of the Google model, so we just return -1
          return -1



# test code
if __name__ == "__main__":
  # Create an instance of GoogleClient
  google_client = GoogleClient(
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

  # Use the client to get a completion
  prompt = "Who are you?"
  response = google_client.get_completion(prompt)
  print("Response:", response)