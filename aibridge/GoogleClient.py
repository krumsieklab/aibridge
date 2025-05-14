import time

from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

from aibridge.llm import LLM

class GoogleClient(LLM):
    def __init__(self, api_endpoint: str, project_id: str, location: str, model_id: str, parameters_dict: dict = None, max_retries: int = 3, wait_time: int = 1):
        """
        Initialize the GoogleClient with the necessary parameters to access the Google AI Platform.

        Args:
            api_endpoint (str): The endpoint of the API.
            project_id (str): The Google Cloud project ID.
            location (str): The location of the model.
            model_id (str): The ID of the model.
            parameters_dict (dict, optional): The parameters for the prediction request.
            max_retries (int, optional): Maximum number of retries in case of failure. Default is 3.
            wait_time (int, optional): Time to wait between retries in seconds. Default is 1.
        """
        # Initialize superclass with no cost structure
        super().__init__()
        # Store parameters
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
        self.max_retries = max_retries
        self.wait_time = wait_time

    def get_completion(self, prompt: str) -> str:
        """
        Get a completion from the Google AI Platform.

        Args:
            prompt (str): The prompt to send to the Google AI Platform.

        Returns:
            str: The completion text from the Google AI Platform.
        """
        instance_dict = {"content": prompt}
        instance = json_format.ParseDict(instance_dict, Value())
        instances = [instance]
        parameters = json_format.ParseDict(self.parameters_dict, Value())

        for i in range(self.max_retries):
            try:
                response = self.client.predict(endpoint=self.endpoint, instances=instances, parameters=parameters)
                predictions = response.predictions
                if predictions:
                    result = dict(predictions[0]).get('content', '').strip()
                    return result
                return ""
            except Exception as e:
                print(f"Google AI Platform error: {str(e)}")
                print("Retrying...")
                time.sleep(self.wait_time)

        raise Exception(f"Failed to get response from Google AI Platform after {self.max_retries} retries")



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