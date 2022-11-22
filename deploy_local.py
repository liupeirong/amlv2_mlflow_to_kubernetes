import os
from dotenv import load_dotenv
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    KubernetesOnlineEndpoint,
    KubernetesOnlineDeployment,
    CodeConfiguration,
    Model,
    Environment)
from azure.ai.ml.exceptions import LocalEndpointNotFoundError
from azure.identity import DefaultAzureCredential

load_dotenv()
subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")
workspace = os.getenv("AZURE_ML_WORKSPACE")
inference_cluster = os.getenv("AZURE_ML_K8S_CLUSTER")
model_name = os.getenv("MODEL_NAME")
model_version = os.getenv("MODEL_VERSION")

ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)

# get or create the local endpoint
online_endpoint_name = "iris-endpoint-local"
try:
    endpoint = ml_client.online_endpoints.get(name=online_endpoint_name, local=True)
except LocalEndpointNotFoundError:
    print("Local endpoint not found. Creating a new one...")
    local_endpoint = KubernetesOnlineEndpoint(
        name=online_endpoint_name,
        compute=inference_cluster,
        auth_mode="key",
    )
    # for local deployment, async API is returned synchronously
    ml_client.online_endpoints.begin_create_or_update(local_endpoint, local=True)
    endpoint = ml_client.online_endpoints.get(name=online_endpoint_name, local=True)

# get or create the local deployment
deployment_name = "default"
try:
    deployment = ml_client.online_deployments.get(name=deployment_name, endpoint_name=online_endpoint_name, local=True)
except LocalEndpointNotFoundError:
    print("Local deployment not found. Creating a new one...")
    # prepare model and env for deployment
    #   model must be downloaded from Azure ML workspace portal to "model" folder on the local machine,
    #   the following won't work for local deployment:
    #       model = ml_client.models.get(name=model_name, version=model_version)
    model = Model(path="model", type="mlflow_model")
    env = Environment(
        conda_file="model/conda.yaml",
        image="mcr.microsoft.com/azureml/lightgbm-3.2-ubuntu18.04-py37-cpu-inference:20221107.v3",
    )
    local_deployment = KubernetesOnlineDeployment(
        name=deployment_name,
        endpoint_name=online_endpoint_name,
        model=model,
        environment=env,
        code_configuration=CodeConfiguration(
            code="scoring/", scoring_script="score.py"
        ),
    )
    ml_client.online_deployments.begin_create_or_update(local_deployment, local=True)

# validate inference result
print("Creating default deployment. Check the status in the Azure portal.")
result = ml_client.online_endpoints.invoke(
   endpoint_name=online_endpoint_name,
   request_file='./data/request-local.json',
   local=True)

print(result)
