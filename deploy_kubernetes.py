import os
from dotenv import load_dotenv
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    KubernetesOnlineEndpoint,
    KubernetesOnlineDeployment,
    CodeConfiguration,
    Environment)
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential

load_dotenv()
subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")
workspace = os.getenv("AZURE_ML_WORKSPACE")
inference_cluster = os.getenv("AZURE_ML_K8S_CLUSTER")
model_name = os.getenv("MODEL_NAME")
model_version = os.getenv("MODEL_VERSION")

ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)

# create or get the endpoint
online_endpoint_name = "irisk8s-endpoint"
try:
    endpoint = ml_client.online_endpoints.get(name=online_endpoint_name)
except ResourceNotFoundError:
    endpoint = KubernetesOnlineEndpoint(
        name=online_endpoint_name,
        compute=inference_cluster,
        auth_mode="key",
    )
    endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# get or create the deployment
deployment_name = "default"
try:
    deployment = ml_client.online_deployments.get(name=deployment_name, endpoint_name=online_endpoint_name)
except ResourceNotFoundError:
    print("Deployment not found. Creating a new one...")
    # prepare model and env for deployment
    model = ml_client.models.get(name=model_name, version=model_version)
    # you still need to download the model in order to get the conda.yaml to set up the environment
    env = Environment(
        conda_file="model/conda.yaml",
        image="mcr.microsoft.com/azureml/lightgbm-3.2-ubuntu18.04-py37-cpu-inference:20221107.v3",
    )
    k8s_deployment = KubernetesOnlineDeployment(
        name=deployment_name,
        endpoint_name=online_endpoint_name,
        model=model,
        environment=env,
        code_configuration=CodeConfiguration(
            code="scoring/", scoring_script="score.py"
        ),
    )
    deployment = ml_client.online_deployments.begin_create_or_update(k8s_deployment).result()

# validate inference result
percent = endpoint.traffic.get(deployment_name)
if percent == 0:
    endpoint.traffic = {deployment_name: 100} 
    ml_client.begin_create_or_update(endpoint).result()

print(f"k8s endpoint scoring uri: {endpoint.scoring_uri}")
# your dev machine probably doesn't have access to the k8s IP, so from a machine that can reach k8s cluster, run
# on linux:
# curl -d '{"data":[[1,2,3,4]]}' -H "Content-Type: application/json" -H "Authorization: Bearer <your key>" -X POST http://<your scoring_uri>
# on windows:
# curl -d "{\"data\":[[1,2,3,4]]}" -H "Content-Type: application/json" -H "Authorization: Bearer <your key>" -X POST http://<your scoring_uri>
