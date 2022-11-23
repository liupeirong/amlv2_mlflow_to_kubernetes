import os
from time import sleep
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import (
    Model,
    AmlCompute,
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
)

load_dotenv()
subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")
workspace = os.getenv("AZURE_ML_WORKSPACE")
training_cluster = os.getenv("AZURE_ML_TRAINING_CLUSTER")
registered_model_name = os.getenv("MODEL_NAME")
registered_model_version = os.getenv("MODEL_VERSION")

ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)

# get or create training cluster
try:
    ml_client.compute.get(training_cluster)
except Exception:
    print ("Training cluster is not found, creating a new one...")
    compute = AmlCompute(
        name=training_cluster, size="STANDARD_D2_V2", min_instances=0, max_instances=1
    )
    ml_client.compute.begin_create_or_update(compute).result()

# train the model if not already registered
try:
    registered_model = ml_client.models.get(registered_model_name, version=registered_model_version)
except Exception:
    print ("Model is not registered, training a new one...")
    command_job = command(
        code="./training",
        command="python train.py --iris-csv ${{inputs.iris_csv}} --learning-rate ${{inputs.learning_rate}} --boosting ${{inputs.boosting}}",
        environment="AzureML-lightgbm-3.2-ubuntu18.04-py37-cpu@latest",
        inputs={
            "iris_csv": Input(
                type="uri_file",
                path="https://azuremlexamples.blob.core.windows.net/datasets/iris.csv",
            ),
            "learning_rate": 0.9,
            "boosting": "gbdt",
        },
        compute=training_cluster,
    )
    returned_job = ml_client.jobs.create_or_update(command_job)
    while returned_job.status != "Completed":
        returned_job = ml_client.jobs.get(returned_job.name)
        sleep(10)
        print(returned_job.status)

    # register the model
    run_model = Model(
        path="azureml://jobs/{}/outputs/artifacts/paths/model/".format(returned_job.name),
        name=registered_model_name,
        description="Model created from run.",
        type="mlflow_model",
    )
    registered_model = ml_client.models.create_or_update(run_model)

# create an online endpoint for validation
online_endpoint_name = "iris-managed-endpoint"
try:
    endpoint = ml_client.online_endpoints.get(name=online_endpoint_name)
except Exception:
    print ("Managed endpoint is not found, creating a new one...")
    managed_endpoint = ManagedOnlineEndpoint(
        name=online_endpoint_name,
        auth_mode="key",
    )
    endpoint = ml_client.online_endpoints.begin_create_or_update(managed_endpoint).result()

# create an online deployment.
deployment_name = "blue"
try:
    deployment = ml_client.online_deployments.get(name=deployment_name, endpoint_name=online_endpoint_name)
except Exception:
    print("Deployment is not found, creating a new one...")
    managed_deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=online_endpoint_name,
        model=registered_model,
        instance_type="Standard_DS3_v2",
        instance_count=1,
    )
    deployment = ml_client.online_deployments.begin_create_or_update(managed_deployment).result()

# test the blue deployment with some sample data
print("Validate inferencing results on managed endpoint...")
result = ml_client.online_endpoints.invoke(
   endpoint_name=online_endpoint_name,
   deployment_name=deployment_name,
   request_file='./data/request.json')

print(result)