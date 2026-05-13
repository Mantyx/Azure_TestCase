"""Launch Azure ML training job."""

import argparse
import sys
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError

PROD_COMPUTE = "gpu-cluster"
PROD_VM_SIZE = "Standard_NC24ads_A100_v4"

TEST_COMPUTE = "gpu-cluster-test"
TEST_VM_SIZE = "Standard_NC4as_T4_v3"


def get_or_create_compute(ml_client: MLClient, compute_name: str, size: str):
    try:
        compute = ml_client.compute.get(compute_name)
        print(f"Found existing compute target: {compute_name}")
    except ResourceNotFoundError:
        print(f"Creating new compute target: {compute_name} (Size: {size})")
        compute = AmlCompute(
            name=compute_name,
            size=size,
            min_instances=0,
            max_instances=2,
            idle_time_before_scale_down=120,
            tier="Dedicated",
        )
        ml_client.compute.begin_create_or_update(compute).result()
        print(f"Compute target {compute_name} created successfully.")
    return compute

def main():
    parser = argparse.ArgumentParser(description="Launch Azure ML training job.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Name of the Gold dataset to train on")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    
    # Azure ML specific arguments
    parser.add_argument("--subscription-id", type=str, default="390fc122-22d8-4646-9468-8325c6c4ae79", help="Azure Subscription ID")
    parser.add_argument("--resource-group", type=str, default="ml-resourcegroup_1", help="Azure Resource Group")
    parser.add_argument("--workspace-name", type=str, default="my-workspace", help="Azure ML Workspace name")
    parser.add_argument("--test", action="store_true", help="Run infrastructure test without data")
    
    args = parser.parse_args()

    # Validate arguments
    if not args.test and not args.dataset_name:
        raise ValueError("--dataset-name is required when not running in --test mode")

    print("Authenticating with Azure...")
    credential = DefaultAzureCredential()
    
    ml_client = MLClient(
        credential=credential,
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name,
    )

    print(f"Connected to workspace: {ml_client.workspace_name}")

    if args.test:
        compute_name = TEST_COMPUTE
        vm_size = TEST_VM_SIZE
    else:
        compute_name = PROD_COMPUTE
        vm_size = PROD_VM_SIZE

    get_or_create_compute(ml_client, compute_name, vm_size)

    print("Configuring training job...")

    if args.test:
        print(f"Running in TEST MODE on {TEST_VM_SIZE} (T4 GPU, fits within quota)")
        job = command(
            code="./src",
            command="python train.py --test-mode",
            environment="AzureML-ACPT-pytorch-1.13-py38-cuda11.7-gpu@latest",
            compute=compute_name,
            display_name="infrastructure-test",
            experiment_name="infrastructure-tests",
        )
    else:
        # In production, you would register your Gold dataset as an Azure ML Data Asset
        # and pass it as an input. For now, we pass the dataset name as a string argument.
        job = command(
            code="./src",  # Local path to the source code directory
            command=(
                "python train.py "
                "--dataset-name ${{inputs.dataset_name}} "
                "--epochs ${{inputs.epochs}} "
                "--batch-size ${{inputs.batch_size}} "
                "--lr ${{inputs.lr}}"
            ),
            inputs={
                "dataset_name": args.dataset_name,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
            },
            environment="AzureML-ACPT-pytorch-1.13-py38-cuda11.7-gpu@latest",
            compute=compute_name,
            display_name=f"train-{args.dataset_name}",
            experiment_name="surgical-ai-training",
        )

    # Submit the job
    print("Submitting job to Azure ML...")
    returned_job = ml_client.jobs.create_or_update(job)
    
    print(f"Job submitted successfully!")
    print(f"Job Name: {returned_job.name}")
    print(f"Track job progress at: {returned_job.studio_url}")

if __name__ == "__main__":
    main()
