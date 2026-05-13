# Azure ML Setup

> We use **Azure ML**, not Azure AI Foundry — we train custom PyTorch models from scratch on surgical data, not configure pre-trained foundation models.

## Before you start ML projects

Azure Machine Learning resources are created inside a subscription in a fixed order:

1. **Resource group** — Create a resource group first. It is the container that holds your Azure resources and is required before you can provision anything else for this workload.
2. **ML workspace** — Create an Azure Machine Learning workspace in that resource group. The workspace is the top-level object for experiments, compute, data connections, and the model registry.
3. **ML projects** — Only after the workspace exists can you run training jobs, register models, and use the other features documented in this repo.

You cannot create or use a workspace without a resource group; you cannot meaningfully start an ML project without a workspace.

## Prerequisites

- Azure subscription
- A **resource group** and an **Azure ML workspace** in that group (see [Before you start ML projects](#before-you-start-ml-projects))
- Azure CLI installed and configured
- [uv](https://github.com/astral-sh/uv) for Python package management

## Configuration

Configure Azure CLI:
```bash
az login
az account set --subscription <subscription-id>
```

## Workspace

After you have a resource group, create or select the Azure ML workspace that this project uses. Azure ML workspace configuration and concepts.
