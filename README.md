# Train pytorch model on CIFAR10 with Azure Machine Learning
Based on https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-1st-experiment-sdk-train

### Clone this repo
e.g. from a Juypter Notebook with !git clone https://github.com/rohoffgit/CIFAR10-AzureMachineLearning

### How to
1. Create compute target (AML compute cluster, 1 node GPU, idle time 360 seconds)
2. Use notebooks/setup.ipynb to configure workspace
3. Use notebooks/train.ipynb to train pytorch model on CIFAR10

### For more information see
https://docs.microsoft.com/en-us/azure/machine-learning/