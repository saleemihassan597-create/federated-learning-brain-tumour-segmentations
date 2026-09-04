"""pytorchexample: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets_loaders import create_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Lambda
from torchvision import models


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
       # Load pretrained MobileNetV2
        self.model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT
        )

        # Freeze backbone (optional)
        for param in self.model.features.parameters():
            param.requires_grad = False

        # Custom classification head
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
      return self.model(x)


pytorch_transforms = Compose([
    Lambda(lambda img: img.convert("RGB")),
    Resize((224, 224)),
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch

brain_dataset = None

def get_dataset():
    global brain_dataset
    DATASET_PATH = "data/brain-tumor-multimodal-image"
    if brain_dataset is None:
        loader = create_dataset('brain_tumor', DATASET_PATH)
        brain_dataset = loader.load()

    return brain_dataset

def load_data(partition_id: int, num_partitions: int, batch_size: int):

    brain_dataset = get_dataset()
    partitioner = IidPartitioner(num_partitions=num_partitions)
    partitioner.dataset = brain_dataset["train"]
    partition = partitioner.load_partition(partition_id)

    partition = partition.train_test_split(test_size=0.2, seed=42,)
    partition = partition.with_transform(apply_transforms)
    trainloader = DataLoader( partition["train"], batch_size=batch_size, shuffle=True,)
    testloader = DataLoader(partition["test"], batch_size=batch_size, shuffle=False,)

    return trainloader, testloader

def load_centralized_dataset():
    """Load test set and return dataloader."""
    # Load entire test set
    dataset = get_dataset()
    test_dataset = dataset["test"]
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)

def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
