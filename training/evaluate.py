import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utils.logger import get_logger

logger = get_logger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    testset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    testloader = DataLoader(testset, batch_size=128, shuffle=False)

    correct = 0
    total = 0

    model.eval()

    logger.info("Starting evaluation...")

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    logger.info(f"Test Accuracy: {accuracy:.2f}%")

    return accuracy