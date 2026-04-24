import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

from models.network import PrunableNN
from training.metrics import sparsity_loss, calculate_sparsity
from utils.logger import get_logger
from utils.config_loader import load_config

logger = get_logger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


def train_model(lambda_val):
    config = load_config()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    trainset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    trainloader = DataLoader(
        trainset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    model = PrunableNN().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"]
    )

    logger.info(f"Training started with lambda={lambda_val}")

    best_acc = 0.0

    for epoch in range(config["epochs"]):
        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            ce_loss = F.cross_entropy(outputs, labels)
            sp_loss = sparsity_loss(model)

            loss = ce_loss + lambda_val * (sp_loss / 10000)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy tracking
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        # Epoch metrics
        accuracy = 100 * correct / total
        sparsity = calculate_sparsity(model)

        logger.info(
            f"[Lambda={lambda_val}] Epoch {epoch+1}/{config['epochs']} | "
            f"Loss={total_loss:.2f} | Acc={accuracy:.2f}% | Sparsity={sparsity:.2f}%"
        )

        if accuracy > best_acc:
            best_acc = accuracy

            os.makedirs("outputs", exist_ok=True)

            model_path = f"outputs/model_lambda_{lambda_val}.pth"
            torch.save(model.state_dict(), model_path)

            logger.info(f"✅ Best model saved at epoch {epoch+1} (Acc={accuracy:.2f}%)")

    logger.info(f"Training completed for lambda={lambda_val}")
    logger.info(f"Best Accuracy: {best_acc:.2f}%")

    return model