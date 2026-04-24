from fastapi import FastAPI, UploadFile, File
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

from models.network import PrunableNN
from utils.logger import get_logger

app = FastAPI(title="Self-Pruning Neural Network API")

logger = get_logger()

classes = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

device = torch.device("cpu")

try:
    model = PrunableNN().to(device)
    model.load_state_dict(
        torch.load("outputs/model_lambda_0.01.pth", map_location=device)
    )
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])


@app.get("/")
def home():
    return {"message": "Self-Pruning Neural Network API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        logger.info("Received prediction request")

        image = Image.open(file.file).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            probs = F.softmax(output, dim=1)
            confidence, pred = torch.max(probs, 1)

        label = classes[pred.item()]
        conf = confidence.item() * 100

        logger.info(f"Prediction: {label}, Confidence: {conf:.2f}%")

        return {
            "prediction": label,
            "confidence": round(conf, 2)
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}