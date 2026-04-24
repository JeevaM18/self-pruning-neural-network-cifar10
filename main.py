from training.train import train_model
from training.evaluate import evaluate
from utils.config_loader import load_config

def main():
    config = load_config()
    results = []

    for lambda_val in config["lambda_values"]:
        model = train_model(lambda_val)
        acc = evaluate(model)

        results.append((lambda_val, acc))

    print("\nFINAL RESULTS:")
    for r in results:
        print(f"Lambda={r[0]} → Accuracy={r[1]:.2f}%")

if __name__ == "__main__":
    main()