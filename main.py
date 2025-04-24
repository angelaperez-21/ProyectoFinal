from src import Training

def main():
    print("[Proceso] Entrenamiento del modelo en curso...")
    Training.train_model()
    Training.classify_image()

if __name__ == "__main__":
    main()

