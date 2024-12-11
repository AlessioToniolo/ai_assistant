from app import create_embedding
import numpy as np


def generate_sample_data():
    robotics_samples = [
        "Pick and place robot arm moving components from conveyor belt to assembly station",
        "Automated welding robot performing precision welds on automotive chassis",
        "Quality control robot using computer vision to inspect manufactured parts",
        "Collaborative robot assisting human workers in packaging finished products",
        "Mobile robot navigating factory floor to transport materials between workstations",
    ]

    embeddings = [create_embedding(text) for text in robotics_samples]

    # save embeddings
    np.save("embeddings.npy", np.array(embeddings))
    with open("texts.txt", "w") as f:
        for text in robotics_samples:
            f.write(text + "\n")


if __name__ == "__main__":
    generate_sample_data()
