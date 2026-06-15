import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from pytorch_metric_learning import losses, miners

# Import our custom modules
from dataset import IndieDogDataset
from model import DogEmbedder

def train():
    # 1. Configuration
    DATA_DIR = "../data/processed_crops" # Path to your YOLO-cropped frames
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {DEVICE}")

    # 2. Setup Dataset and DataLoader
    train_dataset = IndieDogDataset(root_dir=DATA_DIR, is_training=True)
    
    if len(train_dataset) == 0:
        print("Error: No images found. Run the YOLO video processor first!")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    print(f"Loaded {len(train_dataset)} images across {len(train_dataset.classes)} unique dogs.")

    # 3. Initialize Model, Optimizer, Miner, and Loss
    model = DogEmbedder(embedding_dim=128).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # The Miner finds the hardest pairs in the batch
    miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="hard")
    # The Loss calculates the penalty based on those hard pairs
    loss_func = losses.TripletMarginLoss(margin=0.2)

    # 4. The Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Generate 128-D embeddings for the batch
            embeddings = model(images)
            
            # Mine the hard triplets and calculate loss
            hard_pairs = miner(embeddings, labels)
            loss = loss_func(embeddings, labels, hard_pairs)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Average Triplet Loss: {avg_loss:.4f}")

    # 5. Save the trained weights
    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/indie_dog_reid_v1.pth")
    print("Training complete. Model weights saved to weights/indie_dog_reid_v1.pth")

if __name__ == "__main__":
    train()