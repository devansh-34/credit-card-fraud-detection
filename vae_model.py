
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ==================================================
# VARIATIONAL AUTOENCODER - SYNTHETIC FRAUD GENERATION
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "processed"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("=" * 60)
print("       VAE - SYNTHETIC FRAUD GENERATION")
print("=" * 60)
print("Device:", device)

# --------------------------------------------------
# 1. LOAD TRAINING DATA ONLY
# --------------------------------------------------

print("\n1. Loading training data...")

X_train = np.load(DATA_DIR / "X_train.npy").astype(np.float32)
y_train = np.load(DATA_DIR / "y_train.npy").reshape(-1)

# Select only fraudulent training transactions
X_fraud = X_train[y_train == 1]

print("Training dataset:", X_train.shape)
print("Fraud samples:", X_fraud.shape)

if len(X_fraud) < 2:
    raise ValueError("Not enough fraud samples to train VAE.")

# --------------------------------------------------
# 2. PREPARE PYTORCH DATA
# --------------------------------------------------

fraud_tensor = torch.tensor(X_fraud, dtype=torch.float32)

dataset = TensorDataset(fraud_tensor)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# --------------------------------------------------
# 3. DEFINE VAE ARCHITECTURE
# --------------------------------------------------

class VAE(nn.Module):

    def __init__(self, input_dim=30, hidden_dim=32, latent_dim=8):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):

        hidden = self.encoder(x)

        mean = self.mean_layer(hidden)
        logvar = self.logvar_layer(hidden)

        return mean, logvar

    def reparameterize(self, mean, logvar):

        std = torch.exp(0.5 * logvar)

        epsilon = torch.randn_like(std)

        return mean + epsilon * std

    def decode(self, z):

        return self.decoder(z)

    def forward(self, x):

        mean, logvar = self.encode(x)

        z = self.reparameterize(mean, logvar)

        reconstructed = self.decode(z)

        return reconstructed, mean, logvar


# --------------------------------------------------
# 4. INITIALIZE MODEL
# --------------------------------------------------

model = VAE(
    input_dim=30,
    hidden_dim=32,
    latent_dim=8
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print("\n2. VAE initialized")
print("Input features: 30")
print("Hidden neurons: 32")
print("Latent dimensions: 8")

# --------------------------------------------------
# 5. DEFINE VAE LOSS
# --------------------------------------------------

def vae_loss(reconstructed, original, mean, logvar):

    reconstruction_loss = nn.functional.mse_loss(
        reconstructed,
        original,
        reduction="sum"
    )

    kl_divergence = -0.5 * torch.sum(
        1 + logvar - mean.pow(2) - logvar.exp()
    )

    total_loss = reconstruction_loss + kl_divergence

    return total_loss


# --------------------------------------------------
# 6. TRAIN VAE
# --------------------------------------------------

epochs = 100

print("\n3. Starting VAE training...")

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for (batch,) in loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        reconstructed, mean, logvar = model(batch)

        loss = vae_loss(
            reconstructed,
            batch,
            mean,
            logvar
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataset)

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"VAE Loss: {average_loss:.4f}"
        )

# --------------------------------------------------
# 7. GENERATE SYNTHETIC FRAUD TRANSACTIONS
# --------------------------------------------------

print("\n4. Generating synthetic fraud transactions...")

model.eval()

number_of_samples = 1000

with torch.no_grad():

    latent_vectors = torch.randn(
        number_of_samples,
        8,
        device=device
    )

    synthetic_fraud = model.decode(latent_vectors)

    synthetic_fraud = synthetic_fraud.cpu().numpy()

# --------------------------------------------------
# 8. SAVE MODEL AND GENERATED DATA
# --------------------------------------------------

torch.save(
    model.state_dict(),
    MODEL_DIR / "vae_model.pth"
)

np.save(
    OUTPUT_DIR / "synthetic_fraud.npy",
    synthetic_fraud
)

print("\n5. Results")

print("Original fraud samples:", len(X_fraud))
print("Synthetic fraud samples:", len(synthetic_fraud))
print("Synthetic feature shape:", synthetic_fraud.shape)

print("\nFiles saved:")
print("models/vae_model.pth")
print("outputs/synthetic_fraud.npy")

print("\n" + "=" * 60)
print("       VAE TRAINING COMPLETED!")
print("=" * 60)
