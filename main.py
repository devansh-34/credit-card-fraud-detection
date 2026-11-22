import numpy as np
import pandas as pd
import torch
import sklearn
import streamlit

print("Python environment ready!")
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("PyTorch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())