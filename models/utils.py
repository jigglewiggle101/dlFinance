 # Utilities for model training and evaluation (e.g., data preprocessing)
 
import numpy as np

# Example utility function for data scaling
def scale_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
