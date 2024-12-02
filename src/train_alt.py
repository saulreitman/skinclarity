import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer

# Define the network architecture
class SkinConditionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SkinConditionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

dataset = pd.read_csv('./dataset/productsList.csv')
df = pd.DataFrame(dataset)
df['ingredients'] = df['ingredients'].str.replace(r'\s*\d+(\.\d+)?%', '', regex=True)
# Combine all ingredients into a single string
all_ingredients = ','.join(df['ingredients'].fillna(''))
ingredient_list = [ingredient.strip() for ingredient in all_ingredients.split(',')]
unique_ingredients = set(ingredient_list)

products = {row['name']: {'type': row['type'], 'afterUse': row['afterUse']} for index, row in dataset.iterrows()}

input_size = len(unique_ingredients)
hidden_size = 256
output_size = len(products)  # Number of ingredients (one for each)

model = SkinConditionModel(input_size, hidden_size, output_size)

input_data = torch.randn(10, input_size)  # 10 samples, each with 'input_size' features (condition vector)
output_data = torch.randn(10, output_size)

criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(input_data)
    
    # Compute loss
    loss = criterion(outputs, output_data)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

torch.save(model.state_dict(), 'skinclarity_recommendation_model_v2.pth')
