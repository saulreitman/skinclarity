import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, request, jsonify
import pandas as pd
import ast
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

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model

hidden_size = 256
dataset = pd.read_csv('./dataset/productsList.csv')
df = pd.DataFrame(dataset)
products = {row['name']: {'type': row['type'], 'afterUse': row['afterUse']} for index, row in dataset.iterrows()}
df['ingredients'] = df['ingredients'].str.replace(r'\s*\d+(\.\d+)?%', '', regex=True)
# Combine all ingredients into a single string
all_ingredients = ','.join(df['ingredients'].fillna(''))

ingredient_list = [ingredient.strip() for ingredient in all_ingredients.split(',')]

unique_ingredients = set(ingredient_list)
input_size = len(unique_ingredients)
output_size = len(products)

model = SkinConditionModel(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('skinclarity_recommendation_model_v2.pth'))
model.eval()  # Set to evaluation mode

# Helper function to process user input (e.g., one-hot encoding or vectorization of conditions)
def process_ingredients(input_ingredients):
    # Initialize the MultiLabelBinarizer and fit it to your all_conditions_list
    mlb = MultiLabelBinarizer()
    mlb.fit([unique_ingredients])  # Fit the list of all possible conditions

    input_binary = mlb.transform([input_ingredients])
    
    # Flatten the result (since the result is a 2D array)
    binary_result = input_binary.flatten().tolist()

    # Convert the binary result into a PyTorch tensor
    tensor_result = torch.tensor(binary_result, dtype=torch.float32)  # Use float32 as the model expects float inputs
    
    return tensor_result
    

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    input_condition = request.json.get('ingredients')  # User will send 'condition' in the POST request
    
    # Preprocess the condition (vectorization, etc.)
    input_vector = process_ingredients(input_condition)

    # Perform prediction
    with torch.no_grad():
        predicted_products = model(input_vector)
    
    # Apply sigmoid to get probabilities
    predicted_products = torch.sigmoid(predicted_products)

    # Example: Consider ingredients with probability > 0.5 as 'good' for the condition
    good_products = predicted_products > 0.5
    good_products = good_products.squeeze().tolist()  # Convert to a list

    # Map indices back to ingredient names (you need to map the indices to your ingredients list)
    product_names = list(products.keys())  # Now using keys of the ingredients dictionary
    recommended_ingredients = [
        {
            "name": product_names[i],
            "type": products[product_names[i]]['type'],
            "afterUse": products[product_names[i]]['afterUse']
        }
        for i in range(len(good_products)) if good_products[i]
    ]

    ingredients_only = [product_names[i] for i in range(len(good_products)) if good_products[i]]

    
    return jsonify({"recommended_products": recommended_ingredients})

if __name__ == '__main__':
    app.run(debug=True)
