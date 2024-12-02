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
input_size = 20
hidden_size = 64
dataset = pd.read_csv('./dataset/ingredientsList.csv')
ingredients = {row['name']: {'description': row['short_description'], 'what': row['what_is_it']} for index, row in dataset.iterrows()}
all_conditions = set()
for value in dataset['who_is_it_good_for']:
    unique_condition = ast.literal_eval(value) if isinstance(value, str) else value
    all_conditions.update(unique_condition)
all_conditions_list = list(all_conditions)
output_size = len(ingredients)

model = SkinConditionModel(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('skinclarity_recommendation_model.pth'))
model.eval()  # Set to evaluation mode

# Helper function to process user input (e.g., one-hot encoding or vectorization of conditions)
def process_conditions(input_conditions):
    # Initialize the MultiLabelBinarizer and fit it to your all_conditions_list
    mlb = MultiLabelBinarizer()
    mlb.fit([all_conditions_list])  # Fit the list of all possible conditions

    input_binary = mlb.transform([input_conditions])
    
    # Flatten the result (since the result is a 2D array)
    binary_result = input_binary.flatten().tolist()

    # Convert the binary result into a PyTorch tensor
    tensor_result = torch.tensor(binary_result, dtype=torch.float32)  # Use float32 as the model expects float inputs
    
    return tensor_result
    

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    input_condition = request.json.get('condition')  # User will send 'condition' in the POST request
    
    # Preprocess the condition (vectorization, etc.)
    input_vector = process_conditions(input_condition)

    # Perform prediction
    with torch.no_grad():
        predicted_ingredients = model(input_vector)
    
    # Apply sigmoid to get probabilities
    predicted_ingredients = torch.sigmoid(predicted_ingredients)

    # Example: Consider ingredients with probability > 0.5 as 'good' for the condition
    good_ingredients = predicted_ingredients > 0.5
    good_ingredients = good_ingredients.squeeze().tolist()  # Convert to a list

    # Map indices back to ingredient names (you need to map the indices to your ingredients list)
    ingredient_names = list(ingredients.keys())  # Now using keys of the ingredients dictionary
    recommended_ingredients = [
        {
            "name": ingredient_names[i],
            "short_description": ingredients[ingredient_names[i]]['description'],
            "what": ingredients[ingredient_names[i]]['what']
        }
        for i in range(len(good_ingredients)) if good_ingredients[i]
    ]

    ingredients_only = [ingredient_names[i] for i in range(len(good_ingredients)) if good_ingredients[i]]

    
    return jsonify({"recommended_ingredients": recommended_ingredients})

if __name__ == '__main__':
    app.run(debug=False)
