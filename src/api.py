import torch
import torch.nn as nn
from flask import Flask, request, jsonify
import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer

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

# Load models and datasets
# Model 1: Skin Conditions -> Ingredients
input_size_1 = 20
hidden_size_1 = 64
dataset_ingredients = pd.read_csv('./dataset/ingredientsList.csv')
ingredients = {
    row['name']: {'description': row['short_description'], 'avoid_if': row['who_should_avoid']}
    for _, row in dataset_ingredients.iterrows()
}
all_conditions = set()
for value in dataset_ingredients['who_is_it_good_for']:
    unique_condition = ast.literal_eval(value) if isinstance(value, str) else value
    all_conditions.update(unique_condition)
all_conditions_list = list(all_conditions)
output_size_1 = len(ingredients)

model_1 = SkinConditionModel(input_size_1, hidden_size_1, output_size_1)
model_1.load_state_dict(torch.load('skinclarity_recommendation_model.pth'))
model_1.eval()

# Model 2: Ingredients -> Products
hidden_size_2 = 256
dataset_products = pd.read_csv('./dataset/productsList.csv')
products = {
    row['name']: {'type': row['type'], 'afterUse': row['afterUse'], 'ingredients': row['ingredients'], 'brand': row['brand']}
    for _, row in dataset_products.iterrows()
}
dataset_products['ingredients'] = dataset_products['ingredients'].str.replace(r'\s*\d+(\.\d+)?%', '', regex=True)
all_ingredients = ','.join(dataset_products['ingredients'].fillna(''))
unique_ingredients = set([ingredient.strip() for ingredient in all_ingredients.split(',')])
input_size_2 = len(unique_ingredients)
output_size_2 = len(products)

model_2 = SkinConditionModel(input_size_2, hidden_size_2, output_size_2)
model_2.load_state_dict(torch.load('skinclarity_recommendation_model_v2.pth'))
model_2.eval()

# Preprocessing utilities
def process_conditions(input_conditions):
    mlb = MultiLabelBinarizer()
    mlb.fit([all_conditions_list])
    input_binary = mlb.transform([input_conditions])
    return torch.tensor(input_binary.flatten(), dtype=torch.float32)

def process_ingredients(input_ingredients):
    mlb = MultiLabelBinarizer()
    mlb.fit([unique_ingredients])
    input_binary = mlb.transform([input_ingredients])
    return torch.tensor(input_binary.flatten(), dtype=torch.float32)

# Routes
@app.route('/recommend', methods=['POST'])
def recommend():
    # Get input data from the request
    input_conditions = request.json.get('condition')
    if not input_conditions or not isinstance(input_conditions, list):
        return jsonify({"error": "Invalid input. Please provide a 'condition' key with a list of conditions."}), 400

    # Step 1: Predict ingredients based on skin conditions
    input_vector = process_conditions(input_conditions)
    with torch.no_grad():
        predicted_ingredients = torch.sigmoid(model_1(input_vector))
    good_ingredient_indices = (predicted_ingredients > 0.5).squeeze().tolist()
    
    # Map ingredient indices to names
    ingredient_names = list(ingredients.keys())
    recommended_ingredients = [
        {
            "name": ingredient_names[i],
            "short_description": ingredients[ingredient_names[i]]['description'],
            "avoid_if": str([x for x in ast.literal_eval(ingredients[ingredient_names[i]]['avoid_if']) if x.strip()])
        }
        for i in range(len(good_ingredient_indices)) if good_ingredient_indices[i]
    ]
    
    # Extract ingredient names for the next step
    selected_ingredients = [
        ingredient_names[i] for i in range(len(good_ingredient_indices)) if good_ingredient_indices[i]
    ]

    # Step 2: Filter products that contain the recommended ingredients
    recommended_products = {}
    for ingredient in selected_ingredients:
        recommended_products[ingredient] = []

        for product_name, product_info in products.items():
            # Safely retrieve ingredients list, defaulting to an empty list
            product_ingredients = product_info.get("ingredients", [])
            
            # Ensure product_ingredients is a list, and handle cases where it may be NaN or a float
            if isinstance(product_ingredients, str):
                product_ingredients = [ing.strip() for ing in product_ingredients.split(",")]
            elif isinstance(product_ingredients, float):  # If the value is NaN or a number
                product_ingredients = []

            # Check if the ingredient is in the product's ingredient list
            if ingredient in product_ingredients:
                recommended_products[ingredient].append({
                    "name": product_name,
                    "type": product_info['type'],
                    "afterUse": product_info['afterUse'],
                    "ingredients": product_info['ingredients'],
                    'brand': product_info['brand']
                })

        # Limit to 10 products per ingredient
        recommended_products[ingredient] = recommended_products[ingredient][:10]

    # Combine results into a single response
    response = {
        "recommended_ingredients": recommended_ingredients,
        "recommended_products": recommended_products  # A dictionary keyed by ingredient name
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=False)
