import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Read the product IDs from the CSV file
product_ids = pd.read_csv('rcpl-product.csv')['product_id'].astype(str).tolist()

# Set random seed for reproducibility
np.random.seed(42)

# Define possible values for various fields
sub_categories = ['Fruits', 'Vegetables', 'Dairy', 'Bakery', 'Meat', 'Beverages', 'Snacks', 'Canned Goods', 'Frozen Foods', 'Condiments']
brands = ['FreshFarm', 'NaturePlus', 'GreenGrocer', 'HealthyChoice', 'OrganicLife', 'EcoHarvest', 'PureTaste', 'NutriSelect', 'EarthBite', 'VitaVeg']
units = ['kg', 'g', 'l', 'ml', 'pack']

# Generate the dataset
data = []
for product_id in product_ids:
    sub_category = np.random.choice(sub_categories)
    brand = np.random.choice(brands)
    
    product = {
        'product_id': product_id,
        'name': f'{sub_category} Item {product_id[-4:]}',
        'category': 'Groceries',
        'sub_category': sub_category,
        'brand': brand,
        'price': round(np.random.uniform(1, 50), 2),
        'unit': np.random.choice(units),
        'quantity': np.random.choice([1, 6, 12, 24, 100, 250, 500, 1000]),
        'organic': np.random.choice([True, False]),
        'expiry_days': np.random.randint(1, 365)
    }
    
    data.append(product)

# Convert to DataFrame
df = pd.DataFrame(data)

# Add empty columns for non-grocery fields to match the desired format
non_grocery_fields = ['size', 'color', 'material', 'gender', 'release_year', 'screen_size', 'memory', 
                      'processor', 'os', 'camera', 'battery', 'type', 'dosage', 'condition', 
                      'requires_prescription', 'side_effects', 'expiry_date']
for field in non_grocery_fields:
    df[field] = ''

# Reorder columns to match the desired format
column_order = ['product_id', 'name', 'category', 'sub_category', 'brand', 'price', 'size', 'color', 
                'material', 'gender', 'release_year', 'screen_size', 'memory', 'processor', 'os', 
                'camera', 'battery', 'type', 'dosage', 'condition', 'requires_prescription', 
                'side_effects', 'expiry_date', 'unit', 'quantity', 'organic', 'expiry_days']
df = df[column_order]

# Save to CSV
df.to_csv('rcpl-grocery_dataset.csv', index=False)

print("Grocery dataset has been generated and saved as 'grocery_dataset.csv'")
print("\nSample data:")
print(df.head().to_string())