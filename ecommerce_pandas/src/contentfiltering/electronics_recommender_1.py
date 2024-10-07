import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Sample data (you would replace this with your actual product database)
electronics_data = {
    'id': [1, 2, 3, 4, 5],
    'name': ['iPhone 12', 'Samsung Galaxy S21', 'MacBook Pro', 'Dell XPS 13', 'Sony Bravia TV'],
    'category': ['Smartphone', 'Smartphone', 'Laptop', 'Laptop', 'TV'],
    'brand': ['Apple', 'Samsung', 'Apple', 'Dell', 'Sony'],
    'price': [799, 799, 1299, 999, 1099],
    'screen_size': [6.1, 6.2, 13.3, 13.4, 55],
    'memory': [64, 128, 256, 512, np.nan],
    'processor': ['A14', 'Exynos 2100', 'M1', 'Intel i7', np.nan],
    'os': ['iOS', 'Android', 'macOS', 'Windows', 'Android TV'],
    'camera': [12, 64, np.nan, np.nan, np.nan],
    'battery': [2815, 4000, 58, 52, np.nan],
    'weight': [164, 169, 1400, 1200, 15000],
    'release_year': [2020, 2021, 2020, 2021, 2021]
}

df = pd.DataFrame(electronics_data)

def preprocess_data(df):
    # Create a copy of the dataframe to avoid SettingWithCopyWarning
    df_encoded = df.copy()
    
    # Create dummy variables for categorical features
    categorical_features = ['category', 'brand', 'os', 'processor']
    df_encoded = pd.get_dummies(df_encoded, columns=categorical_features)
    
    # Convert to numeric type and handle NaN values
    numeric_columns = ['price', 'screen_size', 'memory', 'camera', 'battery', 'weight', 'release_year']
    for col in numeric_columns:
        df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
    
    # Normalize numeric features
    scaler = MinMaxScaler()
    df_encoded[numeric_columns] = scaler.fit_transform(df_encoded[numeric_columns].fillna(df_encoded[numeric_columns].mean()))
    
    return df_encoded

def get_similar_products(df, df_encoded, product_id, n=5):
    if product_id not in df['id'].values:
        raise ValueError(f"Product with ID {product_id} not found in the dataset.")
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(df_encoded.drop(['id', 'name'], axis=1))
    
    # Get the index of the product in the dataframe
    idx = df.index[df['id'] == product_id][0]
    
    # Get the pairwise similarity scores for the given product
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the products based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the n most similar products (excluding itself)
    sim_scores = sim_scores[1:n+1]
    
    # Get the product indices
    product_indices = [i[0] for i in sim_scores]
    
    # Return the top n most similar products
    return df.iloc[product_indices][['id', 'name', 'category', 'brand', 'price']]

# Preprocess the data
df_encoded = preprocess_data(df)

try:
    # Get similar products for product with ID 1 (iPhone 12)
    similar_products = get_similar_products(df, df_encoded, product_id=1, n=3)
    print("Products similar to iPhone 12:")
    print(similar_products)

    # Get similar products for product with ID 3 (MacBook Pro)
    similar_products = get_similar_products(df, df_encoded, product_id=3, n=3)
    print("\nProducts similar to MacBook Pro:")
    print(similar_products)

    # Try to get similar products for a non-existent product ID
    similar_products = get_similar_products(df, df_encoded, product_id=10, n=3)
except ValueError as e:
    print(f"Error: {e}")