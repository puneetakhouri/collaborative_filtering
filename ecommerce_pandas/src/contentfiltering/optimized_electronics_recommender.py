import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import gc
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info(f"CSV file read successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        raise

def preprocess_data(df, categorical_features, numeric_features):
    try:
        # Ensure 'id' is treated as a string
        df['id'] = df['id'].astype(str)
        
        # Handle numeric features
        for feature in numeric_features:
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
        
        # Normalize numeric features
        scaler = MinMaxScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features].fillna(df[numeric_features].mean()))
        
        # Create dummy variables for categorical features, but keep original columns
        df_encoded = df.copy()
        for feature in categorical_features:
            dummies = pd.get_dummies(df[feature], prefix=feature, dummy_na=True)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)

        # Drop original categorical columns (including processor and os)
        df_encoded.drop(categorical_features, axis=1, inplace=True)

        # Convert boolean columns (created by get_dummies) to float
        bool_columns = df_encoded.select_dtypes(include='bool').columns
        df_encoded[bool_columns] = df_encoded[bool_columns].astype(float)

        # Ensure all columns (except 'id' and 'name') are numeric
        for col in df_encoded.columns:
            if col not in ['id', 'name']:
                df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce').fillna(0)

        logging.info(f"Data preprocessing complete. Shape: {df_encoded.shape}")
        logging.info(f"Columns after preprocessing: {df_encoded.columns}")
        logging.info(f"Data types after preprocessing: {df_encoded.dtypes}")
        return df_encoded
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise


def get_similar_products(df_processed, original_df, product_id, n=5):
    product_id = str(product_id)  # Ensure product_id is a string
    if product_id not in df_processed['id'].values:
        raise ValueError(f"Product with ID {product_id} not found in the dataset.")
    
    # Separate features for similarity calculation
    feature_columns = [col for col in df_processed.columns if col not in ['id', 'name']]
    features = df_processed[feature_columns]
    
    logging.info(f"Feature shape before conversion to sparse matrix: {features.shape}")
    #logging.info(f"Feature data types: {features.dtypes}")
    
    try:
        sparse_features = csr_matrix(features.values)
    except Exception as e:
        logging.error(f"Error converting to sparse matrix: {e}")
        raise
    
    # Get the index of the product
    idx = df_processed.index[df_processed['id'] == product_id][0]
    
    # Calculate cosine similarity for the specific product
    try:
        similarity = cosine_similarity(sparse_features[idx:idx+1], sparse_features).flatten()
    except Exception as e:
        logging.error(f"Error calculating cosine similarity: {e}")
        raise
    
    # Get top similar products
    similar_indices = similarity.argsort()[::-1][1:n+1]
    
    similar_products = df_processed.iloc[similar_indices][['id', 'name', 'price']]  # Exclude category and brand
    similar_products['similarity'] = similarity[similar_indices]
    
    # Merge with the original DataFrame to add back category and brand
    similar_products = similar_products.merge(original_df[['id', 'category', 'brand']], on='id', how='left')
    
    return similar_products


def main():
    file_path = '../../data/electronics_data.csv'
    categorical_features = ['category', 'brand', 'processor', 'os']
    numeric_features = ['price', 'screen_size', 'memory', 'camera', 'battery', 'weight', 'release_year']
    
    logging.info("Reading and processing data...")
    try:
        df = read_csv(file_path)
        df_processed = preprocess_data(df, categorical_features, numeric_features)
    except Exception as e:
        logging.error(f"Error in data processing: {e}")
        return
    
    logging.info("Data processed. Getting recommendations...")
    try:
        # Get similar products for product with ID 1
        similar_products = get_similar_products(df_processed, df, product_id=1, n=5)
        logging.info("Products similar to product 1:")
        logging.info(similar_products)

        # Clear memory
        del similar_products
        gc.collect()

        # Get similar products for product with ID 3
        similar_products = get_similar_products(df_processed, df, product_id=3, n=5)
        logging.info("\nProducts similar to product 3:")
        logging.info(similar_products)

        # Get similar products for product with ID 9
        similar_products = get_similar_products(df_processed, df, product_id=9, n=5)
        logging.info("Products similar to product 9:")
        logging.info(similar_products)

        # Clear memory
        del similar_products
        gc.collect()

        # Get similar products for product with ID 18
        similar_products = get_similar_products(df_processed, df, product_id=18, n=5)
        logging.info("\nProducts similar to product 18:")
        logging.info(similar_products)

    except Exception as e:
        logging.error(f"Error getting recommendations: {e}")

if __name__ == "__main__":
    main()