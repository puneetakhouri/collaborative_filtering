import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

def load_data(orders_file, catalog_file):
    orders = pd.read_csv(orders_file)
    catalog = pd.read_csv(catalog_file)
    return orders, catalog

def preprocess_data(orders, catalog):
    # Create user-item matrix for collaborative filtering
    user_item = orders.pivot(index='user_id', columns='product_id', values='rating').fillna(0)
    user_to_index = {user: i for i, user in enumerate(user_item.index)}
    item_to_index = {item: i for i, item in enumerate(user_item.columns)}
    
    sparse_user_item = csr_matrix(user_item.values)
    
    # Preprocess catalog for content-based filtering
    numeric_features = ['price', 'feature1', 'feature2', 'feature3']
    scaler = MinMaxScaler()
    catalog[numeric_features] = scaler.fit_transform(catalog[numeric_features])
    
    categorical_features = ['category', 'brand']
    catalog_encoded = pd.get_dummies(catalog, columns=categorical_features)
    
    return sparse_user_item, user_to_index, item_to_index, catalog_encoded

def train_als_model(sparse_user_item, factors=50, iterations=15):
    model = AlternatingLeastSquares(factors=factors, iterations=iterations)
    model.fit(sparse_user_item)
    return model

def get_collaborative_recommendations(model, user_index, sparse_user_item, item_to_index, n=10):
    recommendations = model.recommend(user_index, sparse_user_item[user_index], N=n)
    return [(list(item_to_index.keys())[list(item_to_index.values()).index(item)], score) for item, score in zip(*recommendations)]

def get_content_based_recommendations(product_id, catalog_encoded, n=10):
    product_features = catalog_encoded.set_index('product_id')
    product_vec = product_features.loc[product_id].values.reshape(1, -1)
    similarity_scores = cosine_similarity(product_vec, product_features.values)
    similar_indices = similarity_scores.argsort()[0][::-1][1:n+1]
    return [(product_features.index[i], similarity_scores[0][i]) for i in similar_indices]

def hybrid_recommendations(user_id, product_id, orders, catalog, als_model, user_to_index, item_to_index, catalog_encoded, n=10):
    collab_recs = get_collaborative_recommendations(als_model, user_to_index[user_id], 
                                                   csr_matrix(orders.pivot(index='user_id', columns='product_id', values='rating').fillna(0).values), 
                                                   item_to_index, n)
    content_recs = get_content_based_recommendations(product_id, catalog_encoded, n)
    
    # Combine and normalize scores
    all_recs = collab_recs + content_recs
    products, scores = zip(*all_recs)
    normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    
    # Remove duplicates and sort
    unique_recs = sorted(set(zip(products, normalized_scores)), key=lambda x: x[1], reverse=True)
    
    return unique_recs[:n]

def main():
    orders_file = 'user_orders.csv'
    catalog_file = 'product_catalog.csv'
    
    orders, catalog = load_data(orders_file, catalog_file)
    sparse_user_item, user_to_index, item_to_index, catalog_encoded = preprocess_data(orders, catalog)
    
    als_model = train_als_model(sparse_user_item)
    
    # Example usage
    user_id = orders['user_id'].iloc[0]
    product_id = orders['product_id'].iloc[0]
    
    recommendations = hybrid_recommendations(user_id, product_id, orders, catalog, als_model, user_to_index, item_to_index, catalog_encoded)
    
    print(f"Top 10 recommendations for user {user_id} based on product {product_id}:")
    for product, score in recommendations:
        print(f"Product: {product}, Score: {score:.4f}")

if __name__ == "__main__":
    main()