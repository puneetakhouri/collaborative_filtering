import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight

def load_data(ratings_file):
    return pd.read_csv(ratings_file)

def create_user_item_matrix(df):
    df['userId'] = df['userId'].astype(str)
    user_item = df.groupby(['userId', 'productId'])['rating'].mean().unstack()
    
    user_to_index = {user: i for i, user in enumerate(user_item.index)}
    item_to_index = {item: i for i, item in enumerate(user_item.columns)}
    
    row = df['userId'].map(user_to_index)
    col = df['productId'].map(item_to_index)
    data = df['rating']
    sparse_matrix = csr_matrix((data, (row, col)), shape=user_item.shape)
    
    return sparse_matrix, user_to_index, item_to_index

def train_model(sparse_matrix, factors=50, iterations=15, random_seed=42):
    weighted_matrix = bm25_weight(sparse_matrix, K1=100, B=0.8)
    model = AlternatingLeastSquares(factors=factors, iterations=iterations, random_state=random_seed)
    model.fit(weighted_matrix)
    return model

""" def get_recommendations(model, sparse_matrix, user_index, user_to_index, item_to_index, n=10):
    user_id = list(user_to_index.keys())[list(user_to_index.values()).index(user_index)]
    
    recommended = model.recommend(user_index, sparse_matrix[user_index], N=n, filter_already_liked_items=True)
    print(f"recommended - {recommended}")
    
    index_to_item = {v: k for k, v in item_to_index.items()}
    print(f"length of index to item- {len(index_to_item)}")
    
    recommendations = []
    for item, score, *_ in recommended:
        if item < len(index_to_item):
            recommendations.append((index_to_item[int(item)], score))
        if len(recommendations) == n:
            break

    return recommendations """

""" def get_recommendations(model, sparse_matrix, user_index, user_to_index, item_to_index, n=10):
    user_id = list(user_to_index.keys())[list(user_to_index.values()).index(user_index)]
    
    recommended = model.recommend(user_index, sparse_matrix[user_index], N=n, filter_already_liked_items=True)
    print(f"recommended - {recommended}")
    
    index_to_item = {v: k for k, v in item_to_index.items()}
    print(f"length of index to item- {len(index_to_item)}")
    
    recommendations = []
    item_indices, scores = recommended
    for item, score in zip(item_indices, scores):
        if item < len(index_to_item):
            recommendations.append((index_to_item[item], score))
        if len(recommendations) == n:
            break

    return recommendations """

def get_recommendations(model, sparse_matrix, user_index, user_to_index, item_to_index, n=10):
    #user_id = list(user_to_index.keys())[list(user_to_index.values()).index(user_index)]
    
    recommended = model.recommend(user_index, sparse_matrix[user_index], N=n, filter_already_liked_items=True)
    
    index_to_item = {v: k for k, v in item_to_index.items()}
    
    recommendations = []
    item_indices, scores = recommended
    
    # Normalize scores to 0-1 range
    min_score, max_score = min(scores), max(scores)
    normalized_scores = (scores - min_score) / (max_score - min_score)
    
    for item, score in zip(item_indices, normalized_scores):
        if item < len(index_to_item):
            recommendations.append((index_to_item[item], score))
        if len(recommendations) == n:
            break

    return recommendations

def run_recommendation_engine(ratings_file, user_id, n=10):
    df = load_data(ratings_file)
    df['userId'] = df['userId'].astype(str)
    
    sparse_matrix, user_to_index, item_to_index = create_user_item_matrix(df)
    model = train_model(sparse_matrix)
    
    if user_id in user_to_index:
        user_index = user_to_index[user_id]
        recommendations = get_recommendations(model, sparse_matrix, user_index, user_to_index, item_to_index, n)
        return recommendations
    else:
        print(f"User {user_id} not found in the dataset.")
        return []

if __name__ == "__main__":
    ratings_file = "../data/user_ratings.csv"
    user_id = "100020"
    recommendations = run_recommendation_engine(ratings_file, user_id)
    if recommendations:
        print(f"Top 10 recommendations for user {user_id}:")
        for i, (product, score) in enumerate(recommendations, 1):
            print(f"{i}. Product {product} (Score: {score:.4f})")
    else:
        print("No recommendations could be generated.")