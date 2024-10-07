import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
from collections import defaultdict
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_data(ratings_file):
    logger.debug(f"Loading data from {ratings_file}")
    try:
        df = pd.read_csv(ratings_file)
        logger.debug(f"Data loaded successfully. Shape: {df.shape}")
        logger.debug(f"Columns: {df.columns}")
        logger.debug(f"Sample data:\n{df.head()}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def create_user_item_matrix(df):
    logger.debug("Creating user-item matrix")
    try:
        df['userId'] = df['userId'].astype(str)
        user_item = df.groupby(['userId', 'productId'])['rating'].mean().unstack()
        
        user_to_index = {user: i for i, user in enumerate(user_item.index)}
        item_to_index = {item: i for i, item in enumerate(user_item.columns)}
        
        row = df['userId'].map(user_to_index)
        col = df['productId'].map(item_to_index)
        data = df['rating']
        sparse_matrix = csr_matrix((data, (row, col)), shape=user_item.shape)
        
        logger.debug(f"User-item matrix created. Shape: {sparse_matrix.shape}")
        logger.debug(f"Number of users: {len(user_to_index)}")
        logger.debug(f"Number of items: {len(item_to_index)}")
        return sparse_matrix, user_to_index, item_to_index
    except Exception as e:
        logger.error(f"Error creating user-item matrix: {str(e)}")
        raise

def train_model(sparse_matrix, factors=50, iterations=15, random_seed=42):
    logger.debug("Training model")
    try:
        weighted_matrix = bm25_weight(sparse_matrix, K1=100, B=0.8)
        model = AlternatingLeastSquares(factors=factors, iterations=iterations, random_state=random_seed)
        model.fit(weighted_matrix)
        logger.debug("Model trained successfully")
        return model
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

def get_recommendations(model, sparse_matrix, user_index, user_to_index, item_to_index, item_to_category, n=10):
    logger.debug(f"Getting recommendations for user index {user_index}")
    try:
        index_to_item = {v: k for k, v in item_to_index.items()}
        recommendations = defaultdict(list)
        categories = set(item_to_category.values())
        logger.debug(f"Number of categories: {len(categories)}")
        
        initial_fetch = n * len(categories) * 2
        fetch_size = initial_fetch
        total_items = len(item_to_index)
        logger.debug(f"initial_fetch-{initial_fetch}, total_items={total_items}")
        while len([cat for cat, items in recommendations.items() if len(items) < n]) > 0 and fetch_size <= total_items:
            logger.debug(f"Fetching {fetch_size} recommendations")
            recommended = model.recommend(user_index, sparse_matrix[user_index], N=fetch_size, filter_already_liked_items=True)
            
            item_indices, scores = recommended
            
            logger.debug(f"Number of recommendations returned: {len(item_indices)}")
            
            if len(scores) > 0:
                min_score, max_score = min(scores), max(scores)
                normalized_scores = (scores - min_score) / (max_score - min_score) if max_score > min_score else scores
            else:
                logger.warning("No recommendations returned by the model")
                break
            
            for item, score in zip(item_indices, normalized_scores):
                if item < len(index_to_item):
                    product_id = index_to_item[item]
                    category = item_to_category.get(product_id, "Unknown")
                    if len(recommendations[category]) < n:
                        recommendations[category].append((product_id, score))
            
            logger.debug(f"Current recommendations count: {sum(len(items) for items in recommendations.values())}")
            
            if len([cat for cat, items in recommendations.items() if len(items) < n]) > 0:
                fetch_size *= 2
                fetch_size = min(fetch_size, total_items)
            else:
                break
        
        top_recommendations = {category: sorted(items, key=lambda x: x[1], reverse=True)[:n] 
                               for category, items in recommendations.items()}
        
        logger.debug(f"Generated recommendations for {len(top_recommendations)} categories")
        return top_recommendations
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise

def run_recommendation_engine(ratings_file, user_id, n=10):
    logger.info(f"Running recommendation engine for user {user_id}")
    try:
        df = load_data(ratings_file)
        df['userId'] = df['userId'].astype(str)
        
        sparse_matrix, user_to_index, item_to_index = create_user_item_matrix(df)
        model = train_model(sparse_matrix)
        
        item_to_category = dict(zip(df['productId'], df['category']))
        logger.debug(f"Number of items with categories: {len(item_to_category)}")
        
        if user_id in user_to_index:
            user_index = user_to_index[user_id]
            logger.debug(f"User {user_id} found. Index: {user_index}")
            recommendations = get_recommendations(model, sparse_matrix, user_index, user_to_index, item_to_index, item_to_category, n)
            return recommendations
        else:
            logger.warning(f"User {user_id} not found in the dataset.")
            return {}
    except Exception as e:
        logger.error(f"Error in recommendation engine: {str(e)}")
        return {}

if __name__ == "__main__":
    ratings_file = "../../data/user_ratings_with_category.csv"
    user_id = "100020"
    recommendations = run_recommendation_engine(ratings_file, user_id)
    if recommendations:
        print(f"Top recommendations for user {user_id} by category:")
        for category, items in recommendations.items():
            print(f"\n{category}:")
            for i, (product, score) in enumerate(items, 1):
                print(f"{i}. Product {product} (Score: {score:.4f})")
    else:
        print("No recommendations could be generated.")