import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_data(ratings_file):
    return pd.read_csv(ratings_file)

def create_user_item_matrix(df):
    # Group by userId and productId, and take the mean of ratings
    df_grouped = df.groupby(['userId', 'productId'])['rating'].mean().reset_index()
    return df_grouped.pivot(index='userId', columns='productId', values='rating').fillna(0)

def compute_user_similarity(user_item_matrix):
    return pd.DataFrame(cosine_similarity(user_item_matrix), 
                        index=user_item_matrix.index, 
                        columns=user_item_matrix.index)

def get_top_n_recommendations(user_id, user_item_matrix, user_similarity, n=10):
    if user_id not in user_item_matrix.index:
        print(f"User {user_id} not found in the dataset.")
        return pd.Series()

    user_ratings = user_item_matrix.loc[user_id]
    similar_users = user_similarity[user_id].sort_values(ascending=False)[1:11]  # top 10 similar users
    
    similar_users_ratings = user_item_matrix.loc[similar_users.index]
    
    # Weight ratings by similarity
    weighted_ratings = similar_users_ratings.mul(similar_users, axis=0)
    
    # Sum of weights (similarities)
    sum_of_weights = similar_users.sum()
    
    # Weighted average of ratings
    recommendations = weighted_ratings.sum() / sum_of_weights
    
    # Filter out items the user has already rated
    recommendations = recommendations[user_ratings == 0]
    
    return recommendations.sort_values(ascending=False).head(n)

def run_recommendation_engine(ratings_file, user_id):
    # Load data
    df = load_data(ratings_file)
    
    # Create user-item matrix
    user_item_matrix = create_user_item_matrix(df)
    
    # Compute user similarity
    user_similarity = compute_user_similarity(user_item_matrix)
    
    # Get recommendations
    recommendations = get_top_n_recommendations(user_id, user_item_matrix, user_similarity)
    
    return recommendations

# Example usage
if __name__ == "__main__":
    ratings_file = "../data/user_ratings_simplified.csv"
    user_id = "125"
    recommendations = run_recommendation_engine(ratings_file, user_id)
    if not recommendations.empty:
        print(f"Top 10 recommendations for user {user_id}:")
        for i, (product, score) in enumerate(recommendations.items(), 1):
            print(f"{i}. Product {product} (Score: {score:.2f})")
    else:
        print("No recommendations could be generated.")