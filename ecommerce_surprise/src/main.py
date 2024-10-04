import pandas as pd
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from collections import defaultdict

# Step 1: Prepare the data
def load_data(ratings_file):
    df = pd.read_csv(ratings_file)
    reader = Reader(rating_scale=(1, 5))
    return Dataset.load_from_df(df[['productId', 'userId', 'rating']], reader)

# Step 2: Split the data into training and testing sets
def split_data(data):
    return train_test_split(data, test_size=0.25, random_state=42)

# Step 3: Train the model
def train_model(trainset):
    model = SVD()
    model.fit(trainset)
    return model

# Step 4: Get top N recommendations for a user
def get_top_n_recommendations(model, trainset, userId, n=10):
    # Get all products
    all_products = trainset.all_items()
    
    # Get products the user has already rated
    user_data = trainset.ur[trainset.to_inner_uid(userId)]
    user_rated_products = [trainset.to_raw_iid(product) for (product, _) in user_data]
    
    # Get predictions for all products the user hasn't rated
    predictions = []
    for product in all_products:
        if trainset.to_raw_iid(product) not in user_rated_products:
            predictions.append(model.predict(userId, trainset.to_raw_iid(product)))
    
    # Sort predictions by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    # Return top N recommendations
    return [pred.iid for pred in predictions[:n]]

# Main function to run the recommendation engine
def run_recommendation_engine(ratings_file, userId):
    # Load data
    data = load_data(ratings_file)
    
    # Split data
    trainset, testset = split_data(data)
    
    # Train model
    model = train_model(trainset)
    
    # Get recommendations
    recommendations = get_top_n_recommendations(model, trainset, userId)
    
    return recommendations

# Example usage
if __name__ == "__main__":
    ratings_file = "../data/user_ratings.csv"
    userId = "21983"
    recommendations = run_recommendation_engine(ratings_file, userId)
    print(f"Top 10 recommendations for user {userId}:")
    for i, product in enumerate(recommendations, 1):
        print(f"{i}. {product}")
