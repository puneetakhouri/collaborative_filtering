import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, auc_score
from sklearn.model_selection import train_test_split
from collections import defaultdict
import scipy.sparse as sp

# Set random seed for reproducibility
np.random.seed(42)

# [Include all functions from the previous generate_dataset.py here]
# generate_users, generate_electronics, generate_groceries, generate_fashion, generate_medicines, 
# generate_products, generate_orders, generate_inventory, generate_warehouse_areas

def generate_dataset():
    """users_df = generate_users()
    products_df = generate_products()
    orders_df = generate_orders(users_df, products_df)
    inventory_df = generate_inventory(products_df)
    warehouse_areas_df = generate_warehouse_areas()"""

    users_df = pd.read_csv('../data/users.csv')
    products_df = pd.read_csv('../data/products.csv')
    orders_df = pd.read_csv('../data/orders.csv')
    inventory_df = pd.read_csv('../data/inventory.csv')
    warehouse_areas_df = pd.read_csv('../data/warehouse_areas.csv')

    return users_df, products_df, orders_df, inventory_df, warehouse_areas_df

def get_available_products(user_id, users_df, inventory_df, warehouse_areas_df):
    # Get user's cities
    user_cities = users_df[users_df['user_id'] == user_id]['city'].unique()
    
    # Filter inventory for these cities and items with stock > 0
    available_inventory = inventory_df[
        (inventory_df['city'].isin(user_cities)) & 
        (inventory_df['stock'] > 0)
    ]
    
    # Get unique product IDs that are available
    available_products = available_inventory['product_id'].unique()
    
    return set(available_products)

def filter_recommendations(recommendations, available_products):
    return [rec for rec in recommendations if rec[0] in available_products]


def prepare_data_for_lightfm(users_df, products_df, orders_df):
    # Prepare user features
    users_df['age_group'] = pd.cut(users_df['age'], bins=[0, 30, 50, 100], labels=['young', 'middle', 'senior'])
    user_features = {
        row['user_id']: [f"city:{row['city']}", f"gender:{row['gender']}", f"age_group:{row['age_group']}"]
        for _, row in users_df.iterrows()
    }

    # Prepare item features
    products_df['price_category'] = pd.qcut(products_df['price'], q=5, labels=['very_cheap', 'cheap', 'medium', 'expensive', 'very_expensive'])
    
    item_features = {}
    for _, row in products_df.iterrows():
        features = [
            f"category:{row['category']}",
            f"price_category:{row['price_category']}"
        ]
        
        if row['category'] == 'Medicines':
            features.extend([
                f"type:{row['type']}",
                f"condition:{row['condition']}",
                f"requires_prescription:{row['requires_prescription']}",
                f"side_effects:{row['side_effects']}"
            ])
        elif row['category'] == 'Groceries':
            features.extend([
                f"sub_category:{row['sub_category']}",
                f"unit:{row['unit']}",
                f"organic:{row['organic']}"
            ])
        elif row['category'] == 'Fashion':
            features.extend([
                f"sub_category:{row['sub_category']}",
                f"brand:{row['brand']}",
                f"size:{row['size']}",
                f"color:{row['color']}",
                f"material:{row['material']}",
                f"gender:{row['gender']}"
            ])
        elif row['category'] == 'Electronics':
            features.extend([
                f"brand:{row['brand']}",
                f"release_year:{row['release_year']}"
            ])
            if pd.notnull(row.get('screen_size')):
                features.append(f"screen_size:{row['screen_size']}")
            if pd.notnull(row.get('memory')):
                features.append(f"memory:{row['memory']}")
            if pd.notnull(row.get('processor')):
                features.append(f"processor:{row['processor']}")
            if pd.notnull(row.get('os')):
                features.append(f"os:{row['os']}")
            if pd.notnull(row.get('camera')):
                features.append(f"camera:{row['camera']}")
            if pd.notnull(row.get('battery')):
                features.append(f"battery:{row['battery']}")
        
        item_features[row['product_id']] = features

    # Create LightFM dataset
    dataset = Dataset()
    dataset.fit(
        users=users_df['user_id'],
        items=products_df['product_id'],
        user_features=set(feature for features in user_features.values() for feature in features),
        item_features=set(feature for features in item_features.values() for feature in features)
    )

    # Build interactions
    (interactions, weights) = dataset.build_interactions(orders_df[['user_id', 'product_id']].values)

    # Build feature matrices
    user_features_matrix = dataset.build_user_features(
        (user_id, features) for user_id, features in user_features.items()
    )
    item_features_matrix = dataset.build_item_features(
        (item_id, features) for item_id, features in item_features.items()
    )

    return dataset, interactions, weights, user_features_matrix, item_features_matrix

def train_and_evaluate_model(dataset, interactions, user_features, item_features):
    # Split the data into training and test sets
    train_interactions, test_interactions = train_test_split(interactions, test_size=0.2, random_state=42)

    # Create and train the model
    model = LightFM(loss='warp', no_components=64, learning_rate=0.05, random_state=42)
    model.fit(train_interactions, user_features=user_features, item_features=item_features, epochs=30, num_threads=4)

    # Evaluate the model
    train_precision = precision_at_k(model, train_interactions, k=10, user_features=user_features, item_features=item_features).mean()
    test_precision = precision_at_k(model, test_interactions, k=10, user_features=user_features, item_features=item_features).mean()

    train_auc = auc_score(model, train_interactions, user_features=user_features, item_features=item_features).mean()
    test_auc = auc_score(model, test_interactions, user_features=user_features, item_features=item_features).mean()

    print(f"Train Precision@10: {train_precision:.4f}")
    print(f"Test Precision@10: {test_precision:.4f}")
    print(f"Train AUC: {train_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

    return model

def get_home_page_recommendations(model, dataset, user_id, user_features, users_df, item_features, products_df, inventory_df, warehouse_areas_df, user_city, n=10):
    user_index = dataset.mapping()[0][user_id]
    n_items = item_features.shape[0]
    
    # Get personalized scores
    scores = model.predict(user_index, np.arange(n_items), user_features=user_features, item_features=item_features)
    
    # Get popular items (based on interaction count)
    item_popularity = np.array(item_features.sum(axis=1)).flatten()
    
    # Combine personalized scores with popularity (you can adjust the weights)
    combined_scores = 0.7 * scores + 0.3 * item_popularity

     # Get available products
    available_products = get_available_products(user_id, users_df, inventory_df, warehouse_areas_df)
    
    # Create a dictionary to store recommendations by category
    category_recommendations = defaultdict(list)
    
    # Sort items by score and group by category
    sorted_items = sorted(enumerate(combined_scores), key=lambda x: x[1], reverse=True)
    for item_index, score in sorted_items:
        product_id = products_df.iloc[item_index]['product_id']
        if product_id not in available_products:
            continue

        product_name = products_df.iloc[item_index]['name']
        category = products_df.iloc[item_index]['category']
        
        if len(category_recommendations[category]) < n:
            category_recommendations[category].append((product_id, product_name, category, score))
        
        # If we have n recommendations for each category, we can stop
        if all(len(recs) == n for recs in category_recommendations.values()):
            break
    
    return category_recommendations

def print_home_page_recommendations(recommendations):
    for category, items in recommendations.items():
        print(f"\nTop 10 recommendations for {category}:")
        for i, (product_id, product_name, _, score) in enumerate(items, 1):
            print(f"{i}. {product_name} (ID: {product_id}, Score: {score:.4f})")

def get_pdp_recommendations(model, dataset, user_id, product_id, user_features, users_df, item_features, products_df, inventory_df, warehouse_areas_df, user_city, n=5):
    user_index = dataset.mapping()[0][user_id]
    try:
        item_index = dataset.mapping()[2][product_id]
    except KeyError:
        print(f"Product ID {product_id} not found in the dataset.")
        return []

    n_items = item_features.shape[0]
    
    # Get the features of the current product
    current_item_features = item_features[item_index].toarray().flatten()
    
    # Calculate similarity between the current product and all other products
    item_features_array = item_features.toarray()
    print(f"item_features_array shape: {item_features_array.shape}")
    similarity_scores = np.dot(item_features_array, current_item_features) / (
        np.linalg.norm(item_features_array, axis=1) * np.linalg.norm(current_item_features)
    )
    
    # Combine similarity with personalized scores
    personalized_scores = model.predict(user_index, np.arange(n_items), user_features=user_features, item_features=item_features)
    combined_scores = 0.7 * similarity_scores + 0.3 * personalized_scores
    
    # Sort and get top N recommendations (excluding the current product)
    top_items = sorted(enumerate(combined_scores), key=lambda x: x[1], reverse=True)
    top_items = [item for item in top_items if item[0] != item_index][:n]
    
    # Get available products
    available_products = get_available_products(user_id, users_df, inventory_df, warehouse_areas_df)

    recommendations = []
    for item_index, score in top_items:
        product_id = products_df.iloc[item_index]['product_id']
        # Check if the product is available
        if product_id not in available_products:
            continue

        product_name = products_df.iloc[item_index]['name']
        category = products_df.iloc[item_index]['category']
        recommendations.append((product_id, product_name, category, score))
    
    return recommendations

def get_cart_recommendations(model, dataset, user_id, cart_product_ids, user_features, users_df, item_features, products_df, inventory_df, warehouse_areas_df, user_city, n=5):

    user_index = dataset.mapping()[0][user_id]
    n_items = item_features.shape[0]
    
    # Get the features of the products in the cart
    cart_item_indices = []
    for pid in cart_product_ids:
        try:
            cart_item_indices.append(dataset.mapping()[2][pid])
        except KeyError:
            print(f"Product ID {pid} not found in the dataset.")
    
    if not cart_item_indices:
        print("No valid products in the cart.")
        return []

    cart_item_features = item_features[cart_item_indices].toarray()
    
    # Calculate average features of cart items
    avg_cart_features = cart_item_features.mean(axis=0)
    print(f"avg_cart_features shape: {avg_cart_features.shape}")

    
    # Calculate similarity between the average cart features and all other products
    item_features_array = item_features.toarray()
    print(f"item_features_array shape: {item_features_array.shape}")
    similarity_scores = np.dot(item_features_array, avg_cart_features) / (
        np.linalg.norm(item_features_array, axis=1) * np.linalg.norm(avg_cart_features)
    )
    
    # Combine similarity with personalized scores
    personalized_scores = model.predict(user_index, np.arange(n_items), user_features=user_features, item_features=item_features)
    combined_scores = 0.6 * similarity_scores + 0.4 * personalized_scores
    
    # Sort and get top N recommendations (excluding items already in the cart)
    top_items = sorted(enumerate(combined_scores), key=lambda x: x[1], reverse=True)
    top_items = [item for item in top_items if item[0] not in cart_item_indices][:n]
    
    # Get available products
    available_products = get_available_products(user_id, users_df, inventory_df, warehouse_areas_df)

    recommendations = []
    for item_index, score in top_items:
        product_id = products_df.iloc[item_index]['product_id']
        # Check if the product is available
        if product_id not in available_products:
            continue
        product_name = products_df.iloc[item_index]['name']
        category = products_df.iloc[item_index]['category']
        recommendations.append((product_id, product_name, category, score))
    
    return recommendations

def main():
    # Generate dataset
    print("Generating dataset...")
    users_df, products_df, orders_df, inventory_df, warehouse_areas_df = generate_dataset()

    # Prepare data for LightFM
    print("Preparing data for LightFM...")
    dataset, interactions, weights, user_features, item_features = prepare_data_for_lightfm(users_df, products_df, orders_df)

    # Train and evaluate the model
    print("Training and evaluating the model...")
    model = train_and_evaluate_model(dataset, interactions, user_features, item_features)

    # Generate recommendations for a sample user
    print("\nGenerating recommendations for a sample user...")
    sample_user_id = "U0012"  # Sample user ID for faster execution, can be changed to any other valid user ID from the dataset
    user_city = users_df[users_df['user_id'] == sample_user_id]['city'].values[0]

    # Home page recommendations
    print(f"\nTop 10 home page recommendations for user {sample_user_id}:")
    home_recommendations = get_home_page_recommendations(model, dataset, sample_user_id, user_features, users_df, item_features, products_df, inventory_df, warehouse_areas_df, user_city)
    print_home_page_recommendations(home_recommendations)
    #for i, (product_id, product_name, category, score) in enumerate(home_recommendations, 1):
    #    print(f"{i}. {product_name} (ID: {product_id}, Category: {category}, Score: {score:.4f})")

    # PDP recommendations
    sample_product_id = products_df['product_id'].iloc[0]  # Just using the first product as an example
    print(f"\nTop 5 product detail page recommendations for user {sample_user_id} viewing product {sample_product_id}:")
    pdp_recommendations = get_pdp_recommendations(model, dataset, sample_user_id, sample_product_id, user_features, users_df, item_features, products_df, inventory_df, warehouse_areas_df, user_city)
    if pdp_recommendations:
        for i, (product_id, product_name, category, score) in enumerate(pdp_recommendations, 1):
            print(f"{i}. {product_name} (ID: {product_id}, Category: {category}, Score: {score:.4f})")
    else:
        print("No recommendations available.")

    # Cart recommendations
    sample_cart = [products_df['product_id'].iloc[i] for i in range(3)]  # Just using the first 3 products as an example cart
    print(f"\nTop 5 cart recommendations for user {sample_user_id} with cart containing {sample_cart}:")
    cart_recommendations = get_cart_recommendations(model, dataset, sample_user_id, sample_cart, user_features, users_df, item_features, products_df, inventory_df, warehouse_areas_df, user_city)
    if cart_recommendations:
        for i, (product_id, product_name, category, score) in enumerate(cart_recommendations, 1):
            print(f"{i}. {product_name} (ID: {product_id}, Category: {category}, Score: {score:.4f})")
    else:
        print("No recommendations available.")

if __name__ == "__main__":
    main()