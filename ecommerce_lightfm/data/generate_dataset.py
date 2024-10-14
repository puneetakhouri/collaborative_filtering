import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate User Data (unchanged)
def generate_users(n_users=1000):
    users = []
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
    for i in range(n_users):
        user_cities = np.random.choice(cities, size=np.random.randint(1, 3), replace=False)
        for city in user_cities:
            users.append({
                'user_id': f'U{i+1:04d}',
                'city': city,
                'age': np.random.randint(18, 70),
                'gender': np.random.choice(['M', 'F'])
            })
    return pd.DataFrame(users)

# Generate Electronics Data
def generate_electronics(n_products):
    electronics = []
    brands = ['Apple', 'Samsung', 'Sony', 'LG', 'Dell', 'Lenovo', 'Bose', 'Canon', 'Microsoft']
    product_types = ['Smartphone', 'Laptop', 'TV', 'Tablet', 'Camera', 'Headphones', 'Gaming Console']
    
    for i in range(n_products):
        product_type = np.random.choice(product_types)
        brand = np.random.choice(brands)
        
        product = {
            'product_id': f'E{i+1:04d}',
            'name': f'{brand} {product_type}',
            'category': 'Electronics',
            'brand': brand,
            'price': np.random.uniform(100, 2000),
            'release_year': np.random.randint(2018, 2024)
        }
        
        if product_type in ['Smartphone', 'Tablet', 'Laptop', 'TV']:
            product['screen_size'] = np.random.choice([5.5, 6.1, 6.7, 10.9, 13.3, 15.6, 27, 32, 43, 55, 65])
        
        if product_type in ['Smartphone', 'Tablet', 'Laptop']:
            product['memory'] = np.random.choice([64, 128, 256, 512, 1024])
            product['processor'] = np.random.choice(['A14', 'M1', 'Intel i5', 'Intel i7', 'Snapdragon 888', 'Exynos 2100'])
            product['os'] = np.random.choice(['iOS', 'Android', 'Windows', 'macOS'])
        
        if product_type in ['Smartphone', 'Tablet', 'Camera']:
            product['camera'] = np.random.choice([12, 16, 24, 48, 64, 108])
        
        if product_type in ['Smartphone', 'Tablet', 'Laptop']:
            product['battery'] = np.random.randint(2000, 5000)
        
        electronics.append(product)
    
    return electronics

# Generate Groceries Data
def generate_groceries(n_products):
    groceries = []
    categories = ['Fruits', 'Vegetables', 'Dairy', 'Bakery', 'Meat', 'Beverages', 'Snacks']
    units = ['kg', 'g', 'l', 'ml', 'pack']
    
    for i in range(n_products):
        category = np.random.choice(categories)
        unit = np.random.choice(units)
        
        product = {
            'product_id': f'G{i+1:04d}',
            'name': f'{category} Item {i+1}',
            'category': 'Groceries',
            'sub_category': category,
            'price': np.random.uniform(1, 50),
            'unit': unit,
            'quantity': np.random.choice([1, 6, 12, 24, 100, 250, 500, 1000]),
            'organic': np.random.choice([True, False]),
            'expiry_days': np.random.randint(1, 365)
        }
        
        groceries.append(product)
    
    return groceries

# Generate Fashion Data
def generate_fashion(n_products):
    fashion = []
    categories = ['Shirts', 'Pants', 'Dresses', 'Shoes', 'Accessories']
    sizes = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
    colors = ['Red', 'Blue', 'Green', 'Black', 'White', 'Yellow', 'Purple']
    brands = ['Nike', 'Adidas', 'Zara', 'H&M', 'Gucci', 'Levi\'s', 'Under Armour']
    
    for i in range(n_products):
        category = np.random.choice(categories)
        
        product = {
            'product_id': f'F{i+1:04d}',
            'name': f'{category} Item {i+1}',
            'category': 'Fashion',
            'sub_category': category,
            'brand': np.random.choice(brands),
            'price': np.random.uniform(10, 500),
            'size': np.random.choice(sizes),
            'color': np.random.choice(colors),
            'material': np.random.choice(['Cotton', 'Polyester', 'Leather', 'Denim', 'Silk']),
            'gender': np.random.choice(['Men', 'Women', 'Unisex'])
        }
        
        fashion.append(product)
    
    return fashion

# Generate Medicines Data
def generate_medicines(n_products):
    medicines = []
    types = ['Tablet', 'Capsule', 'Syrup', 'Injection', 'Cream']
    conditions = ['Pain Relief', 'Fever', 'Cold & Flu', 'Allergy', 'Digestive Health', 'Heart Health', 'Diabetes']
    
    for i in range(n_products):
        product = {
            'product_id': f'M{i+1:04d}',
            'name': f'Medicine {i+1}',
            'category': 'Medicines',
            'type': np.random.choice(types),
            'price': np.random.uniform(5, 100),
            'dosage': f'{np.random.randint(1, 1000)} mg',
            'condition': np.random.choice(conditions),
            'requires_prescription': np.random.choice([True, False]),
            'side_effects': np.random.choice(['Drowsiness', 'Nausea', 'Headache', 'None'], p=[0.3, 0.3, 0.3, 0.1]),
            'expiry_date': (datetime.now() + timedelta(days=np.random.randint(30, 1095))).strftime('%Y-%m-%d')
        }
        
        medicines.append(product)
    
    return medicines

# Modified Product Data Generation
def generate_products(n_products=500):
    n_per_category = n_products // 4
    
    electronics = generate_electronics(n_per_category)
    groceries = generate_groceries(n_per_category)
    fashion = generate_fashion(n_per_category)
    medicines = generate_medicines(n_per_category)
    
    all_products = electronics + groceries + fashion + medicines
    np.random.shuffle(all_products)
    
    return pd.DataFrame(all_products)

# Generate Order History (modified to use new product IDs)
def generate_orders(users, products, n_orders=5000):
    orders = []
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    for _ in range(n_orders):
        user = np.random.choice(users['user_id'])
        product = np.random.choice(products['product_id'])
        order_date = start_date + timedelta(days=np.random.randint(0, (end_date - start_date).days))
        orders.append({
            'order_id': f'O{_+1:04d}',
            'user_id': user,
            'product_id': product,
            'order_date': order_date,
            'quantity': np.random.randint(1, 5)
        })
    return pd.DataFrame(orders)

# Generate Inventory Data (modified to use new product IDs)
def generate_inventory(products, warehouse_areas):
    inventory = []
    for _, product in products.iterrows():
        # Get all warehouses
        warehouses = warehouse_areas['warehouse'].unique()
        # Randomly choose 1-3 warehouses for this product
        product_warehouses = np.random.choice(warehouses, size=np.random.randint(1, 4), replace=False)
        for warehouse in product_warehouses:
            # Get cities served by this warehouse
            served_cities = warehouse_areas[warehouse_areas['warehouse'] == warehouse]['city'].unique()
            # Randomly choose 1-2 cities for this product-warehouse combination
            product_cities = np.random.choice(served_cities, size=np.random.randint(1, 2), replace=False)
            for city in product_cities:
                inventory.append({
                    'product_id': product['product_id'],
                    'warehouse': warehouse,
                    'city': city,
                    'stock': np.random.randint(0, 100)
                })
    return pd.DataFrame(inventory)

# Generate Warehouse Service Areas (unchanged)
def generate_warehouse_areas():
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
    warehouses = ['WH1', 'WH2', 'WH3', 'WH4']
    warehouse_areas = []
    
    for warehouse in warehouses:
        served_cities = np.random.choice(cities, size=np.random.randint(1, 4), replace=False)
        for city in served_cities:
            warehouse_areas.append({
                'warehouse': warehouse,
                'city': city
            })
    
    return pd.DataFrame(warehouse_areas)

# Main function to generate the dataset
def generate_dataset():
    users_df = generate_users()
    products_df = generate_products()
    warehouse_areas_df = generate_warehouse_areas()
    orders_df = generate_orders(users_df, products_df)
    inventory_df = generate_inventory(products_df, warehouse_areas_df)

    # Save to CSV files
    users_df.to_csv('users.csv', index=False)
    products_df.to_csv('products.csv', index=False)
    orders_df.to_csv('orders.csv', index=False)
    inventory_df.to_csv('inventory.csv', index=False)
    warehouse_areas_df.to_csv('warehouse_areas.csv', index=False)

    print("Dataset files have been created: users.csv, products.df, orders.csv, inventory.csv, warehouse_areas.csv")

    # Display sample data
    print("\nUsers Sample:")
    print(users_df.head(10))
    print("\nWarehouse Service Areas Sample:")
    print(warehouse_areas_df.head(10))
    print("\nInventory Sample:")
    print(inventory_df.head(10))

if __name__ == "__main__":
    generate_dataset()