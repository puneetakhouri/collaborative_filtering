import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# url - https://www.geeksforgeeks.org/e-commerce-product-recommendations-using-catboost/?ref=ml_lbp

# Load the dataset
url = "../data/Online-Retail.xlsx"
df = pd.read_excel(url)

print("Display the first few rows of the dataset")
# Display the first few rows of the dataset
print(df.head())
#print(df.count())

print("Drop rows with missing values")
# Drop rows with missing values
df.dropna(inplace=True)

print("Convert InvoiceDate to datetime")
# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

print("Create a new column for total price")
# Create a new column for total price
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

print("Filter out negative quantities")
# Filter out negative quantities
df = df[df['Quantity'] > 0]

print("Display the cleaned dataset")
# Display the cleaned dataset
print(df.head())

# Define reference date as one day after the last invoice date
reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# Group by CustomerID to calculate RFM
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',  # Frequency
    'TotalPrice': 'sum'  # Monetary
})

# Rename columns
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Display the RFM dataframe
print(rfm.head())

# Split the data
X = rfm[['Recency', 'Frequency', 'Monetary']]
y = np.where(rfm['Monetary'] > rfm['Monetary'].median(), 1, 0)  # Target: 1 if above median, else 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Display the shapes of the splits")
# Display the shapes of the splits
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Initialize CatBoostClassifier
model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, verbose=False)

# Fit the model
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f'Accuracy: {accuracy:.2f}')

# Get feature importance
feature_importance = model.get_feature_importance()
features = X.columns

# Create a dataframe for visualization
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.show()