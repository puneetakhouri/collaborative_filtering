import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
users = pd.read_csv('../dataset/Users.csv')
books = pd.read_csv('../dataset/Books.csv')
ratings = pd.read_csv('../dataset/Ratings.csv')

# Get dataset info
users.info()
books.info()
ratings.info()

# Drop rows with duplicate book title
new_books = books.drop_duplicates('Book-Title')

# Merge ratings and new_books df
ratings_with_name = ratings.merge(new_books, on='ISBN')

# Drop non-relevant columns
ratings_with_name.drop(['ISBN', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis = 1, inplace = True)

# Merge new 'ratings_with_name' df with users df
users_ratings_matrix = ratings_with_name.merge(users, on='User-ID')

# Drop non-relevant columns
users_ratings_matrix.drop(['Location', 'Age'], axis = 1, inplace = True)

# Print the first few rows of the new dataframe
users_ratings_matrix.head()

# Check for null values
users_ratings_matrix.isna().sum()
# Drop null values
users_ratings_matrix.dropna(inplace = True)
print(users_ratings_matrix.isna().sum())

# Filter down 'users_ratings_matrix' on the basis of users who gave many book ratings
x = users_ratings_matrix.groupby('User-ID').count()['Book-Rating'] > 100
knowledgeable_users = x[x].index
filtered_users_ratings = users_ratings_matrix[users_ratings_matrix['User-ID'].isin(knowledgeable_users)]

# Filter down 'users_ratings_matrix' on the basis of books with most ratings
y = filtered_users_ratings.groupby('Book-Title').count()['Book-Rating'] >= 50
famous_books = y[y].index
final_users_ratings = filtered_users_ratings[filtered_users_ratings['Book-Title'].isin(famous_books)]

# Pivot table creation
pivot_table = final_users_ratings.pivot_table(index = 'Book-Title', columns = 'User-ID', values = 'Book-Rating')

# Filling the NA values with '0'
pivot_table.fillna(0, inplace = True)
pivot_table.head()

# Standardize the pivot table
scaler = StandardScaler(with_mean=True, with_std=True)
pivot_table_normalized = scaler.fit_transform(pivot_table)

# Calculate the similarity matrix for all the books
similarity_score = cosine_similarity(pivot_table_normalized)

def recommend(book_name):
    
    # Returns the numerical index for the book_name
    index = np.where(pivot_table.index==book_name)[0][0]
    
    # Sorts the similarities for the book_name in descending order
    similar_books = sorted(list(enumerate(similarity_score[index])),key=lambda x:x[1], reverse=True)[1:6]
    
    # To return result in list format
    data = []
    
    for index,similarity in similar_books:
        item = []
        # Get the book details by index
        temp_df = new_books[new_books['Book-Title'] == pivot_table.index[index]]
        
        # Only add the title, author, and image-url to the result
        item.extend(temp_df['Book-Title'].values)
        item.extend(temp_df['Book-Author'].values)
        item.extend(temp_df['Image-URL-M'].values)
        
        data.append(item)
    return data

if __name__ == "__main__":
    # Example book name to get recommendations
    book_name = "River's End"
    recommendations = recommend(book_name)
    
    # Print the recommended books
    for rec in recommendations:
        print(f"Book: {rec[0]}, Author: {rec[1]}, Image URL: {rec[2]}")
