import pandas as pd
import numpy as np

def clean_price(price_str):
    if pd.isna(price_str):
        return np.nan
    cleaned = str(price_str).replace(',', '').replace('$', '').strip()
    try:
        return float(cleaned)
    except:
        return np.nan

def clean_reviews(review_str):
    if pd.isna(review_str):
        return 0
    cleaned = str(review_str).replace(',', '').strip()
    try:
        return int(cleaned)
    except:
        return 0

def load_and_clean_data():
    print("Loading mobile data...")
    mobile = pd.read_csv('data/amazon_mobile.csv', encoding='latin-1')
    mobile['category'] = 'mobile'
    
    print("Loading laptop data...")
    laptop = pd.read_csv('data/amazon_laptop.csv', encoding='latin-1')
    laptop['category'] = 'laptop'
    
    mobile.columns = mobile.columns.str.strip()
    laptop.columns = laptop.columns.str.strip()
    
    mobile = mobile.rename(columns={
        'Product Description': 'title',
        'Product Description ': 'title',
        'Price(Dollar)': 'price',
        'Number of reviews': 'reviews',
    })
    
    laptop = laptop.rename(columns={
        'Product Description': 'title',
        'Product Description ': 'title',
        'Price(Dollar)': 'price',
        'Number of  reviews': 'reviews',
    })
    
    mobile = mobile[['title', 'price', 'reviews', 'category']]
    laptop = laptop[['title', 'price', 'reviews', 'category']]
    
    df = pd.concat([mobile, laptop], ignore_index=True)
    
    print("Cleaning data...")
    df['price'] = df['price'].apply(clean_price)
    df['reviews'] = df['reviews'].apply(clean_reviews)
    df = df[df['title'].notna()].copy()
    
    print("\nTotal products:", len(df))
    print("Mobile:", len(df[df['category']=='mobile']))
    print("Laptop:", len(df[df['category']=='laptop']))
    
    return df

if __name__ == "__main__":
    df = load_and_clean_data()
    df.to_csv('data/cleaned_data.csv', index=False)
    print("\n✅ Data preprocessing complete!")
    print("Saved to: data/cleaned_data.csv")
