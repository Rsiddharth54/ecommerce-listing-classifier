import pandas as pd
import numpy as np
import re

def extract_title_features(df):
    print("Extracting title features...")
    
    # Basic length features
    df['title_length'] = df['title'].str.len()
    df['word_count'] = df['title'].str.split().str.len()
    df['avg_word_length'] = df['title'].apply(lambda x: np.mean([len(word) for word in str(x).split()]))
    
    # Character-based features
    df['uppercase_ratio'] = df['title'].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0)
    df['digit_count'] = df['title'].apply(lambda x: sum(c.isdigit() for c in str(x)))
    df['special_char_count'] = df['title'].apply(lambda x: len(re.findall(r'[^a-zA-Z0-9\s]', str(x))))
    
    # Suspicious patterns
    df['has_exclamation'] = df['title'].str.contains('!', na=False).astype(int)
    df['exclamation_count'] = df['title'].apply(lambda x: str(x).count('!'))
    df['has_all_caps_word'] = df['title'].apply(lambda x: any(word.isupper() and len(word) > 2 for word in str(x).split()))
    
    # Keyword stuffing detection
    df['unique_word_ratio'] = df['title'].apply(lambda x: len(set(str(x).lower().split())) / len(str(x).split()) if len(str(x).split()) > 0 else 1)
    
    # Spam keywords
    spam_keywords = ['buy now', 'limited time', 'act fast', 'cheap', 'deal', 'sale']
    df['spam_keyword_count'] = df['title'].apply(lambda x: sum(keyword in str(x).lower() for keyword in spam_keywords))
    
    # Readability
    df['has_proper_capitalization'] = df['title'].apply(lambda x: str(x)[0].isupper() if len(str(x)) > 0 else False).astype(int)
    
    return df

def create_quality_labels(df):
    print("Creating quality labels...")
    
    # Initialize quality score
    df['quality_score'] = 0
    
    # Good signals (add points)
    df.loc[df['title_length'].between(50, 180), 'quality_score'] += 1
    df.loc[df['word_count'].between(8, 25), 'quality_score'] += 1
    df.loc[df['uppercase_ratio'] < 0.3, 'quality_score'] += 1
    df.loc[df['unique_word_ratio'] > 0.7, 'quality_score'] += 1
    df.loc[df['has_proper_capitalization'] == 1, 'quality_score'] += 1
    df.loc[df['reviews'] > 100, 'quality_score'] += 1
    
    # Bad signals (subtract points)
    df.loc[df['exclamation_count'] > 2, 'quality_score'] -= 2
    df.loc[df['has_all_caps_word'] == 1, 'quality_score'] -= 1
    df.loc[df['spam_keyword_count'] > 0, 'quality_score'] -= 1
    df.loc[df['special_char_count'] > 10, 'quality_score'] -= 1
    
    # Create binary labels (High quality: score >= 3)
    df['quality_label'] = (df['quality_score'] >= 3).astype(int)
    
    print("\nQuality distribution:")
    print(df['quality_label'].value_counts())
    print("High quality:", (df['quality_label']==1).sum(), f"({(df['quality_label']==1).sum()/len(df)*100:.1f}%)")
    print("Low quality:", (df['quality_label']==0).sum(), f"({(df['quality_label']==0).sum()/len(df)*100:.1f}%)")
    
    return df

if __name__ == "__main__":
    df = pd.read_csv('data/cleaned_data.csv')
    df = extract_title_features(df)
    df = create_quality_labels(df)
    df.to_csv('data/featured_data.csv', index=False)
    print("\n✅ Feature engineering complete!")
    print("Saved to: data/featured_data.csv")
    print(f"Features created: {len([col for col in df.columns if col not in ['title', 'price', 'reviews', 'category']])}")
