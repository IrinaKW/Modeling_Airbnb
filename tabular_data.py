#%%
import pandas as pd

def remove_rows_with_missing_ratings(df):
    rating_col_lst=[col for col in df.columns if 'rating' in col]
    df.dropna(subset=rating_col_lst, inplace=True)
    return df

def combine_description_strings(df):
    rep_chars = '\*|\]|\['
    df.dropna(subset=['Description'], inplace=True)
    df['Description']=df['Description'].str.replace(rep_chars,"")
    df['Description'].replace([r"\\n", "\n", r"\'"], [" "," ",""], regex=True, inplace=True)
    df['Description']=df['Description'].str.strip('About this space, ')
    return df

def set_default_feature_values(df):
    col_list=["guests", "beds", "bathrooms", "bedrooms"]
    df[col_list] = df[col_list].fillna(1)
    return df

def clean_tabular_data(df):
    df=set_default_feature_values(df)
    df=remove_rows_with_missing_ratings(df)
    df=combine_description_strings(df)
    return df


if __name__ == "__main__":
    path='tabular_data/listing.csv'
    listing_df=pd.read_csv(path)
    listing_df.drop('Unnamed: 19', axis=1, inplace=True)
    clean_listing_df=clean_tabular_data(listing_df)
    clean_listing_df.to_csv('tabular_data/clean_tabular_data.csv', index=False)

# %%
