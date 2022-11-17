import pandas as pd
from load_data import download_file_ziplink

def remove_rows_with_missing_ratings(df):
    """removes rows that have missing values in the columns containing 'rating' in their names
    """
    rating_col_lst=[col for col in df.columns if 'rating' in col]
    df.dropna(subset=rating_col_lst, inplace=True)
    return df

def combine_description_strings(df):
    """Description column is being turned into string, 'About this space' is dropped, some symbols are dropped as well.
    """
    rep_chars = '\*|\]|\['
    df.dropna(subset=['Description'], inplace=True)
    df['Description']=df['Description'].str.replace(rep_chars,"")
    df['Description'].replace([r"\\n", "\n", r"\'"], [" "," ",""], regex=True, inplace=True)
    df['Description']=df['Description'].str.strip('About this space, ')
    return df

def set_default_feature_values(df):
    """updates rows that have missing values in the columns guests, beds, bathrooms, bedrooms 
    by replacing NaN with 1.
    """
    col_list=["guests", "beds", "bathrooms", "bedrooms"]
    df[col_list] = df[col_list].fillna(1)
    return df

def clean_tabular_data(df):
    df=set_default_feature_values(df)
    df=remove_rows_with_missing_ratings(df)
    df=combine_description_strings(df)
    return df

def load_airbnb(df,label_col):
    """takes cleaned dataset and turns it into tuple of features (non-object values) and labels
    Args:
        df (dataframe): dataframe of features and labels
        label_col (string): name of the column that is set as labels
    Returns:
        tuple: dataframe of features and labels (series)
    """
    label=df[label_col]
    features=df.drop(label_col, axis=1)
    features=features.select_dtypes(exclude=[object])
    return (features, label)

if __name__ == "__main__":
    link='https://aicore-project-files.s3.eu-west-1.amazonaws.com/airbnb-property-listings.zip'
    file='tabular_data/listing.csv'
    download_file_ziplink(link, file)
    path='tabular_data/listing.csv'
    listing_df=pd.read_csv(path)
    listing_df.drop('Unnamed: 19', axis=1, inplace=True)
    clean_listing_df=clean_tabular_data(listing_df)
    clean_listing_df.to_csv('tabular_data/clean_tabular_data.csv', index=False)


