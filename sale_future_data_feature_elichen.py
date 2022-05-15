# Python version: 3.9
import pandas as pd
from df_helper import read_clean_csv 
from encoding import (encode_city_land_usage, 
                      transform_area,
                      encode_elevator,
                      expand_encoded_main_usage,
                      expand_encoded_building_materials)
pd.set_option('display.max_columns', None)


columns_map = {
        '都市土地使用分區': 'City_Land_Usage',
        '主要用途': 'Main_Usage',
        '車位移轉總面積(平方公尺)': 'Parking_Area',
        '車位移轉總面積平方公尺': 'Parking_Area',
        '主要建材': 'Building_Materials'
    }
features_encoder = {'City_Land_Usage': encode_city_land_usage,
                    'Parking_Area': transform_area}
remaining_cols = list(columns_map.values()) + ['編號']


def transform_df(df, columns_encoding_map):
    for column, encoding_func in columns_encoding_map.items():
        if column in df.columns:
            df[column] = df[column].apply(encoding_func)


def preprocess_features(path, out_path):
    df = read_clean_csv(path, dtype='str')
    df.rename(columns=columns_map, inplace=True)
    df = df.loc[:, df.columns.isin(remaining_cols)]
    transform_df(df, features_encoder)
    expand_encoded_main_usage(df, main_usage_column='Main_Usage')
    expand_encoded_building_materials(df, \
        building_material_column='Building_Materials')
    df.drop(columns=['Main_Usage', 'Building_Materials'], inplace=True)
    df.to_csv(out_path)
    return df

sale_future_data = preprocess_features('./sale_future_data.csv', out_path='./data/sale_future_data_feature_elichen.csv')
print(sale_future_data.head())
