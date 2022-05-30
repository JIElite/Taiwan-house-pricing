def split_features_target(df):
    X = df.drop(columns=['Unit_Price_Ping'])
    y = df['Unit_Price_Ping']
    return X, y