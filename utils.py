DEPENDENT_TARGETS = ['Total_price', 'Unit_Price_Ping']


def split_features_target(df, target_field='Unit_Price_Ping'):
    X = df.drop(columns=DEPENDENT_TARGETS)
    y = df[target_field]
    return X, y
