import pandas as pd


def clean_invalid(dataframe):
    """ Clean invalid data

    *** This is a prerequisite of the following code cells ***
    We transform the features based on the clean data
    -------------------------------------------------------
    Question: 是不是因為清完資料，都市住宅分區的 unique 才會變少？Yes
    """
    dataframe = dataframe[(dataframe['交易標的']!='土地') & (~dataframe['交易標的'].isna())]
    dataframe['Month'] = dataframe['交易年月日'].str[:-2].astype('float')
    dataframe = dataframe.query("Month>=10601 and Month<=11103 ")
    dataframe = dataframe[(dataframe.Month!=10600) & (dataframe.Month!=10700) & \
                          (dataframe.Month!=10800) & (dataframe.Month!=10900) & (dataframe.Month!=11000)]
    
    return dataframe


def show_num_unique_vals(df):
    print('Number of unique values for each column:')
    for column in df.columns:
        print(f'- {column}: {len(df[column].unique())}')


def read_clean_csv(path, verbose=True, **kwargs):
    # Read CSV
    df = pd.read_csv(path, **kwargs)
    if verbose:
        print('Number of rows in raw data:', len(df))
        show_num_unique_vals(df)
        
    # Clean invalid data
    df = clean_invalid(df)
    if verbose:
        print('Number of row in cleaned data:', len(df))
        show_num_unique_vals(df)
        
    return df


def find_substring_and_propotion(dataframe, column, pattern, verbose=True):
    """List the unique content contains specific pattern in a given column.
    
    Example:
        column = '主要建材'
        substring = '混凝土'
        find_substring_and_propotion(sale_data, column, substring)
        
        鋼筋混凝土造
        鋼骨鋼筋混凝土造
        鋼筋混凝土加強磚造
        鋼筋混凝土構造
        鋼骨混凝土造
        鋼筋混凝土加強空心磚造
        ...
        ...
        Number of observations in the dataframe: 900651
        The propotion: 83.85%
    """
    for elem in dataframe[column].unique():
        # ignore nan
        if not isinstance(elem, str):
            continue
        if pattern in elem:
            print(elem)
        
    sample_count = len([data for data in dataframe[column] \
                        if isinstance(data, str) and pattern in data])
    total_sample = len(dataframe)
    propotion = sample_count / total_sample  
    if verbose:
        print('Number of observations in the dataframe:', sample_count)
        print(f'The propotion: {propotion * 100:.2f}%')
        
    return sample_count, propotion*100