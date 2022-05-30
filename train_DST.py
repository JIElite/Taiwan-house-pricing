import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import VALID_METRICS
from sklearn.tree import DecisionTreeRegressor

from eval import simple_evaluate
from utils import split_features_target


def clean_and_drop(df):
    # 只篩選有包含 '住' 用途的交易案
    df = df.loc[df['Main_Usage_Living'] == 1]
    df = df.drop(columns=['Main_Usage_Living'])
    
    # 因為都是 0
    df = df.drop(columns=['Non_City_Land_Usage', 'Main_Usage_Walk', 
                          'Main_Usage_Selling',
                          'Main_Usage_SnE'])
    
    # 只有 344 筆是包含工廠用途，且都不具住宅用途，故剔除
    df = df.loc[df['Main_Usage_Manufacturing'] == 0]
    df = df.drop(columns=['Main_Usage_Manufacturing'])
    
    # 只有 76 筆是包含停車用途，且都不具住宅用途，故剔除
    df = df.loc[df['Main_Usage_Parking'] == 0]
    df = df.drop(columns=['Main_Usage_Parking'])
    
    # 只有 78 筆有農業用途，且都不具住宅用途，故剔除
    df = df.loc[df['Main_Usage_Farm'] == 0]
    df = df.drop(columns=['Main_Usage_Farm'])
    
    # NOTICE: 我沒有錢，所以我先只買 6 房以下的
    df = df.loc[df['room'] < 6]
    
    df = df.loc[df['trading_floors_count'] == 1]
    
    # 雖然有 95 個樣本包含地下室，但是樣本太少，可能不足以推廣
    # 所以先剔除，剔除完後，都是 0 所以直接 drop
    df = df.loc[df['including_basement'] == 0]
    df = df.drop(columns=['including_basement'])
    
    # 所有的樣本都不包含人行道，所以直接去除這個 feature
    df = df.drop(columns=['including_arcade'])

    # 剔除交易樓層高度是 -1 (原本有一個樣本)
    df = df.loc[df['min_floors_height'] != -1]

    # 剔除交易建物是 0 個樓層的情況
    df = df.loc[df['building_total_floors'] != 0]
    
    # 因為車位交易 50 坪以上的資料只有 22 筆，所以先去除
    # 因為浮點數在硬體儲存會有小數點，故不能直接用 == 50.0 去比較
    df = df.loc[df['Parking_Area'] < 49.5]
    
    # 把農舍，廠辦踢掉
    df = df.loc[df['Building_Types'] < 8]

    # 把超大轉移坪數刪掉
    df = df.loc[df['Transfer_Total_Ping'] < 150]
    
    # 我先刪除 area_m2, 因為覺得跟 area_ping 的意義很類似，但是不確定會不會有些微差距。
    # 因為在 future data 中，manager 都是 0，所以也把這個欄位刪除
    # trading_floor_count 有 0 的情況，這樣應該不是房屋交易
    df = df.drop(columns=['address', 'area_m2', 'manager', 'Building_Material_stone', 
                     'TDATE', 'Total_price', '編號'])
    
    # Convert the categorical features' dtype to 'category'
    category_columns = ['Type', 'Month', 'Month_raw',
                       'City_Land_Usage', 'Main_Usage_Business',
                       'Building_Material_S', 'Building_Material_R', 'Building_Material_C',
                       'Building_Material_steel', 'Building_Material_B', 
                       'Building_Material_W', 'Building_Material_iron',
                       'Building_Material_tile', 'Building_Material_clay',
                       'Building_Material_RC_reinforce',
                       'Parking_Space_Types', 'Building_Types']
    df.loc[:, category_columns] = df.loc[:, category_columns].astype('category')
    return df



TRAIN_DATA_PATH = './merged_data/clean_data_future_train.csv'
TEST_DATA_PATH = './merged_data/clean_data_future_test.csv'
VAL_SIZE = 0.1
MAX_DEPTH = 20

df_future = pd.read_csv(TRAIN_DATA_PATH)
df_future_test = pd.read_csv(TEST_DATA_PATH)

df_future = clean_and_drop(df_future)
df_future = df_future.sample(frac=1, random_state=0)
X_train, y_train = split_features_target(df_future)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VAL_SIZE)

df_future_test = clean_and_drop(df_future_test)
X_test, y_test = split_features_target(df_future_test)

model = DecisionTreeRegressor(max_depth=MAX_DEPTH)
model.fit(X_train, y_train)

print('Training performance: ')
simple_evaluate(model, X_train, y_train, verbose=True)
print()
print('Evaluation performance: ')
simple_evaluate(model, X_val, y_val, verbose=True)
print()
print('Test performance:')
simple_evaluate(model, X_test, y_test, verbose=True)