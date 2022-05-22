def encode_city_land_usage(x):
    """Encode the content in column: '都市土地使用分區'

    Encode {nan/非都市: 0, '住': 1, '農': 2, '工': 3, '商': 4,
             '住商': 5, '其他住宅': 6, '其他跟住無關的': 7}
    """

    # nan remains
    if not isinstance(x, str):
        return 0

    if '非都市' in x:
        return 0

    if '其他' in x:
        end_idx = x.find('(') if '(' in x else len(x) + 1
        if '住商' in x[:end_idx] or '住宅商業' in x[:end_idx]:
            return 5
        if '住宅' in x[:end_idx] or '住' in x[:end_idx]:
            return 6
        return 7

    if '住' in x:
        return 1
    elif '農' in x:
        return 2
    elif '工' in x:
        return 3
    elif '商' in x:
        return 4
    else:
        # There is no such case in the data
        raise ValueError('Unexpected 都市使用分區:', x)


def encode_main_usage(x):
    # nan remains
    if not isinstance(x, str):
        return x
    
    living = lambda x: True if '住' in x else False
    farm_related = lambda x: True if '農舍' in x or '農業' in x or \
                                     '畜牧' in x or '豬舍' in x or \
                                     '雞舍' in x or '羊舍' in x or \
                                     '牛舍' in x or '禽舍' in x or \
                                     '堆肥舍' in x else False
    manufaturing = lambda x: True if '工場' in x or '工廠' in x or '廠房' in x else False
    business = lambda x: True if '事務所' in x or '辦公' in x or '商業' in x else False
    selling = lambda x: True if '店鋪' in x or '商店' in x or \
                                '店舖' in x or '零售' in x or \
                                '店房' in x or '商場' in x or \
                                '百貨' in x else False
    medical = lambda x: True if '醫' in x or '病' in x or '診所' in x else False
    parking = lambda x: True if '停車' in x or '車庫' in x else False
    others = lambda x: True if '金融' in x or '健身' in x else False
    
    # Ignore the comment in the field
    end_idx = x.find('（') if '（' in x else len(x) - 1
    x = x[:end_idx]

    if living(x):     
        if selling(x):
            return 5
        elif business(x):
            return 4
        elif manufaturing(x):
            return 3
        elif farm_related(x):
            return 2
        elif medical(x):
            return 6
        elif others(x):
            return 8
        else:
            return 1
    else:
        if manufaturing(x):
            return 3
        elif business(x):
            return 4
        elif selling(x):
            return 5
        elif farm_related(x):
            return 2
        elif medical(x):
            return 6
        elif parking(x):
            return 7
        else:
            # Cases list
            # mutiple usage:
            # - https://gist.github.com/JIElite/73a870e25390a23a676da3f9a2c86436
            # single usage:
            # - https://gist.github.com/JIElite/329341d1573088e389cbc84f38dd15ac
            return 8


def encode_elevator(x):
    """Transform the elevator field data to one-hot encoding
    
    If there is nan in the '電梯' field, we can use '建物型態' to
    fix the missing value.
    """
    if not isinstance(x['電梯'], str):
        if x['建物型態'] in ['華廈(10層含以下有電梯)', '住宅大樓(11層含以上有電梯)']:
            return 1
        return 0
    
    map_ = {'有': 1, '無': 0}
    return map_[x['電梯']]


def transform_area(area):
    """Convert the unit of the area from square meter to Ping(坪)
    """
    # If the original area is nan, then it remains.
    if not isinstance(area, str):
        return area
    
    return float(round(float(area) / 3.3058))


def init_expanded_dict(prefix, suffices):
    result = {}
    for suffix in suffices:
        result[prefix + suffix] = []
    return result


def expand_encoded_main_usage(df, main_usage_column='主要用途'):
    """Expand the columns of encoded main usage

    This is an in-plance operation
    """
    sidewalk_conds = ['人行', '步道', '走道', '騎樓']
    living_conds = ['住', '民宿']
    selling_conds = ['店舖', '店鋪', '店房', '店房', '商店', '店', '零售', \
                    '商場', '百貨']
    manufaturing_conds = ['廠房', '工業', '工廠', '工場']
    business_conds = ['商業', '辦公', '事務所', '服務業', '工商服務業', '住商']
    parking_conds = ['停車', '車庫']
    sport_and_entertainment_conds = ['運動', '健身', '休閒', '保齡球館', \
                                    '活動室', '交誼廳', '娛樂']
    farm_conds = ['農舍', '農業', '畜牧', '豬舍', '雞舍', '羊舍', '牛舍', \
                '禽舍', '堆肥舍']
    # The following conditions have not been used due to low frequency,
    # and I just take a note here.
    # dorm_conds = ['宿舍']
    # medical_conds = ['醫', '病', '診所']
    # excluded_conds = ['銀行營業廳', '銀行', '金融機構', '保險公司', \
    #                 '保險分支機構', '電影院', \
    #                 '汽車改裝業及汽車修理（甲種汽車修理廠）業', \
    #                 '長期照顧機構〈養護型〉',
    #                 ]
    def contain_usage(raw_usage, cond_list):
        if not isinstance(raw_usage, str):
            return 0
        for cond in cond_list:
            if cond in raw_usage:
                return 1
        return 0

    def encode_main_usage(raw_usage, suffices_cond_dict):
        encoding = {}
        for cond_name, cond_list in suffices_cond_dict.items():
            encoding[cond_name] = contain_usage(raw_usage, cond_list)
        return encoding 


    prefix_ = 'Main_Usage_'
    suffices_cond_dict = {'Walk': sidewalk_conds, 'Living': living_conds,
                          'Selling': selling_conds, 'Manufacturing': manufaturing_conds,
                          'Business': business_conds, 'Parking': parking_conds,
                          'SnE': sport_and_entertainment_conds, 'Farm': farm_conds,
                        }
    expanded_main_usage = init_expanded_dict(prefix=prefix_, \
        suffices=suffices_cond_dict.keys())
    
    for raw_main_usage in df[main_usage_column]:
        encoding = encode_main_usage(raw_main_usage, suffices_cond_dict)
        for suffix_key in suffices_cond_dict.keys():
            category = prefix_ + suffix_key
            expanded_main_usage[category].append(encoding[suffix_key])

    for key in expanded_main_usage.keys():
        df[key] = expanded_main_usage[key]


def init_build_material_cond_dist():
    """
    The building materials could be decomposed into the following categories:
    
    --------RC or SRC -----------------------------------------
    - 鋼筋混凝土造, 鋼筋混凝土構造, 鋼筋混凝土, ＲＣ造, 鋼筋混凝土結構造,
        Ｒ．Ｃ造, "鋼筋混凝土（ＲＣ）", "ＲＣ鋼筋混凝土造", "ＲＣ構架造",
        ＲＣ結構造, "Ｒ．Ｃ構造", "鋼筋混凝土造（ＲＣ）", "R.C造", "ＲＣ構造",
        "Ｒ．Ｃ構架造",  "Ｒ．Ｃ構架", "Ｒ‧Ｃ鋼筋混凝土造", "Ｒ．Ｃ結構", "ＲＣ"
    - "鋼骨ＲＣ造", "ＲＣ鋼骨造", "ＳＲＣ造", "ＳＲＣ"
    - 鋼筋
    - 鋼骨, 鋼骨造, 鋼骨構造, 鋼骨結構造
    - 鋼造, 鋼構造
    - 預力混凝土造 (a kind of RC)
    
    --------Concrete-----------
    - 混凝土
    
    -------- Stone -------------
    - 石造 (跟磚石造撞)
    
    -------- Brick related ----------
    - 磚造 (跟土磚, 木石磚撞, conflict ignored)
    - 加強磚造 
    - 磚木, 磚木造
    - 磚石造, 木石磚造
    - 土磚造, 土磚石混合造
    
    --------- Wood, Earth, and Bamboo ------------------
    - 木造 (跟土木造, 磚木造撞, conflict partially ignored)
    - 竹造 (跟土竹造撞, conflict ignored)
    - 土木造
    - 土造, 土塊造 (跟混凝土造撞)
    
    ---- Iron ----
    - 鐵造 (跟鋼鐵造撞)
    - 鐵架
    - 鐵筋
    - 鐵骨, 鐵骨造
    
    --- Others ----
    - 瓦屋頂, 瓦頂
    - 水泥
    - "ＲＣ補強", "鋼筋補強"
    
    -------- Ignored Materials ---------
    - 土竹造 (ignored, duplicated with 加強磚造)
    - 見其他登記事項, 見其它登記事項, 見使用執照 (ignored)
    """
    S_cond_list = ['鋼骨', 'S', 'Ｓ']
    R_cond_list = ['鋼筋', 'R', 'Ｒ']
    C_cond_list = ['混凝土', 'C', 'Ｃ']
    steel_cond_list = ['鋼造', '鋼構造']
    stone_cond_list = ['石造']
    brick_cond_list = ['磚造', '加強磚造', '磚木', '磚木造',
                       '磚石造', '木石磚造', '土磚造', '土磚石混合造']
    wood_earth_bamboo_list = ['木造', '竹造', '土木造',
                              '土造', '土塊造']
    iron_cond_list = ['鐵造', '鐵架', '鐵筋', '鐵骨']
    tile_roof_list = ['瓦屋頂', '瓦頂']
    clay_cond_list = ['水泥']
    RC_reinforce_list = ['ＲＣ補強', '鋼筋補強']
    
    suffices_cond_dict = {'S': S_cond_list, 'R': R_cond_list,
                     'C': C_cond_list, 'steel': steel_cond_list,
                     'stone': stone_cond_list, 'B': brick_cond_list,
                     'W': wood_earth_bamboo_list, 'iron': iron_cond_list,
                     'tile': tile_roof_list, 'clay': clay_cond_list,
                     'RC_reinforce': RC_reinforce_list}

    return suffices_cond_dict


def encode_materials(raw_material_string, suffices_cond_dict):
    def init_conflict_dict():
        conflict_dict = {'石造': '磚石造', '木造': '磚木造', 
                        '土造':'混凝土造', '鐵造': '鋼鐵造'}
        return conflict_dict

    def is_built_with(raw_material_string, cond_list, conflict_dict=None):
        """Decide whether the raw material string contains specific building materials.

        Example:
            if the raw_material_string is "鋼筋混凝土造", we want to detect the used materials
            contain "鋼筋(R)", "混凝土". However, there is "土造" in the raw string, which conflicts
            with the condition "混凝土造", so we need to handle such condition.
        """
        # nan -> 0
        if not isinstance(raw_material_string, str):
            return 0 
        
        if not conflict_dict:
            conflict_dict = {}

        # Because the values in conflict_dict are all single value,
        # we don't need to use a loop for checking each value with
        # respect to specific conflict key.     
        for cond in cond_list:
            if cond in raw_material_string and cond not in conflict_dict:
                return 1
        return 0

    
    def handle_special_cases(encoding, raw_material_string):
        """Handle the special cases with in-place operation.
        """
        # Special Cases Processing
        RC_cond_list = ['預力混凝土造']
        if is_built_with(raw_material_string, RC_cond_list):
            encoding['R'] = 1
            encoding['C'] = 0

    encoding = {}
    conflict_dict = init_conflict_dict()
    for suffix, suffix_cond_list in suffices_cond_dict.items():
        encoding[suffix] = is_built_with(raw_material_string, suffix_cond_list,\
            conflict_dict)
    
    handle_special_cases(encoding, raw_material_string)

    return encoding


def expand_encoded_building_materials(df, building_material_column='主要建材'):
    """Expand the columns of encoded building materials

    This is an in-plance operation
    """
    # Initialize the sale_data_building_materials like:
    # sale_data_building_materials = {
        # 'Building_Material_S': [],
        # 'Building_Material_R': [],
        # 'Building_Material_C': [],
    #   ...
    # }
    suffices_cond_dict = init_build_material_cond_dist()
    prefix_ = 'Building_Material_'
    sale_data_building_materials = init_expanded_dict(prefix=prefix_,\
        suffices=suffices_cond_dict.keys())
                                                      
    for raw_material in df[building_material_column]:
        encode_result = encode_materials(raw_material, suffices_cond_dict)
        for suffix_key in encode_result.keys():
            column = prefix_ + suffix_key
            sale_data_building_materials[column].append(encode_result[suffix_key])
            
    for key in sale_data_building_materials.keys():
        df[key] = sale_data_building_materials[key]