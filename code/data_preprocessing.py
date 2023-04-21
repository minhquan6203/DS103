from datetime import datetime
from sklearn.experimental import enable_iterative_imputer  # needed to load iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from scipy.stats import ks_2samp

import pandas as pd
import numpy as np
import pickle
import re


def merge_gender_cols(data):
    """
    Combine multiple gender columns into a single one
    :param data: data frame containing original gender columms
    :return: row-wise value of the new gender column
    """
    if (data['gioiTinh'].lower() == data['info_social_sex']) & (data['gioiTinh'] != 'nan') & \
            (data['info_social_sex'] != 'nan'):
        val = data['info_social_sex']
    elif (data['gioiTinh'] == 'nan') & (data['info_social_sex'] != 'nan'):
        val = data['info_social_sex']
    elif (data['gioiTinh'] != 'nan') & (data['info_social_sex'] == 'nan'):
        val = data['gioiTinh'].lower()
    elif data['gioiTinh'].lower() != data['info_social_sex']:
        val = 'other'
    else:
        val = 'missing'
    return val


def process_gender(data):
    """
    Convert original gender columns to type string and merge them into a single variable
    :param data: data frame containing original gender columms
    :return: data frame with the new combined gender column
    """
    data = (
        data
        .assign(gioiTinh=lambda x: x['gioiTinh'].astype(str),
                info_social_sex=lambda x: x['info_social_sex'].astype(str),
                gender=lambda x: x.apply(merge_gender_cols, axis=1))
    )
    return data


def create_date_diff_col(data, start_col, end_col):
    """
    Create column indicating number of days in between a start-date column and an end-date column
    :param data: original data frame
    :param start_col: start-date column
    :param end_col: end-date column
    :return: days difference column
    """
    return (pd.to_datetime(data[end_col]) - pd.to_datetime(data[start_col])).dt.days


def process_date_diffs(data):
    """
    Apply create_date_diff_col onto all start-date/end-date column combinations in original data
    :param data: original data frame
    :return: data frame with newly created date difference columns
    """
    data = (
        data
        .assign(F_date_diff=lambda x: create_date_diff_col(x, 'F_startDate', 'F_endDate'),
                E_date_diff=lambda x: create_date_diff_col(x, 'E_startDate', 'E_endDate'),
                C_date_diff=lambda x: create_date_diff_col(x, 'C_startDate', 'C_endDate'),
                G_date_diff=lambda x: create_date_diff_col(x, 'G_startDate', 'G_endDate'),
                A_date_diff=lambda x: create_date_diff_col(x, 'A_startDate', 'A_endDate'))
    )
    return data


def process_date_cols(data, cols):
    """
    Extract month and year values of all date and datetime columns
    :param data: original data frame
    :param cols: columns of type date or datetime
    :return: data frame with processed date columns
    """
    for col in cols:
        if col == 'Field_34':
            data[col + '_year'] = data[col].str[:4].astype('category')
            data[col + '_month'] = data[col].str[4:6].astype(float).astype('category')
        else:
            tmp = pd.to_datetime(data[col])
            data[col + '_year'] = tmp.dt.year.astype('category')
            data[col + '_month'] = tmp.dt.month.astype('category')
        data = data.drop(col, axis=1)
    return data


def process_age(data):
    """
    Create current age variable based on date of birth column
    :param data: original data frame
    :return: data frame with age column
    """
    data['current_age'] = datetime.today().year - data['ngaySinh'].astype(str).str[:4].replace('nan', np.nan).astype(float)
    return data


def categorize_job(string):
    """
    Group job titles in Vietnamese into various job groups in English
    :param string: job title in Vietnamese
    :return: corresponding job group in English
    """
    if type(string) == str:
        blue_collar_matches = ['công nhân', 'cnv', 'cn', 'may công nghiệp', 'lao động', 'thợ', 'coõng nhaõn', 'c.n',
                               'lđ', 'bảo vệ', 'phụ kho', 'phục vụ', 'cán sự', 'vệ sỹ', 'phết keo', 'phụ việc', 'tạp vụ',
                               'gia công', 'cao su', 'kho', 'may', 'công', 'sửa', 'cong nhan', 'cảnh vệ', 'ép cao tần',
                               'đóng gói', 'giao nhận']
        teacher_matches = ['giáo viên', 'gv', 'gíao viên', 'giáo', 'giảng', 'bảo mẫu', 'cô nuôi']
        white_collar_matches = ['nhân viên', 'kế toán', 'cán bộ', 'nv', 'cb', 'nhõn viờn', 'giao dịch viên', 'operator',
                                'thủ', 'kinh doanh', 'văn thư', 'staff', 'trợ lý', 'bí thư', 'kinh tế', 'văn phòng',
                                'thư ký']
        driver_matches = ['tài xế', 'lái', 'tài xê', 'phụ xe']
        manager_matches = ['quản lý', 'phó phòng', 'hiệu phó', 'giám đốc', 'hiệu trưởng', 'chủ tịch', 'trưởng phòng']
        supervisor_matches = ['giám sát', 'tổ', 'cán sự']
        healthcare_matches = ['bác sĩ', 'dược', 'y sĩ', 'y sỹ', 'hộ sinh', 'y tá', 'dưỡng', 'hộ lý', 'bác sỹ', 'dịch tễ']
        chef_matches = ['bếp']
        engineer_matches = ['sư']
        specialist_matches = ['chuyên viên', 'chuyờn viờn', 'thanh tra', 'phiên dịch', 'kiểm ngân']
        technician_matches = ['kỹ thuật', 'kĩ thuật', 'ktv', 'kiểm soát', 'kỷ thuật', 'lập trình']
        entertainment_matches = ['diễn viên', 'ca sĩ', 'phim', 'chụp', 'entertainment', 'phóng viên']
        travel_matches = ['tiếp viên', 'hướng dẫn viên']
        fitness_matches = ['huấn luyện']
        customer_matches = ['bán', 'thu ngân', 'bán hàng', 'lễ tân', 'cửa hàng', 'tiếp thị', 'chăm sóc', 'tư vấn',
                            'cửa hàng']
        missing_matches = ['undefined', 'thieu chuc danh']

        if any(x in string for x in blue_collar_matches):
            return 'blue_collar'
        elif any(x in string for x in teacher_matches):
            return 'teacher'
        elif any(x in string for x in white_collar_matches):
            return 'white_collar'
        elif any(x in string for x in driver_matches):
            return 'driver'
        elif any(x in string for x in manager_matches):
            return 'manager'
        elif any(x in string for x in supervisor_matches):
            return 'supervisor'
        elif any(x in string for x in healthcare_matches):
            return 'healthcare_worker'
        elif any(x in string for x in chef_matches):
            return 'chef'
        elif any(x in string for x in engineer_matches):
            return 'engineer'
        elif any(x in string for x in specialist_matches):
            return 'specialist'
        elif any(x in string for x in technician_matches):
            return 'technician'
        elif any(x in string for x in entertainment_matches):
            return 'entertainment'
        elif any(x in string for x in travel_matches):
            return 'travel_worker'
        elif any(x in string for x in fitness_matches):
            return 'fitness'
        elif any(x in string for x in customer_matches):
            return 'customer_matches'
        elif any(x in string for x in missing_matches):
            return 'missing'
        else:
            return 'other'
    else:
        return string


def normalize_str(s):
    """
    Strip whitespace, lowercase and remove unnecessary characters from strings
    :param s: original string
    :return: normalized string
    """
    s = str(s).strip().lower()
    s = re.sub(' +', " ", s)
    return s


def process_macv(data):
    """
    Process the job title column into a cleaner version of itself with fewer unique levels
    :param data: original data frame
    :return: data frame with new variable indicating job groups
    """
    data['job'] = data['maCv'].apply(normalize_str).apply(categorize_job).astype('category')
    data = data.drop('maCv', axis=1)
    return data


def categorize_location(x):
    """
    Categorize location variable into either Vietnam or International
    :param x: original location value
    :return: new location value
    """
    if x != 'nan':
        if 'vietnam' in x:
            return 'VN'
        elif 'missing' in x:
            return 'missing'
        else:
            return 'International'


def process_country(data):
    """
    Process all location-related columns at country-level
    :param data: original data frame
    :return: data frame with processed location columns
    """
    data['homeTownCountry'] = data['homeTownCountry'].apply(normalize_str).apply(categorize_location).astype('category')
    data['currentLocationCountry'] = data['currentLocationCountry'].apply(normalize_str).apply(categorize_location).astype('category')
    return data


def categorize_welfare(string):
    """
    Extract welfare information from Field_46 and Field_61
    :param string: string value from Field_46 and Field_61
    :return: binary value
    """
    if type(string) == str:
        matches = ['thất nghiệp', 'dịch vụ việc làm', 'bhtn', 'trợ cấp', 'nghèo', 'khó khăn', 'Bảo trợ xã hội',
                   'Ốm đau dài']
        if any(x in string for x in matches):
            return 1
        return 0


def process_welfare(data):
    """
    Create binary variables to indicate welfare recipients
    :param data: original data frame
    :return: data frame with new columns
    """
    data['Field_46_welfare'] = data['Field_46'].apply(normalize_str).apply(categorize_welfare)
    data['Field_61_welfare'] = data['Field_61'].apply(normalize_str).apply(categorize_welfare)
    return data


def create_missing_indicators(data, cols):
    """
    Create missing indicators for categorical variables
    :param data: original data frame
    :param cols: categorical columns
    :return: data frame with new missing indicators
    """
    missing_inds = data[cols].apply(lambda x: np.where(x.notnull(), 1, 0), axis=0)
    new_col_names = cols + '_missing'
    data = data.copy()
    data[new_col_names] = missing_inds
    return data


def replace_missing_level(data, cols):
    """
    Replace NA values in categorical columns with 'missing' level
    :param data: original data frame
    :param cols: categorical columns
    :return: new data frame
    """
    new_cols = data[cols].apply(lambda x: np.where(x.isnull(), 'missing', x), axis=0)
    data = data.copy()
    data[cols] = new_cols
    return data


def drop_correlated_cols(data, thres):
    """
    Drop columns whose correlation with another is higher than a certain threshold
    :param data: original data frame
    :param thres: threshold
    :return: new data frame with correlated colums dropped
    """
    corr_mat = data.drop('label', axis=1).corr().abs()
    upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(np.bool))  # select only upper triangle
    to_drop_cols = [column for column in upper.columns if any(upper[column] > thres)]
    data = data.copy().drop(to_drop_cols, axis=1)
    return data


def drop_missing_rows(data, thres):
    """
    Drop rows whose missing percentage is greater than a certain threshold
    :param data: original data frame
    :param thres: threshold
    :return: new data frame with highly missing rows dropped
    """
    data = data.copy()
    data = (
        data
        .assign(row_missing_perc=lambda x: x.drop('label', axis=1).isnull().mean(axis=1))
        .query('row_missing_perc < {}'.format(thres))
        .drop('row_missing_perc', axis=1)
    )
    return data


def create_impute_estimator(data):
    """
    Train an iterative imputer based on extra trees regression (similar to missForest package in R)
    :param data: training data
    :return: impute estimator
    """
    tree_imputer = IterativeImputer(random_state=42, add_indicator=True,
                                    estimator=ExtraTreesRegressor(n_estimators=10, random_state=42))
    tree_imputer.fit(data)
    pickle.dump(tree_imputer, open('data/intermediate/tree_imputer.pkl', 'wb'))
    return tree_imputer


def impute_num_cols_parent(data):
    """
    Apply the trained impute estimator on all numerical columns in training set.
    At the same time, save names of columns being imputed in a global dictionary as well as their new column names
    for corresponding missing indicators so that they can be re-used when we apply the impute estimator on the test set.
    :param data: training data
    :return: imputed training data
    """
    num_cols = data.select_dtypes('number').drop('label', axis=1)
    num_cols_full = num_cols[num_cols.columns[num_cols.isnull().mean() == 0]]
    imputed_colnames = num_cols.columns
    imputed_colnames_missing = [i + '_missing' for i in imputed_colnames if i not in num_cols_full.columns]
    all_new_colnames = imputed_colnames.tolist() + imputed_colnames_missing

    global ctx
    ctx = {'cols_to_impute': imputed_colnames, 'new_col_names': all_new_colnames}

    # tree_imputer = create_impute_estimator(num_cols)
    tree_imputer = pickle.load(open('data/intermediate/tree_imputer.pkl', 'rb'))
    imputed_cols = pd.DataFrame(tree_imputer.transform(num_cols), columns=all_new_colnames)
    data = data.copy()
    data = pd.concat([data.drop(num_cols, axis=1), imputed_cols], axis=1)
    return data


def impute_num_cols_child(data):
    """
    Apply the trained impute estimator on numerical columns in test set. Note that we will only impute
    columns which exist in the training set (thanks to ctx global dictionary).
    :param data: test data
    :return: imputed test data
    """
    tree_imputer = pickle.load(open('data/intermediate/tree_imputer.pkl', 'rb'))
    num_cols = data[ctx['cols_to_impute']]
    all_new_colnames = ctx['new_col_names']
    imputed_cols = pd.DataFrame(tree_imputer.transform(num_cols), columns=all_new_colnames)
    data = data.copy()
    data = pd.concat([data.drop(num_cols, axis=1), imputed_cols], axis=1)
    return data


def dummy_encode(data, columns_arg=None):
    """
    One-hot encode categorical variables in data set. At the same time, save names of columns that are encoded in the
    training set in a global dictionary, so that we will only encode these same columns in the test set.
    :param data: original data frame
    :param columns_arg: if None, encode all existing categorical variables; otherwise, encode only the list of columns
                        supplied.
    :return: data frame with new dummies variables
    """
    if columns_arg is None:
        global cte
        cte = data.select_dtypes(include=['object', 'category']).columns

    return pd.get_dummies(data, columns=columns_arg, drop_first=True)


if __name__ == '__main__':
    train = pd.read_csv('data/raw/train.csv', low_memory=False)
    test = pd.read_csv('data/raw/test.csv', low_memory=False)

    missing_cols = [i for i in train.columns if train[i].isna().mean() != 0]
    meta_cols = ['id']
    location_cols = ['currentLocationLocationId', 'currentLocationLatitude', 'currentLocationLongitude',
                     'homeTownLocationId', 'homeTownLatitude', 'homeTownLongitude', 'data.basic_info.locale',
                     'homeTownName', 'currentLocationName', 'diaChi', 'currentLocationCity', 'currentLocationName',
                     'homeTownCity', 'currentLocationState', 'homeTownState']
    one_uniq_cols = train.columns[train.nunique().values == 1].values  # columns with 1 unique value (no variance)
    multi_uniq_cols = ['Field_4', 'Field_12', 'Field_36', 'Field_47', 'Field_54', 'Field_55', 'Field_62', 'Field_65',
                       'Field_66', 'data.basic_info.locale', 'currentLocationCity', 'currentLocationCountry',
                       'currentLocationName', 'currentLocationState', 'homeTownCity', 'homeTownCountry', 'homeTownName',
                       'homeTownState', 'brief']  # categorical columns with more than 1 unique level
    date_cols = ['Field_' + str(i) for i in [1, 2, 5, 6, 7, 8, 9, 11, 15, 25, 32, 33, 34, 35, 40, 43, 44]]
    new_year_cols = ['Field_' + str(i) + '_year' for i in [1, 2, 5, 6, 7, 8, 9, 11, 15, 25, 32, 33, 34, 35, 40, 43, 44]]
    new_month_cols = ['Field_' + str(i) + '_month' for i in [1, 2, 5, 6, 7, 8, 9, 11, 15, 25, 32, 33, 34, 35, 40, 43, 44]]
    processed_cols = ['ngaySinh', 'namSinh', 'gioiTinh', 'info_social_sex', 'F_startDate', 'F_endDate', 'E_startDate',
                      'E_endDate', 'C_startDate', 'C_endDate', 'G_startDate', 'G_endDate', 'A_startDate', 'A_endDate',
                      'Field_46', 'Field_61']
    trivial_cols = ['Field_' + str(i) for i in [18, 38, 45, 48, 49, 54, 55, 56, 65, 68]]  # cryptic columns with mixed data types and >10k levels

    train1 = (
        train
        .pipe(drop_missing_rows, 0.90)  # drop 4730 rows
        .pipe(process_gender)
        .pipe(process_age)
        .pipe(process_date_diffs)
        .pipe(process_date_cols, date_cols)
        .pipe(process_macv)
        .pipe(process_country)
        .pipe(process_welfare)
        .pipe(create_missing_indicators, one_uniq_cols)  # add 45 cols
        .pipe(replace_missing_level, multi_uniq_cols + new_year_cols + new_month_cols)
        .drop(columns=meta_cols + location_cols + processed_cols + one_uniq_cols.tolist() + trivial_cols, axis=1)  # drop 70 cols
        .pipe(drop_correlated_cols, 0.90)  # drop 61 cols
        .reset_index(drop=True)  # save the data here to use for `src/explore_imputation.py` script
        .pipe(impute_num_cols_parent)
        .pipe(dummy_encode)
        .reset_index(drop=True)
    )
    train1.to_pickle('data/intermediate/train1.pkl')

    ks_stats, p_vals = [], []
    for i in train1.columns:
        stat, pval = ks_2samp(train1.loc[train1['label'] == 0, i], train1.loc[train1['label'] == 1, i])
        ks_stats.append(stat)
        p_vals.append(pval)
    ks_res = (pd.DataFrame({'column': train1.columns, 'statistic': ks_stats, 'pvalue': p_vals})
                .sort_values(by=['statistic'], ascending=False))
    fs_sel = ks_res.loc[ks_res['pvalue'] <= 0.001, 'column'].tolist()

    train2 = train1.filter(fs_sel)
    train2.to_pickle('data/intermediate/train2.pkl')

    test1 = (
        test
        .pipe(process_gender)
        .pipe(process_age)
        .pipe(process_date_diffs)
        .pipe(process_date_cols, date_cols)
        .pipe(process_macv)
        .pipe(process_country)
        .pipe(process_welfare)
        .pipe(create_missing_indicators, one_uniq_cols)
        .pipe(replace_missing_level, multi_uniq_cols + new_year_cols + new_month_cols)
        .reset_index(drop=True)
        .pipe(impute_num_cols_child)
        .pipe(dummy_encode, cte)
        .filter(fs_sel)
    )
    test1.to_pickle('data/intermediate/test1.pkl')