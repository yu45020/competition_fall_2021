import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


def load_data_split(DATA_PATH="data/train_sample.parquet",
                    var_category_type='all',
                    validate_size=0.2,
                    test_size=0.2,
                    random_seed=2,
                    standardize=True,
                    target_col="incwage",
                    rest_set_ages=False

                    ):
    dat = pd.read_parquet(DATA_PATH)  # 318074, 12
    assert dat.incwage.min() > 0 and dat.incwage.max() < 999998
    dat_y = np.log(dat.incwage)
    dat_x = dat.drop(target_col, axis=1)

    dat_x['weight'] = 1.0
    dat_x.loc[dat_x.educ > 6, 'weight'] = 2.0
    dat_x.loc[dat_x.educ == 11, 'weight'] = 3.0

    if var_category_type == 'all':

        var_categorical = ['sex', 'marst', 'race', 'bpl', 'language', 'educ', 'degfield',
                           'vetstat', 'occ'] + ['age', 'uhrswork', 'wkswork1']
        var_numeric = []
        if rest_set_ages:
            dat_x.loc[dat_x.age <= 25, 'age'] = 25
            dat_x.loc[dat_x.age >= 75, 'age'] = 75
        # 11 unique educ values, put more weights on 10 and 11

    elif var_category_type == "partial":
        var_categorical = ['sex', 'marst', 'race', 'bpl', 'language', 'educ', 'degfield',
                           'vetstat', 'occ']
        var_numeric = ['age', 'uhrswork', 'wkswork1']
    else:
        print("var_category_type is either all | partial")
        raise ValueError
    cat_uniqueN = list(dat_x[var_categorical].nunique())

    print(f"Categorical variables {len(var_categorical)}: {', '.join(var_categorical)}")
    print(f"Numeric variables {len(var_numeric)}: {', '.join(var_numeric)}")

    X_train, X_test, y_train, y_test = train_test_split(dat_x, dat_y, test_size=validate_size + test_size,
                                                        random_state=random_seed, shuffle=True)

    X_validate, X_test, y_validate, y_test = train_test_split(X_test, y_test, test_size=test_size,
                                                              random_state=random_seed * 2, shuffle=True)

    ordinal_encoder = OrdinalEncoder().fit(dat_x[var_categorical])

    X_train_cat = ordinal_encoder.transform(X_train[var_categorical])
    X_train_cont = X_train[var_numeric].values

    X_val_cat = ordinal_encoder.transform(X_validate[var_categorical])
    X_val_cont = X_validate[var_numeric].values

    X_test_cat = ordinal_encoder.transform(X_test[var_categorical])
    X_test_cont = X_test[var_numeric].values

    if standardize and len(var_numeric) > 0:
        standardizer = StandardScaler(with_std=True, with_mean=True).fit(X_train_cont)
        X_train_cont = standardizer.transform(X_train_cont)
        X_val_cont = standardizer.transform(X_val_cont)
        X_test_cont = standardizer.transform(X_test_cont)
        print("Continuous variables are standardized to mean 0 and std 1")

    else:
        standardizer = None

    return {
        'train': {'x_cat': X_train_cat, 'x_num': X_train_cont, 'y': y_train.values.reshape(-1, 1),
                  'weight': X_train['weight'].values.reshape(-1, 1)},
        'validate': {'x_cat': X_val_cat, "x_num": X_val_cont, 'y': y_validate.values.reshape(-1, 1),
                     'weight': X_validate['weight'].values.reshape(-1, 1)},
        'test': {'x_cat': X_test_cat, "x_num": X_test_cont, 'y': y_test.values.reshape(-1, 1),
                 'weight': X_test['weight'].values.reshape(-1, 1)},
        'var_cat': var_categorical,
        'var_cat_uniqueN': cat_uniqueN,
        'var_num': var_numeric,
        "var_y": ['incwage'],
        'categorical_encoder': ordinal_encoder,
        'standardizer': standardizer
    }


def combine_pd(dat_dic, dat_type):
    var_cat, var_num = dat_dic['var_cat'], dat_dic['var_num']
    out = np.concatenate([dat_dic[dat_type]['x_cat'], dat_dic[dat_type]['x_num']], axis=-1)
    out = pd.DataFrame(out, columns=var_cat + var_num)
    out[var_cat] = out[var_cat].astype('category')
    out['y'] = dat_dic[dat_type]['y']
    if 'case_id' in dat_dic[dat_type]:
        out['case_id'] = dat_dic[dat_type]['case_id']
    return out


def load_inference_data(DATA_PATH: str,
                        var_cat_list: list,
                        var_num_list: list,
                        encoder_cat: OrdinalEncoder,
                        encoder_num: StandardScaler or None,
                        target_col="incwage",
                        rest_set_ages=False):
    dat = pd.read_parquet(DATA_PATH)
    if rest_set_ages:
        dat.loc[dat.age <= 25, 'age'] = 25
        dat.loc[dat.age >= 75, 'age'] = 75
    dat_x_cat = encoder_cat.transform(dat[var_cat_list])
    if len(var_num_list) > 0:
        dat_x_num = encoder_num.transform(dat[var_num_list])
    else:
        dat_x_num = dat[var_num_list].values

    if target_col in dat:
        y = np.log(dat.incwage.values).reshape(-1, 1)
    else:
        y = None
    if 'case_id' in dat:
        y_id = dat['case_id']
    else:
        y_id = None
    return {"inference": {'x_cat': dat_x_cat, 'x_num': dat_x_num, 'y': y, 'case_id': y_id},
            'var_cat': var_cat_list, "var_num": var_num_list, }
