# %% Imports
''' Module Docstring
'''
import os
from contextlib import contextmanager
import gc
import time
import warnings

# Data manipulation
import numpy as np
import pandas as pd

# Data preparation
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from pickle import dump

# %% Settings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.display.max_rows = 500
pd.options.display.max_columns = 500
FEATURE_SELECTION = None  # None or integer
STRATEGY_RESAMPLING = "undersampled"  # undersampled, original, SMOTE


# %% Définitions de fonctions
@contextmanager
def timer(title):
    '''Timer
    '''
    t0 = time.time()
    yield
    print(f"{title} - done in {time.time() - t0}s")


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category=True):
    ''' OHE
    '''
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# Preprocess application_train.csv and application_test.csv
def get_application_data(num_rows=None, nan_as_category=False, application_test_data=False):
    '''Import principal dataset 'application_train.csv and applies first cleaning operations
    '''
    if application_test_data:
        print('In get_applciation_data/1a')
        df = pd.read_csv(os.path.join("Dataset", "application_test.csv"), nrows=num_rows)
    else:
        print('In get_applciation_data/1b')
        df = pd.read_csv(os.path.join("Dataset", "application_train.csv"), nrows=num_rows)
    df = df[df['CODE_GENDER'] != 'XNA']

    print('In get_applciation_data/2')
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], _ = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, _ = one_hot_encoder(df, nan_as_category)

    print('In get_applciation_data/3')
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    print('In get_applciation_data/4')
    gc.collect()
    print('In get_applciation_data/5')
    return df


# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows=None, nan_as_category=True):
    ''' import and clean bureau_and_balance data.
    '''
    bureau = pd.read_csv(os.path.join("Dataset", "bureau.csv"), nrows=num_rows)
    bb = pd.read_csv(os.path.join("Dataset", "bureau_balance.csv"), nrows=num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat:
        cat_aggregations[cat] = ['mean']
    for cat in bb_cat:
        cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg


# Preprocess previous_applications.csv
# def previous_applications(num_rows=None, nan_as_category=True):
def previous_applications(num_rows=None):
    '''Import and process previous_application.csv
    '''
    prev = pd.read_csv(os.path.join("Dataset", "previous_application.csv"), nrows=num_rows)
    # prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg


# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows=None):
    ''' import and clean pos_cash data.csv
    '''
    pos = pd.read_csv(os.path.join("Dataset", "POS_CASH_balance.csv"), nrows=num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category=True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg


# Preprocess installments_payments.csv
def installments_payments(num_rows=None):
    ''' import and clean installments_payments data.
    '''
    ins = pd.read_csv(os.path.join("Dataset", "installments_payments.csv"), nrows=num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg


# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows=None):
    '''Import and process credit_card_balance.csv
    '''
    cc = pd.read_csv(os.path.join("Dataset", "credit_card_balance.csv"), nrows=num_rows)
    # cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)
    cc, _ = one_hot_encoder(cc, nan_as_category=True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg


def data_preparation(main_dataset_only=True, debug=False, new_data=False):
    '''Process datasets into actionnable train and test sets.
    '''
    num_rows = 10000 if debug else None
    print('In Data_prep.data_preparation/1')

    data = get_application_data(num_rows, application_test_data=new_data)
    if new_data:
        print('In Data_prep.data_preparation/2')
        y_train_ = pd.DataFrame(np.zeros((data.shape[0], 2)), columns=['SK_ID_CURR', 'TARGET'])
    else:
        print('In Data_prep.data_preparation/3')
        y_train_ = data[['SK_ID_CURR', 'TARGET']].copy()
        data.drop(columns=['TARGET'], inplace=True)

    print('In Data_prep.data_preparation/4')
    if not main_dataset_only:
        with timer("Process bureau and bureau_balance"):
            bureau = bureau_and_balance(num_rows)
            print("Bureau df shape:", bureau.shape)
            # df = df.join(bureau, how='left', on='SK_ID_CURR')
            data = data.join(bureau, how='left', on='SK_ID_CURR')
            del bureau
            gc.collect()
        with timer("Process previous_applications"):
            prev = previous_applications(num_rows)
            print("Previous applications df shape:", prev.shape)
            # df = df.join(prev, how='left', on='SK_ID_CURR')
            data = data.join(prev, how='left', on='SK_ID_CURR')
            del prev
            gc.collect()
        with timer("Process POS-CASH balance"):
            pos = pos_cash(num_rows)
            print("Pos-cash balance df shape:", pos.shape)
            # df = df.join(pos, how='left', on='SK_ID_CURR')
            data = data.join(pos, how='left', on='SK_ID_CURR')
            del pos
            gc.collect()
        with timer("Process installments payments"):
            ins = installments_payments(num_rows)
            print("Installments payments df shape:", ins.shape)
            # df = df.join(ins, how='left', on='SK_ID_CURR')
            data = data.join(ins, how='left', on='SK_ID_CURR')
            del ins
            gc.collect()
        with timer("Process credit card balance"):
            cc = credit_card_balance(num_rows)
            print("Credit card balance df shape:", cc.shape)
            # df = df.join(cc, how='left', on='SK_ID_CURR')
            data = data.join(cc, how='left', on='SK_ID_CURR')
            del cc
            gc.collect()

    print('In Data_prep.data_preparation/5')
    x_train_, x_test_, y_train_, y_test_ = train_test_split(data, y_train_, test_size=.25, random_state=10)

    return x_train_, x_test_, y_train_, y_test_


# Over and under sampling
def resampling(x, y, strategy="undersampled"):
    '''Applies a resampling strategy to X and y.
    -----------
    Parameters:
        - X : DataFrame : X_Train set to be resampled.
        - y : DataFrame : y_train set to be resampled.
        - strategy: String : Strategy to be applied among:
                                - "undersampled" : Under-sampling by setting both target categories to the size of the imbalanced class.
                                - "oversampled" : Over-sampling by multiplying the number of entries of the imbalanced target class to
                                                    match the other class size.
                                - "SMOTE" : Over-sampling using the Synthetic Minority Oversampling Technique strategy.
                                - else, no strategy is applied and both X_train and y_train are returned unchanged.
    -----------
    Returns:
        - X_train_resampled : DataFrame : X_train after application of a resampling strategy.
        - y_train_resampled : DataFrame : y_train after application of a resampling strategy.
    '''
    if isinstance(y, np.ndarray):
        print("is nd.array")
        y = pd.Series(y)
        y = pd.DataFrame(y)
    if isinstance(y, pd.core.series.Series):
        print("is pd.Series")
        y = y.to_frame()
    if strategy == "undersampled":
        train = x.set_index('SK_ID_CURR').join(y.set_index("SK_ID_CURR"))

        train_pos = train.query("TARGET == 1")
        train_neg = train.query("TARGET == 0")
        train_neg = train_neg.sample(train_pos.shape[0]).copy()
        train = pd.concat([train_pos, train_neg], axis=0)

        nb_pos = train[train['TARGET'] == 1].shape[0]
        nb_neg = train[train['TARGET'] == 0].shape[0]
        print(f'Proportion de targets négatives (après under-sampling): {round(100*nb_neg/(nb_pos+nb_neg),2)}%')

        y_train_resampled = train.pop('TARGET')
        x_train_resampled = train.reset_index(names=['SK_ID_CURR'])
        print("In undersampling, x_train_resampled.columns:", x_train_resampled.columns)
        print("In undersampling, y_train_resampled.head:", y_train_resampled.head())

    elif strategy == "oversampled":
        train = x.set_index('SK_ID_CURR').join(y.set_index("SK_ID_CURR"))
        print("oversample - x.shape:", x.shape)
        print("oversample - y.shape:", y.shape)
        print("oversample - train.shape:", train.shape)
        print('resampling.oversampled X.isna().sum().sum():', x.isna().sum().sum())
        print('resampling.oversampled y.isna().sum().sum():', y.isna().sum().sum())
        print('resampling.oversampled train.isna().sum().sum():', train.isna().sum().sum())
        train_pos = train.query("TARGET == 1")
        train_neg = train.query("TARGET == 0")

        list_df = [train_pos for i in range(int(train_neg.shape[0]/train_pos.shape[0]))]
        list_df.append(train_neg)
        train = pd.concat(list_df, axis=0)

        nb_pos = train[train['TARGET'] == 1].shape[0]
        nb_neg = train[train['TARGET'] == 0].shape[0]
        print(f'Proportion de targets négatives (après over-sampled): {round(100*nb_neg/(nb_pos+nb_neg),2)}%')

        y_train_resampled = train.pop('TARGET')
        x_train_resampled = train
        print("In resampling.oversampled, x_train_resampled.shape:", x_train_resampled.shape)
        print('resampling.oversampled train.isna().sum().sum():', train.isna().sum().sum())
        print("In resampling.oversampled, NA sur x_train_resampled:", x_train_resampled.isna().sum().sum())

    elif strategy == "SMOTE":
        print("In SMOTE, x.columns:", x.columns)
        smote = SMOTE(sampling_strategy=.7)
        y.drop(columns=['SK_ID_CURR'], inplace=True)
        x_train_resampled, y_train_resampled = smote.fit_resample(x, y)
        y_train_resampled = y_train_resampled.pop("TARGET")

        print("In undersampling, x_train_resampled.columns:", x_train_resampled.columns)
        print("In undersampling, y_train_resampled.head:", y_train_resampled.head())

    elif strategy == "original":
        y_train_resampled = y
        x_train_resampled = x
        print("In resampling.original, x_train_resampled.shape:", x_train_resampled.shape)
    return x_train_resampled, y_train_resampled


# Export des dataset nettoyés
def export_datasets(x_train_, x_test_, y_train_, y_test_, strategy_resampling):
    '''Exporte les données préparées X_train, X_test, y_train, y_test

    ------------
    Inputs:
        - X_train : pandas.DataFrame : X_train
        - X_test : pandas.DataFrame : X_test
        - y_train : pandas.DataFrame ou Series: y_train
        - y_test : pandas.DataFrame ou Series : y_test
    '''
    if strategy_resampling == "original":
        # datasets non resampled
        file_name = ""
    else:
        file_name = f"_{strategy_resampling}"

    x_train_.to_parquet(os.path.join('Dataset', 'Data clean', f'X_train{file_name}.parquet'), index=False)
    if isinstance(y_train_, pd.Series):
        y_train_ = y_train_.to_frame()
    y_train_.to_parquet(os.path.join('Dataset', 'Data clean', f'y_train{file_name}.parquet'), index=False)
    x_test_.to_parquet(os.path.join('Dataset', 'Data clean', 'X_test.parquet'), index=False)
    if isinstance(y_test_, pd.Series):
        y_test_ = y_test_.to_frame()
    y_test_.to_parquet(os.path.join('Dataset', 'Data clean', 'y_test.parquet'), index=False)

    print("Exports: X_train.head(1):", x_train_.head(1))
    print("Exports: X_test.head(1):", x_test_.head(1))
    print("Exports: y_train.head(1):", y_train_.head(1))
    print("Exports: y_test.head(1):", y_test_.head(1))


# %%
if __name__ == "__main__":
    # Récupération des données des différentes bases, séparées en train et test sets.
    X_train, X_test, y_train, y_test = data_preparation()
    features = list(X_train.columns)
    # print("Data prep -  X_train.shape:", X_train.shape)
    # print("Data prep -  X_test.shape:", X_test.shape)
    # print("Data prep -  y_train.shape:", y_train.shape)
    # print("Data prep -  y_test.shape:", y_test.shape)

    # Imputation de valeurs manquantes
    imputer = SimpleImputer(strategy="most_frequent")
    X_train = imputer.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=features)
    print("X_test.shape", X_test.shape)
    X_test = imputer.transform(X_test)
    print("X_test.shape", X_test.shape)
    X_test = pd.DataFrame(X_test, columns=features)
    dump(imputer, open('imputer.pkl', 'wb'))

    print("Imputation -  X_test.shape:", X_test.shape)

    # Features selection kbest
    if FEATURE_SELECTION is not None:
        kbest = SelectKBest(score_func=f_classif, k=FEATURE_SELECTION)
        kbest.fit(X_train, y_train)

        # print("Feature selection", kbest.get_support())
        # print("Features scores", kbest.scores_)
        selected_features = list(X_train.columns[kbest.get_support()])
        print("Selected features:", selected_features)

        X_train = X_train[selected_features]
        X_test = X_test[selected_features]

    # Over/Under-sampling
    X_train, y_train = resampling(X_train, y_train, STRATEGY_RESAMPLING)

    if isinstance(y_test, pd.core.series.Series):
        y_test = y_test.to_frame()
    if isinstance(y_train, pd.core.series.Series):
        y_train = y_train.to_frame()

    y_train['SK_ID_CURR'] = y_train.index
    y_test['SK_ID_CURR'] = y_test.index

    # print("Exit y_test.head(1):", y_test.head(1))
    # print("Exit y_train.head(1):", y_train.head(1))
    export_datasets(X_train, X_test, y_train, y_test, STRATEGY_RESAMPLING)

# %%
