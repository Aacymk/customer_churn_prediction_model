import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import pandas as pd
import kagglehub
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def main():

    path = kagglehub.dataset_download("blastchar/telco-customer-churn")
    csv_path = path + "/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(csv_path)
    df['Churn'] = np.where(df['Churn'] == 'Yes', 1, 0)

    test = df.tail(int(0.25*len(df))).copy()
    train = df.iloc[:len(df)-len(test)].copy()
    
    train_df, scaler = transform_train(train) ### consider if u even need cv

    train_X = train_df.drop(['Churn', 'customerID'], axis=1).copy()
    train_y = train_df['Churn']
    model = XGBClassifier(eval_metric='logloss', random_state=42)

    selector = RFE(estimator=model, n_features_to_select=24, step=1)
    selector.fit(train_X, train_y)

    selected_features = train_X.columns[selector.support_]
    reduced_train_X = train_X[selected_features]

    params = {
        'max_depth':[3,4,5],
        'learning_rate':[0.1, 0.05, 0.01],
        'n_estimators':[200, 400, 700],
        'subsample':[0.6,0.8,1.0],
        'colsample_bytree':[0.6,0.8,1.0]
    }
    grid = GridSearchCV(
        estimator=XGBClassifier(eval_metric='logloss', random_state=42),
        param_grid=params,
        scoring='f1',
        cv=5,
        n_jobs=-1
    )

    grid.fit(reduced_train_X, train_y)
    chosen_threshold = 0.35
    model = XGBClassifier(eval_metric='logloss',
                        max_depth=grid.best_params_['max_depth'],
                        learning_rate=grid.best_params_['learning_rate'],
                        n_estimators=grid.best_params_['n_estimators'],
                        colsample_bytree=grid.best_params_['colsample_bytree'],
                        subsample=grid.best_params_['subsample'],
                        scale_pos_weight=2.7,
                        random_state=42)
    model.fit(reduced_train_X, train_y)
    test_transformed = transform_test(test, scaler)
    test_X = test_transformed[reduced_train_X.columns]
    test_y = test['Churn']
    proba = model.predict_proba(test_X)[:,1]
    preds = (proba >= chosen_threshold).astype(int)
    print(classification_report(preds, test_y))


def create_aggregation_feats(df, bool_cols, cont_cols):
    for bool_col in bool_cols:
        for cont_col in cont_cols:
            means = df.groupby(bool_col)[cont_col].mean()
            stds = df.groupby(bool_col)[cont_col].std()
            new_col1_name = f'{cont_col}_by_{bool_col}_mean'
            new_col2_name = f'{cont_col}_by_{bool_col}_std'
            df[new_col1_name] = df[bool_col].map(means)
            df[new_col2_name] = df[bool_col].map(stds)
    return df

def create_combination_feats(df, bool_cols):
    for col1 in bool_cols:
        for col2 in bool_cols:
            if col1 == col2:
                continue
            if f'{col1}_and_{col2}' not in df.columns and f'{col2}_and_{col1}' not in df.columns:
                df[f'{col1}_and_{col2}'] = df[col1] & df[col2]
            if f'{col1}_or_{col2}' not in df.columns and f'{col2}_or_{col1}' not in df.columns:
                df[f'{col1}_or_{col2}'] = df[col1] | df[col2]
    return df

def transform_train(df):
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    yes_no_cols = [col for col in df.columns if df[col].dropna().isin(['Yes', 'No']).all()]

    for col in yes_no_cols:
        df[col] = np.where(df[col] == 'Yes', True, False).astype('bool')
    df['gender'] = np.where(df['gender'] == 'Male', True, False)
    df['TotalCharges'] = np.log1p(df['TotalCharges'])
    bool_cols = [col for col in df.columns if df[col].dtype == 'bool']
    df = create_aggregation_feats(df, bool_cols, ['TotalCharges', 'tenure', 'MonthlyCharges'])
    df = create_combination_feats(df, bool_cols)
    scaler = MinMaxScaler()
    df['TotalCharges'] = scaler.fit_transform(df[['TotalCharges']])
    categorical = [val for val in df.columns if df[val].dtype == 'O' and val not in ['customerID','Churn']]
    df = pd.get_dummies(df, columns=categorical, drop_first=True)

    return df, scaler

def transform_test(df, scaler):                                         
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    yes_no_cols = [col for col in df.columns if df[col].dropna().isin(['Yes', 'No']).all()]
    for col in yes_no_cols:
        df[col] = np.where(df[col] == 'Yes', True, False).astype('bool')
    df['gender'] = np.where(df['gender'] == 'Male', True, False)
    df['TotalCharges'] = np.log1p(df['TotalCharges'])
    bool_cols = [col for col in df.columns if df[col].dtype == 'bool']
    df = create_aggregation_feats(df, bool_cols, ['TotalCharges', 'tenure', 'MonthlyCharges'])
    df = create_combination_feats(df, bool_cols)
    df['TotalCharges'] = scaler.transform(df[['TotalCharges']])
    categorical = [val for val in df.columns if df[val].dtype == 'O' and val not in ['customerID','Churn']]
    df = pd.get_dummies(df, columns=categorical, drop_first=True)
    return df


if __name__ == "__main__":
    main()