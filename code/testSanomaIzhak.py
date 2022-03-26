# pip install virtualenv
# virtualenv sanoenv
# sanoenv\Scripts\activate
# pip install -r requirements.txt

import sys
sys.path.append('..')
from config import config

import pprint
import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

n_jobs = 4
train_size = .8

def fi_plot(data, model, model_name):
    fi = pd.DataFrame({'feature': data.columns.tolist(), \
                        'feature_importance': model.feature_importances_})
    fi.sort_values(by=['feature_importance'], ascending=False, inplace=True)
    fi.set_index('feature', inplace=True)
    minmaxscaler = MinMaxScaler()
    fi['feature_importance'] = minmaxscaler.fit_transform(fi['feature_importance'].values.reshape(-1,1))
    ax = fi.sort_values(by='feature_importance').tail(30).plot(kind='barh', color='gray', figsize=(16, 12))
    for p in ax.patches:
        ax.annotate(str(('%.3f'%p.get_width())),
        (p.get_x()+p.get_width(), p.get_y()),
        xytext=(5,0),
        textcoords='offset points',
        horizontalalignment='left',
        fontsize=8.)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.yticks(rotation=35, size=7)
    plt.title(f'relative feature importances for {model_name}')
    plt.xlabel('importance')
    plt.ylabel('feature')
    plt.show()


#### READ THE DATA

DATA_DIR = config['DATA_DIR']
dt = pd.read_excel(DATA_DIR, engine='openpyxl', sheet_name='Data')
dt.info()

for col in dt.columns:
    print(f'unique values for variable "{col}"', dt[col].unique(), sep='\n')

target = 'purchase'
#### CREATE THE DUMMIES FOR DEVICE, OS and DAY OF THE WEEK CATEGORIES

device_cols = pd.get_dummies(dt['devicecategory'])
os_cols = pd.get_dummies(dt['operatingsystem'])
dow_cols = pd.get_dummies(dt['dow'])

assert dt.isnull().sum().sum() == 0

cat_cols = [col for col in dt.columns if dt[col].dtype == 'object']
num_cols = [col for col in dt.columns if (dt[col].dtype != 'object')&(col!=target)]

# import pandas_profiling as pp # to run in the Jupyter Notebook
# profile_sanoma = pp.ProfileReport(dt, title="Sanoma Paywall Personalized Offer Dataset", html={"style": {"full_width": True}}, sort=None)


dt = pd.concat([dt, device_cols, os_cols, dow_cols], axis=1)
dt.describe().T

X = dt[num_cols].copy()
y = dt[target]

X_vif = X.copy()
X_vif['const'] = 1
vif = pd.DataFrame()
vif['feature'] = X_vif.columns
vif['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
vif.sort_values('VIF', ascending=False, inplace=True)
print(vif)

X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, train_size=train_size)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, train_size=.5)

assert X_valid.shape == X_test.shape
print(X_valid.shape)

lgbm_params = dict(verbosity=-1,
                    nthread=n_jobs, 
                    random_state=42)

rf_params = dict(n_jobs=n_jobs, 
                random_state=42)

xgboost_params = dict(objective='binary:logistic',
                        seed=42,
                        n_jobs=n_jobs)

lgbm_model = LGBMClassifier(**lgbm_params)
lgbm_model.fit(X=X_train.values, y=y_train.values)
print(f'for LightGBM model accuracy is {accuracy_score(y_valid, lgbm_model.predict(X_valid))*100} percent')
fi_plot(X_valid, lgbm_model, 'LightGBM model')

xgboost_model = XGBClassifier(**xgboost_params)
xgboost_model.fit(X=X_train.values, y=y_train.values)
accuracy_score(y_valid, xgboost_model.predict(X_valid))
print(f'for XGBoost model accuracy is {accuracy_score(y_valid, xgboost_model.predict(X_valid))*100} percent')
fi_plot(X_valid, xgboost_model, 'XGBoost model')

rf_model = RandomForestClassifier(**rf_params)
rf_model.fit(X=X_train.values, y=y_train.values)
accuracy_score(y_valid, rf_model.predict(X_valid))
print(f'for Random Forest model accuracy is {accuracy_score(y_valid, rf_model.predict(X_valid))*100} percent')
fi_plot(X_valid, rf_model, 'Random Forest model')

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X=X_train.values, y=y_train.values)
accuracy_score(y_valid, lr_model.predict(X_valid))