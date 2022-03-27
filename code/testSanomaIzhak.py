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
from sklearn.model_selection import train_test_split, GridSearchCV, GroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from hyperopt import fmin, tpe, hp, Trials, space_eval

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

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

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
print(f'for LightGBM model accuracy is {accuracy_score(y_test, lgbm_model.predict(X_test))*100} percent')
fi_plot(X_test, lgbm_model, 'LightGBM model')

bo_lgbm_space = {'boosting_type': hp.choice('boosting_type', ['dart','gbdt','goss']), 
        'learning_rate': hp.choice('learning_rate', np.arange(0.0005, 0.1005, 0.0005)),
        'n_estimators': hp.choice('n_estimators', np.arange(25, 1000, 25, dtype=int)),
        'max_depth': hp.choice('max_depth', np.arange(5, 1500, 5, dtype=int)),
        'num_leaves': hp.choice('num_leaves', [3,5,7,15,31]),}


def lgbm_objective(params):
    mod = lgb.LGBMClassifier(**lgbm_params)
    score = cross_val_score(mod, X_train.values, y_train.values, scoring="accuracy", cv=GroupKFold(n_splits=4), n_jobs=n_jobs).mean()
    return abs(score)


bo_lgbm_model = LGBMClassifier(random_state=42, verbosity=-1)

trials = Trials()
model_params = bo_lgbm_model.get_params()
bo_lgbm_best = fmin(fn=lgbm_objective, space=bo_lgbm_space, algo=tpe.suggest, max_evals=c20, trials=trials)
print(space_eval(bo_lgbm_space, bo_lgbm_best))
model_params.update(space_eval(bo_lgbm_space, bo_lgbm_best))

hp_lgbm_best_model = lgb.LGBMRegressor(**model_params)


xgboost_model = XGBClassifier(**xgboost_params)
xgboost_model.fit(X=X_train.values, y=y_train.values)
accuracy_score(y_test, xgboost_model.predict(X_test))
print(f'for XGBoost model accuracy is {accuracy_score(y_test, xgboost_model.predict(X_test))*100} percent')
fi_plot(X_test, xgboost_model, 'XGBoost model')

rf_model = RandomForestClassifier(**rf_params)
rf_model.fit(X=X_train.values, y=y_train.values)
accuracy_score(y_test, rf_model.predict(X_test))
print(f'for Random Forest model accuracy is {accuracy_score(y_test, rf_model.predict(X_test))*100} percent')
fi_plot(X_test, rf_model, 'Random Forest model')

dt_model = DecisionTreeClassifier()
dt_model.fit(X=X_train.values, y=y_train.values)
accuracy_score(y_test, dt_model.predict(X_test))
fi_plot(X_test, dt_model, 'Decision Tree model')

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X=X_train.values, y=y_train.values)
accuracy_score(y_test, lr_model.predict(X_test))

ols_model = sm.OLS(y_train, sm.add_constant(X_train))
ols_result = ols_model.fit()
print(ols_result.summary())
y_pred_ols = (ols_result.predict(sm.add_constant(X_test)) > 0.5).astype(int)
accuracy_score(y_test, y_pred_ols)

grid_param_dt = {'criterion': ['gini', 'entropy'],
            'max_depth': [1, 5, 10, 30],
            'max_features': [1, 3, 5, 10, 20],
            'min_samples_leaf': [.1, .2, .5],
            'min_samples_split': [.1, .2, .5]
            }

grid_dt = GridSearchCV(dt_model, param_grid=grid_param_dt, cv=6, n_jobs=-1, scoring='accuracy')
grid_dt.fit(X_train, y_train)
best_dt = grid_dt.best_estimator_

fi_plot(X_test, best_dt, 'Tuned Decision Tree model')
print(f'for Tuned Decision Tree model accuracy is {accuracy_score(y_test, best_dt.predict(X_test))*100} percent')
