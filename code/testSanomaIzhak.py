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
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from hyperopt import fmin, tpe, hp, Trials, space_eval

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

n_jobs = 4
train_size = .8

def fi_plot(data, model, model_name, save=False, plot_name=None):
    '''Function builds the min-max scaled feature importances from given model
        Args:
            data (pandas.DataFrame): 
            model: model object to pull the feature importances (sklear like API needed)
            model_name (str): the name of the model for the title
            save (bool): plot saving flag
            plot_name (str): the name of the figure to save
        Returns: None
    '''
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
    if save:
        plt.gcf().savefig(OUTPUT_DIR+plot_name+'.png')
    plt.show()

#### READ THE DATA

DATA_DIR = config['DATA_DIR']
OUTPUT_DIR = config['OUTPUT_DIR']

dt = pd.read_excel(DATA_DIR, engine='openpyxl', sheet_name='Data')
dt.info()

for col in dt.columns:
    print(f'unique values for variable "{col}"', dt[col].unique(), sep='\n')

target = 'purchase'

# TYPES OF THE COLUMNS: CATEGORIES AND NUMERIC
cat_cols = [col for col in dt.columns if dt[col].dtype == 'object']
num_cols = [col for col in dt.columns if (dt[col].dtype != 'object')&(col!=target)]

#### CHECK THE NULL VALUES
assert dt.isnull().sum().sum() == 0

#### CREATE THE DUMMIES FOR DEVICE, OS and DAY OF THE WEEK CATEGORIES
device_cols = pd.get_dummies(dt['devicecategory'])
os_cols = pd.get_dummies(dt['operatingsystem'])
dow_cols = pd.get_dummies(dt['dow'])

dt = pd.concat([dt, device_cols, os_cols, dow_cols], axis=1)
dt.describe().T

# import pandas_profiling as pp # to run in the Jupyter Notebook
# profile_sanoma = pp.ProfileReport(dt, title="Sanoma Paywall Personalized Offer Dataset", html={"style": {"full_width": True}}, sort=None)

# DATA FRAME (NUMERIC)
X = dt[num_cols].copy()
y = dt[target]

# VARIANCE INFLATION FACTOR TO CHECK FOR MULTICOLLINEARITY (SPOILER: ITS THERE AND ITS BIG)
X_vif = X.copy()
X_vif['const'] = 1
vif = pd.DataFrame()
vif['feature'] = X_vif.columns
vif['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
vif.sort_values('VIF', ascending=False, inplace=True)
print(vif)

# TRAIN TEST SPLIT (NO VALIDATION SET IN THE INTEREST OF TIME)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
print(X_test.shape)

##### SIMPLE ORDINARY LEAST SQUARE (OLS) REGRESSION MODEL 
##### (NB: OLS REGRESSION DOES NOT PROVIDE BINARY PREDICTION OUTPUT, SO 
##### IF PREDICTION IS GREATER THAN .5 IT IS ENCODED AS 1, OTHERWISE 0)
ols_model = sm.OLS(y_train, sm.add_constant(X_train))
ols_result = ols_model.fit()
print(ols_result.summary())
y_pred_ols = (ols_result.predict(sm.add_constant(X_test)) > 0.5).astype(int)
print(f'for OLS regression model accuracy is {accuracy_score(y_test, y_pred_ols)*100} percent')

##### LOGISTIC REGRESSION
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X=X_train.values, y=y_train.values)
print(f'for OLS regression model accuracy is {accuracy_score(y_test, lr_model.predict(X_test))*100} percent')

##### DEFAULT (UNTUNED) DECISION TREE MODEL
dt_model = DecisionTreeClassifier()
dt_model.fit(X=X_train.values, y=y_train.values)
accuracy_score(y_test, dt_model.predict(X_test))
fi_plot(X_test, dt_model, 'Decision Tree model')

##### DEFAULT RANDOM FOREST MODEL
rf_params = dict(n_jobs=n_jobs, 
                random_state=42)
rf_model = RandomForestClassifier(**rf_params)
rf_model.fit(X=X_train.values, y=y_train.values)
accuracy_score(y_test, rf_model.predict(X_test))
print(f'for Random Forest model accuracy is {accuracy_score(y_test, rf_model.predict(X_test))*100} percent')
fi_plot(X_test, rf_model, 'Random Forest model')

##### DEFAULT XGBOOST MODEL
xgboost_params = dict(objective='binary:logistic',
                        seed=42,
                        n_jobs=n_jobs)

xgboost_model = XGBClassifier(**xgboost_params)
xgboost_model.fit(X=X_train.values, y=y_train.values)
accuracy_score(y_test, xgboost_model.predict(X_test))
print(f'for XGBoost model accuracy is {accuracy_score(y_test, xgboost_model.predict(X_test))*100} percent')
fi_plot(X_test, xgboost_model, 'XGBoost model')

##### DEFAULT LGBM MODEL
lgbm_params = dict(verbosity=-1,
                    nthread=-1, 
                    random_state=42)

lgbm_model = LGBMClassifier(**lgbm_params)
lgbm_model.fit(X=X_train.values, y=y_train.values)
print(f'for LightGBM model accuracy is {accuracy_score(y_test, lgbm_model.predict(X_test))*100} percent')
fi_plot(X_test, lgbm_model, 'LightGBM model', save=True, plot_name='default_lgbm_fi')

##### GRID SEARCH TUNED DECISION TREE MODEL
grid_param_dt = {'criterion': ['gini', 'entropy'],
            'max_depth': [1, 5, 10, 30],
            'max_features': [1, 3, 5, 10, 20],
            'min_samples_leaf': [.1, .2, .5],
            'min_samples_split': [.1, .2, .5]
            }

grid_dt = GridSearchCV(dt_model, param_grid=grid_param_dt, cv=5, n_jobs=-1, scoring='accuracy')
grid_dt.fit(X_train, y_train)
best_dt = grid_dt.best_estimator_
print(f'for Tuned Decision Tree model accuracy is {accuracy_score(y_test, best_dt.predict(X_test))*100} percent')
fi_plot(X_test, best_dt, 'Tuned Decision Tree model')

##### GRID SEARCH TUNED RANDOM FOREST
grid_param_rf = {'n_estimators': [5, 10, 20, 50, 100],
            'max_depth': [1, 5, 10, 30, 50],
            'max_features': ['auto', 'sqrt'],
            'min_samples_leaf': [1, 2, 5, 10],
            'min_samples_split': [x for x in np.linspace(0.0001, 1, num=4)],
            'bootstrap': [True, False]}

grid_rf = GridSearchCV(rf_model, param_grid=grid_param_rf, cv=5, n_jobs=-1, scoring='accuracy')
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_

print(f'for Tuned Decision Tree model accuracy is {accuracy_score(y_test, best_dt.predict(X_test))*100} percent')
fi_plot(X_test, best_dt, 'Tuned Decision Tree model')

##### BAYESIAN OPTIMIZATION TUNING OF LGBM CLASSIFIER'S HYPERPARAMETERS
### PARAMETER SEARCH DIMENSIONS
bo_lgbm_space = {'boosting_type': hp.choice('boosting_type', ['dart','gbdt','goss']), 
        'learning_rate': hp.choice('learning_rate', np.arange(0.0005, 0.1005, 0.0005)),
        'n_estimators': hp.choice('n_estimators', np.arange(25, 1000, 25, dtype=int)),
        'max_depth': hp.choice('max_depth', np.arange(5, 1500, 5, dtype=int)),
        'num_leaves': hp.choice('num_leaves', [3,5,7,15,31]),}

### OBJECTIVE FUNCTION FOR THE OPRIMIZER
def lgbm_objective(params):
    mod = LGBMClassifier(**lgbm_params)
    score = cross_val_score(mod, X_train.values, y_train.values, scoring="accuracy", cv=KFold(n_splits=4), n_jobs=n_jobs).mean()
    return abs(score)

bo_lgbm_model = LGBMClassifier(random_state=42, verbosity=-1)

trials = Trials()
model_params = bo_lgbm_model.get_params()
bo_lgbm_best = fmin(fn=lgbm_objective, space=bo_lgbm_space, algo=tpe.suggest, max_evals=20, trials=trials)
print(space_eval(bo_lgbm_space, bo_lgbm_best))
model_params.update(space_eval(bo_lgbm_space, bo_lgbm_best))

bo_lgbm_best_model = LGBMClassifier(**model_params)
bo_lgbm_best_model.fit(X=X_train.values, y=y_train.values)
print(f'for BO Tuned LightGBM model accuracy is {accuracy_score(y_test, bo_lgbm_best_model.predict(X_test))*100} percent')
fi_plot(X_test, bo_lgbm_best_model, 'BO Tuned LightGBM model', save=True, plot_name='bo_tuned_lgbm_fi')

