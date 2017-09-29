import time
import pdb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

def col_split(df, col, test_size=0.1):
    uniques = df[col].unique()
    num_unique = len(df[col].unique())
    test_count = int(num_unique * test_size)
    train_count = num_unique - test_count

    test_eras = np.random.choice(uniques, size=test_count, replace=False)
    test_set = df[df[col].isin(test_eras)]
    train_set = df[~df[col].isin(test_eras)]
    return train_set, test_set

def get_feat_target(df, feat_col, target_col):
    return df[feat_col], df[target_col]

def get_cv_params(model_func):
    lr_params = {
        'C': [10**i for i in range(-3, 3)]
    }

    rf_dict = {
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 10, 100],
        'max_depth': [5, 10],
    }

    gbt_dict = {
        'learning_rate': [0.1, 0.25, 0.5],
        'max_depth': [5],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 10, 100],
    }

    params = {
        LogisticRegression: lr_params,
        RandomForestClassifier: rf_dict,
        GradientBoostingClassifier: gbt_dict
    }

    return params.get(model_func, {})

def col_sample(df, col, frac):
    # Take sample_prob from each era
    grouped = df.groupby(col)
    sampled = grouped.apply(lambda x: x.sample(frac=frac))
    return sampled

def cv_model(model, param_dict, scoring, data_dict, folds=10, n_jobs=4):
    x_train = data_dict['x_train']
    y_train = data_dict['y_train']
    x_test = data_dict['x_test']
    y_test = data_dict['y_test']

    gridcv = GridSearchCV(model, param_dict, scoring=scoring, cv=folds, n_jobs=n_jobs)
    gridcv.fit(x_train, y_train)

    best_model = gridcv.best_estimator_
    y_prob = best_model.predict_proba(x_test)
    y_pred = best_model.predict(x_test)
    loss = log_loss(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)
    return loss, accuracy, best_model, gridcv


def run_all_cv(data_dict, folds=5, n_jobs=8):
    model_funcs = [LogisticRegression, RandomForestClassifier, GradientBoostingClassifier]
    for m in model_funcs:
        start_time = time.time()
        print '=' * 80
        print m.__name__
        model = m()
        cv_params = get_cv_params(m)
        loss, accuracy, best_model, gridcv = cv_model(model, cv_params, 'accuracy',
                                            data_dict, folds=folds, n_jobs=n_jobs)
        elapsed = time.time() - start_time
        print loss, accuracy
        print '{}'.format(best_model)
        print 'Loss: {}'.format(loss)
        print 'Accuracy: {}'.format(accuracy)
        print gridcv.best_params_
        print "Time: {:.2f}".format(elapsed)

def main():
    df = pd.read_csv('data/numerai_training_data.csv')
    feat_cols = df.columns[3:-1]
    target_col = df.columns[-1]

    train, test = era_split(df, 'era', test_size=0.2)
    x_train, y_train = get_feat_target(train, feat_cols, target_col)
    x_test, y_test = get_feat_target(test, feat_cols, target_col)
    data_dict = {
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test
    }

    small_size = 1000
    small_data_dict = {
        'x_train': x_train[:small_size],
        'x_test': x_test[small_size:2*small_size],
        'y_train': y_train[:small_size],
        'y_test': y_test[small_size:2*small_size]
    }
    run_all_cv(data_dict, n_jobs=6)

if __name__ == '__main__':
    main()
