from datetime import datetime
import sys

import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale, PolynomialFeatures
from sklearn.svm import SVR


def compute_rmse(preds, y):
    return np.sqrt(np.mean((preds - y)**2))


def data_v1():
    '''This function loads test and training data into memory, transforms each dataset,
    and returns a X, y, and X_test. It offers iterability, flexibility, and reproducability
    in a data science competition workflow. Credit to @pawel for the inspiration.
    '''

    train = pd.read_csv(train_path)
    X_train, y_train = train.drop(['Id', 'revenue'], axis=1), train['revenue']
    X_test = pd.read_csv(test_path).drop('Id', axis=1)
    X_dict = {'X_train': X_train, 'X_test': X_test}

    start_date = datetime.strptime('01/01/2015', '%m/%d/%Y')

    for X_name, X in X_dict.iteritems():
        X['days_open'] = X['Open Date'].map(lambda x: np.log((start_date - datetime.strptime(x, '%m/%d/%Y')).days + 1))
        X = X.drop('Open Date', axis=1)

        X['city_group'] = X['City Group'].map(lambda x: 1 if x == 'Other' else 0)
        X = X.drop('City Group', axis=1)

        X['P28'] = np.log(X['P28'] + 1)

        X = scale(X[['days_open', 'city_group', 'P28']])
        X = PolynomialFeatures(2, include_bias=False, interaction_only=True).fit_transform(X)

        X_dict[X_name] = X

    return X_dict['X_train'], y_train, X_dict['X_test']


class TFIModelObj(object):
    '''Model class for TFI Kaggle competition. This class accepts a scikit-learn
    model object (or any model object with fit() and predict() methods), as well
    as a `data_func` which returns an array-like X_train, y_train, and X_test.

    With ease, in a K-folds fashion, one can generate out-of-folds predictions and
    inspect the cross-validated error. When satisified with the model's performance,
    it is thereafter trivial to call fit_predict() and generate leaderboard predictions.

    * Note: We take the square root of y before fitting the model, then square
    the predictions. This plays nicely with the initial outliers in our target.
    '''

    def __init__(self, model, data_func):
        self.X_train, self.y_train, self.X_test = data_func()
        self.model = model
        self.cv = KFold(n=self.X_train.shape[0], n_folds=10, random_state=12345)
        self.oof_predictions_train = None
        self.rmse = None
        self.leaderboard_predictions = None

    def cross_validate(self):
        oof_predictions_tr = np.zeros(self.X_train.shape[0])
        for tr_idx, te_idx in self.cv:
            self.model.fit(self.X_train[tr_idx], np.sqrt(self.y_train[tr_idx]))
            preds = self.model.predict(self.X_train[te_idx])
            preds = preds**2
            oof_predictions_tr[te_idx] = preds
        rmse = compute_rmse(oof_predictions_tr, self.y_train)
        self.oof_predictions_train = oof_predictions_tr
        self.rmse = rmse
        return oof_predictions_tr, rmse

    def fit_predict(self):
        self.model.fit(self.X_train, np.sqrt(self.y_train))
        leaderboard_preds = (self.model.predict(self.X_test)**2)
        self.leaderboard_predictions = leaderboard_preds
        return leaderboard_preds


if __name__ == '__main__':
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    sample_submission_path = sys.argv[3]
    submission_path = sys.argv[4]

    # cross validate
    svr = TFIModelObj(model=SVR(C=1000.), data_func=data_v1)
    oof_preds_svr, rmse_svr = svr.cross_validate()
    lr = TFIModelObj(model=LinearRegression(), data_func=data_v1)
    oof_preds_lr, rmse_lr = lr.cross_validate()

    try:
        assert (svr.y_train == lr.y_train).all()
        y = svr.y_train = lr.y_train
        ens_rmse = compute_rmse(np.mean([oof_preds_svr, oof_preds_lr], axis=0), y)
        print('RMSE of averaged models: {}').format(ens_rmse)
    except AssertionError:
        sys.exit('`y_train` from first model != `y_train` from second model. Something went wrong')

    # generate submission
    preds_svr, preds_lr = svr.fit_predict(), lr.fit_predict()
    leaderboard_preds = np.mean([preds_svr, preds_lr], axis=0)
    sub = pd.read_csv(sample_submission_path)
    sub['Prediction'] = leaderboard_preds
    sub.to_csv(submission_path, index_label='Id', index=False)
