import numpy as np
from sklearn.metrics import r2_score
from lightgbm.sklearn import LGBMRegressor

class InstantiationModelLight:
    def __init__(self, features, nbr_genes=7436, num_leaves=31, max_depth=-1, n_estimators=50, boosting_type='gbdt', n_jobs=None):
        self.feature_idx = {k: i for i, k in enumerate(np.unique(features[1,:]))}
        self.create_instantiation(features, nbr_genes)
        self.model = LGBMRegressor(num_leaves=num_leaves, max_depth=max_depth, n_estimators=n_estimators, boosting_type=boosting_type, n_jobs=n_jobs, force_col_wise=True)

    def __call__(self, *args, **kwds):
        pass

    def create_instantiation(self, features, nbr_genes):
        self.features = np.zeros((nbr_genes, len(self.feature_idx)))
        self.features[features[0,:], [self.feature_idx[i.item()] for i in features[1,:]]] = 1
        pass

    def fit(self, X, y):
        interaction_x = self.features[X[0,:]] * self.features[X[1,:]]
        self.model.fit(interaction_x, y)

    def predict(self, X):
        interaction_x = self.features[X[0,:]] * self.features[X[1,:]]
        y = self.model.predict(interaction_x)
        return y

    def score(self, X, y):
        yhat = self.predict(X)
        score = r2_score(y, yhat)
        return score
