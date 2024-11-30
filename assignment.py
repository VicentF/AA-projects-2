import itertools
import time
import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from catboost import CatBoostRegressor, Pool

from DatasetManipulator import *

pd.set_option("display.max_columns", None)


# cMSE, the new metric for evaluation
def error_metric(y, y_hat, c):
    err = y - y_hat
    err = (1-c)*err**2 + c*np.maximum(0, err)**2
    return np.sum(err)/err.shape[0]

def derivative_error_metric(y, y_hat, c, X):
    def derivative_max(e):
        # return 0 if e < 0 else 1
        return (e <= 0).astype(float)
    err = y - y_hat
    err2 = (-1 + c*derivative_max(err)) * err
    return (X.T @ err2) / err.shape[0]

def derivative_abs(w):
    return (w > 0).astype(float) - (w < 0).astype(float)

def product_dict(**kwargs):
    # https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))

# models
class Model:
    def __init__(self, X_train, y_train, X_val, y_val, c_train, c_val, train=False):
        self.start_time = time.time()
        self.std_scaler = StandardScaler()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.c_train = c_train
        self.c_val = c_val
        self.model = None
        if train:
            self.train()
        return

    @staticmethod
    def load(name:str):
        with open(f"models/{name}.pkl", "rb") as f:
            return load(f)

    def save(self, name:str):
        with open(f"models/{name}.pkl", "wb") as f:
            dump(self, f, protocol=5)
        return

    def __repr__(self):
        return f"Model"

    def normalize(self, X_train, X_val, std_scaler=None):
        if std_scaler is None:
            std_scaler = self.std_scaler
        X_train = std_scaler.fit_transform(X_train)
        X_val = std_scaler.transform(X_val)
        return X_train, X_val

    def normalize_train(self, X_train, std_scaler=None):
        if std_scaler is None:
            std_scaler = self.std_scaler
        return std_scaler.fit_transform(X_train)

    def train(self):
        if self.model is None:
            raise Exception("Unassigned model. Shouldn't be none!")
        self.model.fit(self.X_train, self.y_train)
        return

    def logs(self):
        print(f"{self.__class__.__name__} training took {round(time.time() - self.start_time, 2)} seconds")
        print(self.__repr__())
        print(f"cMSE (validation): {error_metric(self.y_val, self.predict(self.X_val), self.c_val)}")
        return

    def logs_generic(self, X, y, c, data_type="validation"):
        print(f"{self.__class__.__name__} training took {round(time.time() - self.start_time, 2)} seconds")
        print(self.__repr__())
        print(f"cMSE ({data_type}): {error_metric(y, self.predict(X), c)}")
        return

    def logs_cv(self):
        print(f"{self.__class__.__name__} training took {round(time.time() - self.start_time, 2)} seconds")
        print(self.__repr__())
        print(f"cMSE (validation): {self.best_cMSE_cross_validation}")
        return

    def predict(self, X):
        return self.model.predict(self.std_scaler.transform(X))

    def final_model_evaluation(self, file_name):
        X_test = read_x_test_averaged_nans()
        Y_pred = self.predict(X_test.to_numpy())
        Y_pred = pd.DataFrame(Y_pred, columns=['0'])
        Y_pred.insert(0, 'id', np.arange(0, len(Y_pred)))
        Y_pred.to_csv(f'FinalEvaluations/{file_name}.csv', index=False)
        return

    def find_best_hyperparameters(self):
        """
        Finds the best hyperparameters according to the lowest cMSE using cross validation
        This function assumes the existence of the following in the model:
            self.hyper_parameters_options - a dictionary in which the keys are the hyperparameter names and the values
                                            are lists of the possible values
            self.calculate_cmse_for_kf - a function that builds the model with the given data and calculates the cmse
                use:
                    self.calculate_cmse_for_kf(
                        X_train, y_train, c_train, X_val, y_val, c_val,
                        **hyperparameters
                    )
            self.chosen_hyper_parameters - a dictionary in which the keys are the hyperparameter names and the values
                                            are the soon-to-be chosen values
        :return: nothing
        """
        best_cMSE = float('inf')
        kf = create_kdata_fold()
        for hyperparameters in product_dict(**self.hyper_parameters_options):
            cv_scores = []
            for train_index, val_index in kf.split(self.X_train):
                X_train, X_val = self.X_train[train_index], self.X_train[val_index]
                y_train, y_val = self.y_train[train_index], self.y_train[val_index]
                c_train, c_val = self.c_train[train_index], self.c_train[val_index]

                cMSE = self.calculate_cmse_for_kf(
                    X_train, y_train, c_train, X_val, y_val, c_val,
                    **hyperparameters
                )
                cv_scores.append(cMSE)
                # print(f"Degree {degree} ({model.n_features_in_} features) cMSE: {cMSE}\n")
            average_CMSE = np.mean(cv_scores)
            if average_CMSE < best_cMSE:
                self.chosen_hyper_parameters = hyperparameters
                best_cMSE = average_CMSE
                __inside = ''.join([f'   {k}: {v} \n' for k, v in self.chosen_hyper_parameters.items()])
                # print(f"NEW BEST!: \n{__inside}   Average Cross-Validation Score: {average_CMSE}")
        self.best_cMSE_cross_validation = best_cMSE

class LinearModelTrainTestVal(Model):
    def __init__(self, X_train, y_train, X_val, y_val, c_train, c_val, grad_descent=False):
        super().__init__(X_train, y_train, X_val, y_val, c_train, c_val)
        self.X_train = self.std_scaler.fit_transform(self.X_train)
        self.model = LinearRegression()
        if grad_descent:
            self.gradient_descent(lasso_weight=0.35, Ridge_weight=0.1)
        else:
            self.train()
        self.logs()
        return

    def gradient_descent(self, learning_rate=1, lasso_weight=0.1, Ridge_weight=0.1):
        self.start_time = time.time()
        MAX_ITERATIONS = 1000
        X_1 = np.concatenate([self.X_train, np.ones((self.X_train.shape[0], 1))], axis=1)
        y = self.y_train.reshape(-1, 1)
        c = self.c_train.reshape(-1, 1)
        #weights = np.zeros((X_1.shape[1], 1))
        weights = np.random.rand(X_1.shape[1], 1)
        for i in range(MAX_ITERATIONS):
            y_hat = X_1 @ weights
            reg = lasso_weight * derivative_abs(weights) + Ridge_weight * weights
            grad = derivative_error_metric(y, y_hat, c, X_1) + reg
            weights -= learning_rate * grad

        self.model.coef_ = weights[:-1].reshape(1, -1)
        self.model.intercept_ = weights[-1]
        return

    def __repr__(self):
        return f"LinearModel: y=[X 1]*[{np.round(self.model.coef_[0], 4)} {self.model.intercept_}]"

class PolynomialModel(Model):
    def __init__(self, X_train, y_train, c_train, degrees, train=True):
        super().__init__(X_train, y_train, None, None, c_train, None)
        # Train the model
        self.poly = None
        self.chosen_hyper_parameters = {
            "degree" : -1,
            "l1_lambda" : -1
        }
        self.hyper_parameters_options = {
            "degree" : degrees,
            "l1_lambda" : [10, 1, 0.1, 0] # bigger lambdas seem to have the same result beyond 10 (I tested up to 10^7)
        }
        if train:
            self.train()
            self.logs_cv()
        # self.logs() use .logs_generic() outside
        return

    def __repr__(self):
        return f"PolynomialModel: degree={self.chosen_hyper_parameters['degree']} ({self.model.n_features_in_} features)"

    def get_model(self, l1_lambda):
        return LinearRegression() if l1_lambda == 0 else Lasso(
            alpha=l1_lambda, tol=0.001, selection='random', warm_start=True, max_iter=1_000
        )

    def calculate_cmse_for_kf(self, X_train, y_train, c_train, X_val, y_val, c_val, degree, l1_lambda):
        # TODO should poly be before or after scaler?
        std_scaler = StandardScaler()
        X_train, X_val = self.normalize(X_train, X_val, std_scaler)
        poly = PolynomialFeatures(degree=degree)
        X_train, X_val = self.normalize(X_train, X_val, poly)
        model = self.get_model(l1_lambda)
        model.fit(X_train, y_train)
        # TODO: should model.predict change to model.score?
        # check https://scikit-learn.org/1.5/modules/cross_validation.html#obtaining-predictions-by-cross-validation
        cMSE = error_metric(y_val, model.predict(X_val), c_val)
        return cMSE

    def train(self):
        #outside choosing
        self.find_best_hyperparameters()
        if any(v == -1 for v in self.chosen_hyper_parameters.values()):
            raise Exception("Some hyperparameters not set!")
        self.std_scaler = StandardScaler()
        X_train = self.normalize_train(self.X_train, self.std_scaler)
        self.poly = PolynomialFeatures(degree=self.chosen_hyper_parameters["degree"])
        X_train = self.normalize_train(X_train, self.poly)
        self.model = self.get_model(self.chosen_hyper_parameters["l1_lambda"])
        self.model.fit(X_train, self.y_train)

    def predict(self, X):
        # return super().predict(self.poly.transform(X))
        return self.model.predict(self.poly.transform(self.std_scaler.transform(X)))

class KNNModel(Model):
    def __init__(self, X_train, y_train, c_train, ks, train=True):
        super().__init__(X_train, y_train, None, None, c_train, None)
        self.chosen_hyper_parameters = {"k": -1}
        self.hyper_parameters_options = {"k": ks}
        if train:
            self.train()
            self.logs_cv()
        return

    def __repr__(self):
        return f"KNNModel: k={self.chosen_hyper_parameters['k']}"

    def get_model(self, k):
        return KNeighborsRegressor(n_neighbors=k)

    def calculate_cmse_for_kf(self, X_train, y_train, c_train, X_val, y_val, c_val, k):
        model = self.get_model(k)
        X_train, X_val = self.normalize(X_train, X_val)
        model.fit(X_train, y_train)
        y_pred_val = model.predict(self.std_scaler.transform(X_val))  # automatically normalizes
        return error_metric(y_val, y_pred_val, c_val)

    def train(self):
        #outside choosing
        self.find_best_hyperparameters()
        if any(v == -1 for v in self.chosen_hyper_parameters.values()):
            raise Exception("Some hyperparameters not set!")
        self.std_scaler = StandardScaler()
        X_train = self.normalize_train(self.X_train, self.std_scaler)
        self.model = self.get_model(self.chosen_hyper_parameters["k"])
        self.model.fit(X_train, self.y_train)


class HistGradientBoostingModel(Model):
    def __init__(self, X_train, y_train, c_train):
        super().__init__(X_train, y_train, None, None, c_train, None)
        self.X_train = self.std_scaler.fit_transform(self.X_train)
        self.model = HistGradientBoostingRegressor(l2_regularization=1)
        self.train()
        # self.logs()
        return

    def train(self):
        #X_train = self.normalize_train(self.X_train)
        self.model.fit(self.X_train, self.y_train.ravel())

    def __repr__(self):
        return f"HistGradientBoostingModel: model={self.model}"
        # return f"LinearModel: y=[X 1]*[{np.round(self.model.coef_[0], 4)} {self.model.intercept_}]"

class CatBoostModel(Model):
    def __init__(self, X_train, y_train, c_train, X_val, y_val, c_val):
        super().__init__(X_train, y_train, X_val, y_val, c_train, c_val)
        self.model = None
        # self.logs()
        self.chosen_hyper_parameters = {
            "distname": "Normal",
            "scale": 1.2
        }
        # self.best_cMSE_cross_validation = 351.9508586445804
        self.hyper_parameters_options = {
            "distname": ["Normal", "Logistic", "Extreme"],
            "scale": [1, 1.2, 2]
        }

        # force best but still doing CV
        self.hyper_parameters_options = {
            "distname": ["Normal"],
            "scale": [1.2]
        }

        self.feature_names = [
            "Age", "Gender", "Stage", "GeneticRisk",
            "TreatmentType", "ComorbidityIndex", "TreatmentResponse"
        ]
        self.cat_features = [
            "Gender", "Stage", "TreatmentType",
        ]
        self.train()
        self.logs_cv()

    def get_model(self, distname, scale):
        return CatBoostRegressor(
            iterations=500,
             loss_function=f'SurvivalAft:dist={distname};scale={scale}',
             eval_metric='SurvivalAft',
             verbose=0
        )

    def calculate_cmse_for_kf(
        self, X_train, y_train, c_train, X_val, y_val, c_val,
        distname, scale
    ):
        x_train_frame = self.get_x_treated(self.X_train)
        y_train_frame = self.get_y_treated(self.y_train, self.c_train)
        x_val_frame = self.get_x_treated(self.X_val)
        y_val_frame = self.get_y_treated(self.y_val, self.c_val)
        self.model = self.get_model(distname, scale)
        train_pool = Pool(x_train_frame, label=y_train_frame, cat_features=self.cat_features)
        val_pool = Pool(x_val_frame, label=y_val_frame, cat_features=self.cat_features)
        self.model.fit(train_pool, eval_set=val_pool, verbose=0)
        cmse = error_metric(y_val, self.predict(X_val), c_val)
        print(f"{distname}:{scale} - {cmse}")
        return cmse

    def train(self):
        self.find_best_hyperparameters()
        if any(v == -1 for v in self.chosen_hyper_parameters.values()):
            raise Exception("Some hyperparameters not set!")
        x_train_frame = self.get_x_treated(self.X_train)
        y_train_frame = self.get_y_treated(self.y_train, self.c_train)
        x_val_frame = self.get_x_treated(self.X_val)
        y_val_frame = self.get_y_treated(self.y_val, self.c_val)
        self.model = self.get_model(**self.chosen_hyper_parameters)

        # features = x_train_frame.columns.difference(['y_lower', 'y_upper'], sort=False)
        train_pool = Pool(x_train_frame, label=y_train_frame, cat_features=self.cat_features)
        val_pool = Pool(x_val_frame, label=y_val_frame, cat_features=self.cat_features)
        self.model.fit(train_pool, eval_set=val_pool, verbose=0)

    def get_x_treated(self, X):
        # X_train = self.normalize_train(self.X_train)
        x_frame = pd.DataFrame(X, columns=self.feature_names)
        # x_train_frame.fillna(-1, inplace=True)
        # print(x_train_frame)

        # x_train_frame = x_train_frame.astype({k: int for k in feature_names}, errors='ignore')
        x_frame = x_frame.astype({k: int for k in self.cat_features}, errors='ignore')
        # print(x_frame)
        # print(x_train_frame)
        # x_train_frame = x_train_frame.replace(-1, None)
        return x_frame

    def get_y_treated(self, y, c):
        y_frame = pd.DataFrame(y)
        y_frame2 = y_frame.copy()
        y_frame2['y_upper'] = np.where(pd.DataFrame(c) == 0, y_frame, -1)
        y_frame2['y_lower'] = y_frame
        return y_frame2.loc[:, ['y_lower', 'y_upper']]

    def __repr__(self):
        __inside = ''.join([f'   {k}: {v}' for k, v in self.chosen_hyper_parameters.items()])
        # print(f"NEW BEST!: \n{__inside}   Average Cross-Validation Score: {average_CMSE}")
        return f"CatBoost: model=CatBoostRegressor w/ {__inside}"
        # return f"LinearModel: y=[X 1]*[{np.round(self.model.coef_[0], 4)} {self.model.intercept_}]"

    def predict(self, X):
        return self.model.predict(self.get_x_treated(X))

class ImputerModel(Model):
    def __init__(self, X_train, y_train, c_train, imputer):
        imputer.fit(X_train)
        X_train_im = imputer.transform(X_train)
        super().__init__(X_train_im, y_train, None, None, c_train, None)
        self.X_train = self.std_scaler.fit_transform(self.X_train)
        self.model = LinearRegression()
        self.imputer = imputer
        self.train()
        self.cv_logs()
        return

    def kf_cv(self):
        kf = create_kdata_fold()
        cv_scores = []
        for train_index, val_index in kf.split(self.X_train):
            X_train, X_val = self.X_train[train_index], self.X_train[val_index]
            y_train, y_val = self.y_train[train_index], self.y_train[val_index]
            c_train, c_val = self.c_train[train_index], self.c_train[val_index]

        cv_scores.append(error_metric(y_val, self.predict(X_val), c_val))
        return np.mean(cv_scores)

    def cv_logs(self):
        print(f"{self.__class__.__name__} training took {round(time.time() - self.start_time, 2)} seconds")
        print(self.__repr__())
        print(f"cMSE (kf cross validation): {self.kf_cv()}")
        return

    def final_model_evaluation(self, file_name):
        X_test = read_x_test().to_numpy()
        X_test = self.imputer.transform(X_test)
        Y_pred = self.predict(X_test)
        Y_pred = pd.DataFrame(Y_pred, columns=['0'])
        Y_pred.insert(0, 'id', np.arange(0, len(Y_pred)))
        Y_pred.to_csv(f'FinalEvaluations/{file_name}.csv', index=False)
        return

    def __repr__(self):
        if isinstance(self.imputer, KNNImputer):
            return f"ImputerModel(KNNImputer, k={self.imputer.n_neighbors}): y=[X 1]*[{np.round(self.model.coef_[0], 4)} {self.model.intercept_}]"
        elif isinstance(self.imputer, SimpleImputer):
            return f"ImputerModel({self.imputer.strategy}): y=[X 1]*[{np.round(self.model.coef_[0], 4)} {self.model.intercept_}]"
        elif isinstance(self.imputer, IterativeImputer):
            return f"ImputerModel(IterativeImputer): y=[X 1]*[{np.round(self.model.coef_[0], 4)} {self.model.intercept_}]"
        else:
            pass


# Creates the plots used for task 1.1
def missingValuesAnalysis(df):
    fig, ax = plt.subplots()
    msno.matrix(df, color=(0.2, 0.4, 0.8), ax=ax)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('lightgray')
    plt.savefig("Plots/missing_data_matrix.png", dpi=300, bbox_inches='tight')

    plt.figure(figsize=(12, 6))  # Width=12, Height=6
    msno.bar(df, color=(0.2, 0.4, 0.8), fontsize=12, sort='ascending')  # Blue bars, optional sorting
    plt.savefig("Plots/missing_data_bar.png", dpi=300, bbox_inches='tight')

    msno.heatmap(df)
    plt.savefig("Plots/missing_data_heatmap.png", dpi=300, bbox_inches='tight')

    msno.dendrogram(df, color=(0.2, 0.4, 0.8))
    plt.savefig("Plots/missing_data_dendogram.png", dpi=300, bbox_inches='tight')
    return

# task 1.2
def baseline():
    (X_train, y_train, c_train), (X_val, y_val, c_val), (X_test, y_test, c_test) = read_pruned_dataset()
    model = LinearModelTrainTestVal(X_train, y_train, X_val, y_val, c_train, c_val, grad_descent=False)
    model.final_model_evaluation("baseline-submission-02")
    return

# task 1.3
def grad_descent():
    (X_train, y_train, c_train), (X_val, y_val, c_val), (X_test, y_test, c_test) = read_pruned_dataset()
    model = LinearModelTrainTestVal(X_train, y_train, X_val, y_val, c_train, c_val, grad_descent=True)
    model.final_model_evaluation("cMSE-baseline-submission-02")
    return

# task 2
def nonlinear():
    #For KNNModel
    (X_train, y_train, c_train), (X_test, y_test, c_test) = read_pruned_dataset_train_test_full()
    model = KNNModel(X_train, y_train, c_train, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    #model.logs_generic(X_test, y_test, c_test, "test")
    model.final_model_evaluation("Nonlinear-submission-00")

    #For PolynomialModel
    (X_train, y_train, c_train), (X_test, y_test, c_test) = read_pruned_dataset_train_test_full()
    model = PolynomialModel(X_train,y_train, c_train, degrees=[1,2,3,4,5,6,7,8,9,10])
    #model.logs_generic(X_test, y_test, c_test, "test")
    model.final_model_evaluation("Nonlinear-submission-01")
    return

# task 3.1
def imputation():
    (X_train, y_train, c_train), (X_val, y_val, c_val), (X_test, y_test, c_test) = read_whole_dataset()
    X_train = np.concatenate([X_train, X_val], axis=0)
    y_train = np.concatenate([y_train, y_val], axis=0)
    c_train = np.concatenate([c_train, c_val], axis=0)

    # Impute missing values using the mean
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean", copy=True)
    model = ImputerModel(X_train, y_train, c_train, imputer)
    #model.final_model_evaluation("handle-missing-submission-00")

    # TODO dont forget to mention that median and most frequent gave the same results
    # Impute missing values using the median
    imputer = SimpleImputer(missing_values=np.nan, strategy="median", copy=True)
    model = ImputerModel(X_train, y_train, c_train, imputer)
    #model.final_model_evaluation("handle-missing-submission-01")

    # Impute missing values using the most frequent value
    imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent", copy=True)
    model = ImputerModel(X_train, y_train, c_train, imputer)
    #model.final_model_evaluation("handle-missing-submission-02")

    # Impute missing values using a constant zero
    imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0, copy=True)
    model = ImputerModel(X_train, y_train, c_train, imputer)
    #model.final_model_evaluation("handle-missing-submission-03")

    # Impute missing values using the KNN algorithm
    imputer = KNNImputer(n_neighbors=5, weights="uniform")
    model = ImputerModel(X_train, y_train, c_train, imputer)
    #model.final_model_evaluation("handle-missing-submission-04")

    # Impute missing values using round-robin linear regression
    imputer = IterativeImputer(max_iter=10, random_state=RANDOM_STATE)
    model = ImputerModel(X_train, y_train, c_train, imputer)
    #model.final_model_evaluation("handle-missing-submission-05")

    return

# task 3.2
def boosting():
    (X_train, y_train, c_train), (X_test, y_test, c_test) = read_split_dataset_train_test_full()
    Model3_2 = HistGradientBoostingModel(X_train, y_train, c_train)
    #Model3_2.logs_generic(X_test, y_test, c_test, "test")

    #(X_train, y_train, c_train), (X_test, y_test, c_test) = read_split_dataset_train_test_full()
    #Model3_2_2 = CatBoostModel(X_train,y_train, c_train)
    (X_train, y_train, c_train), (X_val, y_val, c_val), (X_test, y_test, c_test) = read_split_dataset()
    Model3_2_2 = CatBoostModel(X_train,y_train, c_train, X_val, y_val, c_val)
    # Model3_2_2.logs_generic(X_test, y_test, c_test, "test")
    return


def main():
    (X_train, y_train, c_train), (X_val, y_val, c_val), (X_test, y_test, c_test) = read_pruned_dataset()

    boosting()
    #nonlinear()

    return


if __name__ == '__main__':
    main()
