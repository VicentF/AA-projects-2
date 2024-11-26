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
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from DatasetManipulator import read_pruned_dataset, read_x_test_averaged_nans, create_kdata_fold, \
    read_pruned_dataset_train_test_full


# cMSE, the new metric for evaluation
def error_metric(y, y_hat, c):
    err = y_hat - y
    err = (1-c)*err**2 + c*np.minimum(0, err)**2
    return np.sum(err)/err.shape[0]

def derivative_error_metric(y, y_hat, c, X):
    def derivative_min(e):
        # return 0 if e < 0 else 1
        return (e <= 0).astype(float)
    err = y_hat - y
    err2 = ((1-c) * err) + (c * np.minimum(0, err) * derivative_min(err))
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
                print(f"NEW BEST!: \n{__inside}   Average Cross-Validation Score: {average_CMSE}")

class LinearModelTrainTestVal(Model):
    def __init__(self, X_train, y_train, X_val, y_val, c_train, c_val, grad_descent=False):
        super().__init__(X_train, y_train, X_val, y_val, c_train, c_val)
        self.X_train = self.std_scaler.fit_transform(self.X_train)
        self.model = LinearRegression()
        if grad_descent:
            self.gradient_descent()
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


class LinearModelCV(Model):
    def __init__(self, X_train, y_train, c_train, c_val):
        super().__init__(X_train, y_train, None, None, c_train, None)
        self.kf = create_kdata_fold()  # K-Fold cross-validator
        self.X_train = self.std_scaler.fit_transform(self.X_train)  # Normalize the data
        self.model = LinearRegression()  # Initialize the Linear Regression model
        self.cv_scores = []
        self.train()
        self.logs()
        return

    def __repr__(self):
        return f"LinearModelCV: y=mx+b, m={round(self.model.coef_[1], 2)}, b={round(self.model.coef_[0], 2)}"

    def train(self):
        self.y_train = self.y_train.reset_index(drop=True)
        for train_index, val_index in self.kf.split(self.X_train):
            X_cv_train, X_cv_val = self.X_train[train_index], self.X_train[val_index]
            Y_cv_train, Y_cv_val = self.y_train[train_index], self.y_train[val_index]
            self.model.fit(X_cv_train, Y_cv_train)
            predictions = self.model.predict(X_cv_val)
            score = error_metric(Y_cv_val, predictions, 0)
            self.cv_scores.append(score)
        return

    def logs(self):
        print(f"{self.__class__.__name__} training took {round(time.time() - self.start_time, 2)} seconds")
        print(self.__repr__())

        print("Cross-Validation Results:")
        for i, score in enumerate(self.cv_scores):
            print(f"Fold {i + 1} Score: {score}")
        print(f"Average Cross-Validation Score: {np.mean(self.cv_scores)}")


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
        # self.logs() use .logs_generic() outside
        return

    def __repr__(self):
        return f"PolynomialModel: degree={self.chosen_hyper_parameters['degree']} ({self.model.n_features_in_} features)"

    def get_model(self, l1_lambda):
        return LinearRegression() if l1_lambda == 0 else Lasso(
            alpha=l1_lambda, tol=0.001, selection='random'
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
        return

    def __repr__(self):
        return f"KNNModel: k={self.chosen_hyper_parameters['k']}"

    def get_model(self, k):
        return KNeighborsRegressor(n_neighbors=k)

    def calculate_cmse_for_kf(self, X_train, y_train, c_train, X_val, y_val, c_val, k):
        model = KNeighborsRegressor(n_neighbors=k)
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

"""
def get_scores_for_imputer(imputer, X_missing, y_missing):
    estimator = make_pipeline(imputer, regressor)
    impute_scores = cross_val_score(
        estimator, X_missing, y_missing, scoring="neg_mean_squared_error", cv=N_SPLITS
    )
    return impute_scores


def imbuteValues(df):
    Y_Miss = df['SurvivalTime']
    X_Miss = df.drop(columns = ['SurvivalTime'])

    X_Non_Miss, Y_Non_Miss = dropMissingCensored(df)

    full_scores = cross_val_score(regressor, X_Miss, Y_Non_Miss, scoring=make_scorer(error_metric, X_Miss['Censored']), cv=N_SPLITS)

    N_SPLITS = 4
    regressor = RandomForestRegressor(random_state=0)
    X_labes = []

    cmses = np.zeros(5)
    stds = np.zeros(5)



    return X_Miss,Y_Miss
"""


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

# temporary function
def eval_model(model, X_test, y_test, c_test):
    return error_metric(y_test, model.predict(X_test), c_test)

def main():
    # (X_train, y_train, c_train), (X_val, y_val, c_val), (X_test, y_test, c_test) = read_pruned_dataset()

    #model = LinearModelTrainTestVal(X_train, y_train, X_val, y_val, c_train, c_val, grad_descent=True)
    #model.gradient_descent()
    #model.final_model_evaluation("cMSE-baseline-submission-01")

    #For PolynomialModel
    #(X_train, y_train, c_train), (X_test, y_test, c_test) = read_pruned_dataset_train_test_full()
    #Model3 = PolynomialModel(X_train,y_train, c_train, degrees=[1,2,3,4,5,6,7,8,9,10])
    #Model3.logs_generic(X_test, y_test, c_test, "test")
    #Model3.final_model_evaluation("Nonlinear-submission-02")


    #For KNNModel
    #(X_train, y_train, c_train), (X_test, y_test, c_test) = read_pruned_dataset_train_test_full()
    #Model4 = KNNModel(X_train,y_train, c_train,  [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    #Model4.logs_generic(X_test, y_test, c_test, "test")
    #Model4.final_model_evaluation("Nonlinear-submission-02")

    return


if __name__ == '__main__':
    main()
