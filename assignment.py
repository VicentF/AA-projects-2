import time
import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor

# cMSE, the new metric for evaluation
def error_metric(y, y_hat, c):
    err = y-y_hat
    err = (1-c)*err**2 + c*np.maximum(0,err)**2
    return np.sum(err)/err.shape[0]

# models
class Model:
    def __init__(self, X_train, Y_train, X_val, Y_val):
        self.start_time = time.time()
        self.std_scaler = StandardScaler()
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.model = None
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
        print(std_scaler)
        X_train = std_scaler.fit_transform(X_train)
        X_val = std_scaler.transform(X_val)
        return X_train, X_val

    def train(self):
        if self.model is None:
            raise Exception("Unassigned model. Shouldn't be none!")
        self.model.fit(self.X_train, self.Y_train)
        return

    def logs(self):
        print(f"{self.__class__.__name__} training took {round(time.time() - self.start_time, 2)} seconds")
        print(self.__repr__())
        print(f"cMSE (validation): {error_metric(self.Y_val, self.predict(self.X_val), 0)}")

        return

    def predict(self, X):
        return self.model.predict(self.std_scaler.transform(X))

    def final_model_evaluation(self, file_name=""):
        if file_name == "":
            file_name = self.__class__.__name__
        X_test = pd.read_csv('Datasets/test_data.csv')
        #X_test = X_test.dropna()
        Y_pred = self.predict(X_test)
        Y_pred = pd.DataFrame(Y_pred, columns=['0'])
        Y_pred.insert(0, 'id', np.arange(0, len(Y_pred)))
        Y_pred.to_csv(f'FinalEvaluations/{file_name}.csv', index=False)
        return

class LinearModelTrainTestVal(Model):
    def __init__(self, X_train, Y_train, X_val, Y_val):
        super().__init__(X_train, Y_train, X_val, Y_val)
        self.X_train = self.std_scaler.fit_transform(X_train)
        self.model = LinearRegression()
        self.train()
        self.logs()
        return

    def __repr__(self):
        return

class LinearModelCV(Model):
    def __init__(self, X_train, Y_train, kf):
        super().__init__(X_train, Y_train, None, None)
        self.kf = kf  # K-Fold cross-validator
        self.X_train = self.normalize(X_train)  # Normalize the data
        self.model = LinearRegression()  # Initialize the Linear Regression model
        self.train()
        self.logs()
        return

    def __repr__(self):
        return "LinearModelCV"

    def normalize(self, X_train, std_scaler=None):
        if std_scaler is None:
            std_scaler = self.std_scaler
        X_train = std_scaler.fit_transform(X_train)
        return X_train

    def train(self):
        self.cv_scores = []
        self.Y_train = self.Y_train.reset_index(drop=True)
        for train_index, val_index in self.kf.split(self.X_train):
            X_cv_train, X_cv_val = self.X_train[train_index], self.X_train[val_index]
            Y_cv_train, Y_cv_val = self.Y_train[train_index], self.Y_train[val_index]
            self.model.fit(X_cv_train, Y_cv_train)
            predictions = self.model.predict(X_cv_val)
            print(X_cv_val)
            score = error_metric(Y_cv_val, predictions, 0)
            self.cv_scores.append(score)

    def logs(self):
        print(f"{self.__class__.__name__} training took {round(time.time() - self.start_time, 2)} seconds")
        print(self.__repr__())

        print("Cross-Validation Results:")
        for i, score in enumerate(self.cv_scores):
            print(f"Fold {i + 1} Score: {score}")
        print(f"Average Cross-Validation Score: {np.mean(self.cv_scores)}")

    def final_model_evaluation(self, file_name=""):
        super().final_model_evaluation(file_name)


class PolynomialModel(Model):
    def __init__(self, X_train, Y_train, X_val, Y_val, degrees):
        super().__init__(X_train, Y_train, X_val, Y_val)
        self.degrees = degrees
        # Train the model
        self.poly = None
        self.best_degree = -1
        self.train()
        self.logs()
        return

    def __repr__(self):
        return f"PolynomialModel: degree={self.best_degree} ({self.model.n_features_in_} features)"

    def train(self):
        best_cMSE = float('inf')
        for degree in self.degrees:
            poly = PolynomialFeatures(degree=degree)
            X_train, X_val = self.normalize(self.X_train, self.X_val, poly)
            std_scaler = StandardScaler()
            X_train, X_val = self.normalize(X_train, X_val, std_scaler)
            model = LinearRegression()
            model.fit(X_train, self.Y_train)
            cMSE = error_metric(model.predict(X_val), self.Y_val, 0)
            print(f"Degree {degree} ({model.n_features_in_} features) cMSE: {cMSE}\n")
            if cMSE < best_cMSE:
                best_cMSE = cMSE
                self.model = model
                self.best_degree = degree
                self.poly = poly
                self.std_scaler = std_scaler
        return

    def predict(self, X):
        return super().predict(self.poly.transform(X))


class KNNModel(Model):
    def __init__(self, X_train, Y_train, X_val, Y_val, ks):
        super().__init__(X_train, Y_train, X_val, Y_val)
        self.X_train = self.std_scaler.fit_transform(X_train)  # stores validation without normalization
        self.ks = ks
        self.knn = None
        self.best_k = -1
        self.train()
        self.logs()
        return

    def __repr__(self):
        return f"KNNModel: k={self.best_k}"

    def train(self):
        best_cMSE = float('inf')
        for k in self.ks:
            knn = KNeighborsRegressor(n_neighbors=k)
            model = knn
            model.fit(self.X_train, self.Y_train)
            y_pred_val = model.predict(self.std_scaler.transform(self.X_val))   # automatically normalizes
            cMSE = error_metric(y_pred_val, self.Y_val,0)
            print(f"KNN k={k} cMSE: {cMSE}\n")
            if cMSE < best_cMSE:
                best_cMSE = cMSE
                self.model = model
                self.knn = knn
                self.best_k = k
        return

# Creates a train,validation and test split
def train_val_test_split(X, Y, test_size=0.1, val_size=0.1, random_state=42):
    # First split: Train + Validation vs Test
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )

    # Second split: Train vs Validation
    val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size relative to remaining data
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_val, Y_train_val, test_size=val_size_adjusted, random_state=random_state
    )

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

# Creates a train, cross-validation and test split
def train_cv_test_split(X, Y, test_size=0.1, n_splits=5, random_state=42):
    # Split into Train + Test
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )

    # Create K-Fold cross-validator for the training set
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    return X_train, Y_train, X_test, Y_test, kf

# Creates our X and Y after dropping missing values & the censored data
def dropMissingCensored(df):
    df_cleaned = df.dropna().loc[df['Censored'] != 1]

    #Pair-Plot of the new df and the target variable "SurvivalTime", comment out if not needed
    #sns.pairplot(df_cleaned,hue = 'SurvivalTime', diag_kind='kde')
    #plt.savefig("Plots/after_drop_pairplot_survival_time_V.png", dpi=300, bbox_inches='tight')
    #plt.show()

    Y = df_cleaned['SurvivalTime']
    X = df_cleaned.drop(columns=['SurvivalTime', 'Censored']).rename(columns = {'Unnamed: 0':'id'})
    df = df.rename(columns={'old_column_name': 'new_column_name'})


    return X, Y

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


def main():

    df = pd.read_csv('Datasets/train_data.csv')
    X, Y = dropMissingCensored(df)

    #For LinearModel with train_val_test
    X_train, Y_train, X_val, Y_val, X_test, Y_test = train_val_test_split(X, Y)
    Model1 = LinearModelTrainTestVal(X_train,Y_train,X_val,Y_val)

    #For LinearModel with CrossValidation
    #X_train, Y_train, X_val, Y_val, kf = train_cv_test_split(X,Y)
    #Model2 = LinearModelCV(X_train, Y_train, kf)

    #For PolynominalModel
    X_train, Y_train, X_val, Y_val, X_test, Y_test = train_val_test_split(X, Y)
    Model3 = PolynomialModel(X_train,Y_train,X_val,Y_val, [1,2,3,4,5,6,7,8,9,10])

    #For KNNModel
    X_train, Y_train, X_val, Y_val, X_test, Y_test = train_val_test_split(X, Y)
    Model4 = KNNModel(X_train,Y_train,X_val,Y_val, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

    #Model.final_model_evaluation("baseline-submission-00")
    return


if __name__ == '__main__':
    main()
