import mlflow.sklearn
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SEED = 42

mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment(experiment_name='Boston Housing Regression')
tags = {"dataset": "Boston",
        }


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    boston_data = datasets.load_boston()
    print("Loaded boston dataset")
    X_train, X_test, y_train, y_test = train_test_split(boston_data.data, boston_data.target, random_state=SEED)

    with mlflow.start_run(run_name='Sk_Linear_Regression'):
        mlflow.set_tags(tags)
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_test)

        (rmse, mae, r2) = eval_metrics(y_test, y_pred)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr, "model")
        mlflow.log_artifact(local_path='./mlflow_project/train.py', artifact_path='code')
    pass