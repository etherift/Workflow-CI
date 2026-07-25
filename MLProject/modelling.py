import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="breast_cancer_preprocessing/train.csv")
    parser.add_argument("--test_data", type=str, default="breast_cancer_preprocessing/test.csv")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=-1)
    parser.add_argument("--min_samples_split", type=int, default=2)
    args = parser.parse_args()

    max_depth = None if args.max_depth == -1 else args.max_depth

    train_df = pd.read_csv(args.train_data)
    test_df = pd.read_csv(args.test_data)

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    mlflow.set_experiment("breast_cancer_ci_retraining")

    with mlflow.start_run(run_name="ci_retrain") as run:
        mlflow.log_params({
            "n_estimators": args.n_estimators,
            "max_depth": max_depth,
            "min_samples_split": args.min_samples_split,
        })

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=max_depth,
            min_samples_split=args.min_samples_split,
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

        mlflow.sklearn.log_model(model, artifact_path="model")

        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)

        print(f"Run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
