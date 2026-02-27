import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay, precision_score, recall_score

def log_experiment_run(model, X_train, Y_train, X_dev, Y_dev, run_name, params, gesture_labels):
    
    with mlflow.start_run(run_name=run_name):
       
        mlflow.log_params(params)
        
        
        y_pred = model.predict(X_dev)
        acc = accuracy_score(Y_dev, y_pred)
        f1 = f1_score(Y_dev, y_pred, average='macro')
        precision = precision_score(Y_dev, y_pred, average='macro')
        recall = recall_score(Y_dev, y_pred, average='macro')
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        

        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay.from_predictions(Y_dev, y_pred, display_labels=gesture_labels, cmap='Blues', ax=ax)

        plt.title(f"Confusion Matrix: {run_name}")
        plt.xticks(rotation=45)
        
        plot_path = f"{run_name}_Confusion_Matrix.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close(fig)
        
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model", signature=signature)
        
        return acc, f1