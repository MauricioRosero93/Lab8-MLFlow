import mlflow
from mlflow.models import infer_signature
from utility import pipeline

# Configuración del servidor MLFlow
uri = "http://127.0.0.1:6001"
mlflow.set_tracking_uri(uri)

# Nombre del experimento
email = "mauricio.rosero.h@gmail.com" 
experiment_name = f"{email}-lab8-V2"
mlflow.set_experiment(experiment_name)

# Generar datos
X_train, X_test, y_train, y_test = pipeline.data_preprocessing()

# Parámetros del modelo
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

# Entrenar modelo
trained_model = pipeline.train_logistic_regression(X_train, y_train, params)
accuracy = pipeline.evaluation(trained_model, X_test, y_test)  

# Registrar en MLFlow
with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.set_tag("Training Info", "Basic LR model for digits data")
    
    signature = infer_signature(X_train, trained_model.predict(X_train))
    
    model_info = mlflow.sklearn.log_model(
        sk_model=trained_model,
        artifact_path="digits_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="digits-classifier",
    )