import mlflow
from mlflow.models import infer_signature
from utility import pipeline

# Configuración del servidor (igual que antes)
mlflow.set_tracking_uri("http://127.0.0.1:6001")
mlflow.set_experiment("mauricio.rosero.h@gmail.com-lab8-V3")  

# Cargar datos
X_train, X_test, y_train, y_test = pipeline.data_preprocessing()

# Nuevos parámetros para Random Forest
params = {
    "n_estimators": 150,
    "max_depth": 10,
    "random_state": 42,
    "class_weight": "balanced"  
}

# Entrenar modelo 
trained_model = pipeline.train_random_forest(X_train, y_train, params)
accuracy = pipeline.evaluation(trained_model, X_test, y_test)

# Registrar en MLFlow con metadata mejorada
with mlflow.start_run(run_name="Random Forest v1"):
    # 1. Log de parámetros y métricas
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)
    
    mlflow.set_tags({
        "Training Info": "Random Forest para clasificación de dígitos",
        "Algorithm": "RandomForestClassifier",
        "Data": "Digits dataset (sklearn)",
        "Purpose": "Demo MLFlow Lab8 - RF"
    })
    
    # Descripción del modelo
    mlflow.log_text(
        "Modelo Random Forest con 150 árboles y profundidad máxima 10. "
        "Precisión mejorada.",
        "model_description.txt"
    )
    
    # Registrar el modelo (mismo nombre, nueva versión)
    signature = infer_signature(X_train, trained_model.predict(X_train))
    model_info = mlflow.sklearn.log_model(
        sk_model=trained_model,
        artifact_path="digits_rf_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="digits-classifier"  
    )