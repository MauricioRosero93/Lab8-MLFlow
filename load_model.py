import mlflow
import numpy as np
import pandas as pd  # Necesario para la conversión

# Configuración del servidor MLFlow
uri = "http://127.0.0.1:6001"
mlflow.set_tracking_uri(uri=uri)

# Obtener el URI del modelo registrado (debes reemplazarlo con el tuyo)
# Este URI lo encuentras en la interfaz web de MLFlow
logged_model = "models:/digits-classifier/latest"

# Cargar el modelo
loaded_model = mlflow.sklearn.load_model(logged_model)

# Crear datos de prueba
np.random.seed(42)
data = np.random.rand(1, 64)

# Convertir a DataFrame y predecir
prediction = loaded_model.predict(pd.DataFrame(data))

# Mostrar resultado
print(prediction)
