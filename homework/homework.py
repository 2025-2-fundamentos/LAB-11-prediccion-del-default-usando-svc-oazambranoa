# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#


from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

import os
import json
import gzip
import pickle
import pandas as pd

def cargarDatos(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=False, compression="zip")


def limpiarDatos(df: pd.DataFrame) -> pd.DataFrame:
    dfCopy = df.copy()
    dfCopy = dfCopy.rename(columns={"default payment next month": "default"})
    dfCopy = dfCopy.drop(columns=["ID"])
    dfCopy = dfCopy.loc[dfCopy["MARRIAGE"] != 0]
    dfCopy = dfCopy.loc[dfCopy["EDUCATION"] != 0]
    dfCopy["EDUCATION"] = dfCopy["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
    return dfCopy


def crearPipeline(xTrain: pd.DataFrame) -> Pipeline:
    catFeatures = ["SEX", "EDUCATION", "MARRIAGE"]
    numFeatures = [col for col in xTrain.columns if col not in catFeatures]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(), catFeatures),
            ("scaler", StandardScaler(), numFeatures),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("pca", PCA()),
            ("featureSelection", SelectKBest(score_func=f_classif)),
            ("classifier", SVC(kernel="rbf", random_state=12345, max_iter=-1)),
        ]
    )

    return pipeline


def crearEstimador(pipeline: Pipeline, xTrain: pd.DataFrame) -> GridSearchCV:
    paramGrid = {
        "pca__n_components": [20, xTrain.shape[1] - 2],
        "featureSelection__k": [12],
        "classifier__kernel": ["rbf"],
        "classifier__gamma": [0.1],
    }

    estimator = GridSearchCV(
        estimator=pipeline,
        param_grid=paramGrid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
    )

    return estimator


def guardarModelo(model) -> None:
    os.makedirs("files/models", exist_ok=True)
    modelPath = "files/models/model.pkl.gz"
    with gzip.open(modelPath, "wb") as file:
        pickle.dump(model, file)


def guardarMetricas(model, xTrain, xTest, yTrain, yTest) -> None:
    yTrainPred = model.predict(xTrain)
    yTestPred = model.predict(xTest)

    trainMetrics = {
        "type": "metrics",
        "dataset": "train",
        "precision": precision_score(yTrain, yTrainPred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(yTrain, yTrainPred),
        "recall": recall_score(yTrain, yTrainPred, zero_division=0),
        "f1_score": f1_score(yTrain, yTrainPred, zero_division=0),
    }

    testMetrics = {
        "type": "metrics",
        "dataset": "test",
        "precision": precision_score(yTest, yTestPred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(yTest, yTestPred),
        "recall": recall_score(yTest, yTestPred, zero_division=0),
        "f1_score": f1_score(yTest, yTestPred, zero_division=0),
    }

    os.makedirs("files/output", exist_ok=True)
    outputPath = "files/output/metrics.json"
    with open(outputPath, "w", encoding="utf-8") as f:
        f.write(json.dumps(trainMetrics) + "\n")
        f.write(json.dumps(testMetrics) + "\n")


def guardarMatricesDeConfusion(model, xTrain, xTest, yTrain, yTest) -> None:
    yTrainPred = model.predict(xTrain)
    yTestPred = model.predict(xTest)

    cmTrain = confusion_matrix(yTrain, yTrainPred)
    cmTest = confusion_matrix(yTest, yTestPred)

    def formatoCm(cm, datasetName: str) -> dict:
        return {
            "type": "cm_matrix",
            "dataset": datasetName,
            "true_0": {
                "predicted_0": int(cm[0, 0]),
                "predicted_1": int(cm[0, 1]),
            },
            "true_1": {
                "predicted_0": int(cm[1, 0]),
                "predicted_1": int(cm[1, 1]),
            },
        }

    rows = [
        formatoCm(cmTrain, "train"),
        formatoCm(cmTest, "test"),
    ]

    outputPath = "files/output/metrics.json"
    with open(outputPath, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def ejecutar() -> None:
    testDf = limpiarDatos(cargarDatos("files/input/test_data.csv.zip"))
    trainDf = limpiarDatos(cargarDatos("files/input/train_data.csv.zip"))

    xTrain = trainDf.drop(columns="default")
    yTrain = trainDf["default"]

    xTest = testDf.drop(columns="default")
    yTest = testDf["default"]

    pipeline = crearPipeline(xTrain)
    model = crearEstimador(pipeline, xTrain)
    model.fit(xTrain, yTrain)

    guardarModelo(model)
    guardarMetricas(model, xTrain, xTest, yTrain, yTest)
    guardarMatricesDeConfusion(model, xTrain, xTest, yTrain, yTest)


if __name__ == "__main__":
    ejecutar()
