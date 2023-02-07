# Application of Deep Learning CNN Models for the Detection and Classification of Melanoma Cancer

## **<div align="center">Proyecto Integrador - USFQ</div>**
## *<div align="center">Ingeniería en Ciencias de la Computación</div>*


### **Authors:** José M. Cadena, Noel Pérez PhD.

---

| File | Description |
| ----------- | ----------- |
| cnn-1.py | Modelo CNN sencillo. Dos bloques convolucionales, 2 conv layer, 1 max-pool, 1 dropout. Fully Connected Layer de 512, 128 y 1 neurona. |
| cnn-2.py | Modelo CNN complejo. Cuatro bloques convolucionales, 2 conv layer, 1 max-pool, 1 dropout. Fully Connected Layer de 1024 y 1 neurona. |
| test_run.py | Archivo que toma un modelo entrenado y realiza predicciones con un test set. |
| utils.py | Funciones auxiliares para la realización de plots en función de los datos registrados. Estos se encuentran en Histories y Predictions. |

---

Cada archivo tiene su método ```if __name__=="__main__":``` para ejecutar el respectivo **script**. Revisar en las constantes de cada archivo los *paths* para que funcione. 
