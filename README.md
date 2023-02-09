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

El dataset utilizado puede ser visto en el siguiente link: 
[melanoma-classification-dataset](https://estudusfqedu-my.sharepoint.com/:f:/g/personal/jmcadenaz_estud_usfq_edu_ec/Engte5NXd3JGq_ELC8CLPjoBiCB_NVL_zOPtgD0uFYeD7g?e=bFQ6dw). El dataset se divide en dos carpetas, ```train``` y ```test```. Dentro de cada una de estas, se encuentran las fotografías y un archivo llamado ```labels.csv```, el cual se lee como un dataframe para los generadores de Keras. 

*Importante: Al descargar el dataset y correr el proyecto, actualizar los paths correspondientes.*
