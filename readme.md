
#Cajamar 2022 Foot print


Para la competición, comentar que el script de predicción sería utilzar el main.py y el script de análisis serían los notebooks de la carpeta notebooks, los cuales fueron creados a partir de colab.
Elementos a destacar:

* En la carpeta conf\local se encuentran las variables de entorno (en nuestro caso solo PYTHONPATH)
* En la carpeta data se sigue la estructura recomendada en la convención de la ingeniería de datos (más info https://kedro.readthedocs.io/en/stable/12_faq/01_faq.html) diviendolo dentro de este grupo entre datos/modelos diarios o semanales.
* En la carpeta notebooks se tienen los scripts de análisis, recomendando principalmente el de análisis exploratorio ya que el second análisis fue un testeo de pruebas y no se encuentra muy ordenado
* En la carpeta src se encuentra prácticamente todo el código necesario para la ejecución del programa, y utilizando main debería de funcionar (queda pendiente la normalización de las rutas mediante algunas variables de entorno, pero esto se hará en un futuro trabajo). Los archivos presentes son
    * analysis, el cual se utilizó para analizar las variables con PCA
    * create_features, donde se extraen las características de un dataset limpio y agrupado
    * data_transformations, deprecated
    * generate_df, donde se crea un dataset limpio y agrupado
    * main, donde se llaman a todas las partes necesarias del código
    * pipeline, archivo para ordenar distintos pasos, principalmente usado en la creación de modelos
    * predict, utilizado para realizar las predicciones
    * pycaret_attributes, archivo que sirve de interfaz para cambiar atributos de algunos modelos
    * system_to_train, archivo donde se realiza el entrenamiento de los modelos
    * utils, funciones utilizadas a lo largo de todos los archivos
    
Además el environment utilizado se puede encontrar en environment.yml

TODO:
Destacar que el código es mejorable, sobretodo en el apartado de la creación de funcion entre predict y system_to_train ya que comparten mucho código pero por tiempo no se ha podido realizar
Reducir el numero de try y excepciones que solo se pusieron para debuguear fácilmente