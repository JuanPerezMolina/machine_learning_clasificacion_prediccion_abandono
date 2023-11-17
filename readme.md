<h1 align='center'>
 <b>Machine Learning: clasificación para predicción de abandono</b>
</h1>


## **Algoritmos de clasificación para Predicción de abandono**


## Table of Contents
1. [General Info](#general-info)
2. [Technologies](#technologies)
3. [Installation](#installation)
4. [Collaboration](#collaboration)
5. [FAQs](#faqs)
### General Info
***
El objetivo de este proyecto es aplicar aprendizaje automatizado para predecir si un cliente va a abandonar o no los servicios de una empresa partiendo del análisis de los registros del dataset de todos los clientes de la empresa utilizando los algoritmos de clasificación: K-Vecinos más cercanos, Bernoulli Naïve Bayes y Árbol de Decisión. 

<b>Objetivos:</b> 
- Entender la aplicación de algoritmos de clasificación
- ¿Que algoritmo de clasificación es más preciso?
- Importancia de la estratificación proporcional de las muestras de entrada
- La importancia de aplicar un algoritmo de Support Vector Machines con datos estandarizados.
- Entienda la importancia de usar un estimador no lineal
- La importancia de aplicar un algoritmo de arbol de decisión sin datos estandarizados y combinarlo con Dummy Classifiers.

### Algoritmos de Machine Learning

Existen dos categorías de problemas que pueden ser bien resueltos con la implementación de Machine Learning: Los de clasificación y los de regresión.

**Clasificación**

Cuando necesitamos prever a cuál categoría pertenece una determinada muestra, se trata de un problema de clasificación. Algunos ejemplos que podemos citar son los siguientes:

Prever si un(a) determinado(a) paciente tiene Covid.
Si un(a) cliente es propenso(a) a desistir de la compra.
Si algún usuario web es propenso a hacer clic en un anuncio.
En estos casos mencionados, la previsión se concentra en 0 o 1 (Covid/no Covid, desistir/no desistir, hacer clic/no hacer clic) que es denominada como clasificación binaria, en la cual existen solamente dos clases. Hay también casos en los que la clasificación se da con más de dos clases, conocidas como clasificación multiclase, como el filtrado de los e-mails en las siguientes categorías “principal”, “social”, “promociones”, “importantes” o “foros”.

Entre los algoritmos de clasificación más conocidos, podemos citar los siguientes:

- K-Nearest Neighbors (KNN);
- Support Vector Machine (SVM);
- Decision Tree Classifier;
- Random Forest Classifier.


**Regresión**

Cuando necesitamos prever un valor numérico específico, esto indica que estamos lidiando con un problema de regresión. Algunos ejemplos de estos problemas están relacionados con la previsión de:

- Precios/costos futuros;
- Inventario;
- Ingresos futuros.

En estas situaciones, podemos utilizar algún modelo de regresión para realizar estas previsiones y presentar como respuesta algún valor continuo relacionado al problema. Existen diferentes tipos de algoritmos de Machine Learning utilizados para resolver este tipo de problemas:

- Linear Regression;
- Random Forest Regressor;
- Support Vector Regression (SVR).

### División de los datos para entrenamiento y prueba
Separar el conjunto de datos entre entrenamiento y prueba es una parte importante para lograr medir el desempeño real de un modelo de Machine Learning. Esta etapa consiste en dedicar parte del conjunto de datos para entrenar el modelo y la otra parte para pruebas

**Datos de entrenamiento**

Los datos de entrenamiento son aquellos utilizados para la creación y entrenamiento del modelo. Normalmente, la mayoría de los datos, cerca del 70%, son utilizados para entrenamiento.

**Datos de prueba**

Los datos de prueba son utilizados para comprobar que el modelo realmente funciona. Estos no son utilizados en el entrenamiento del modelo y normalmente representan el 30% de la totalidad de los datos.

Además de ello, al momento de realizar la separación de estos datos es importante que esta sea hecha de forma aleatoria, para garantizar que no habrá ningún patrón al momento de la división de los datos. Así, cada muestra tendrá la misma probabilidad de ser seleccionada.

### Probabilidad condicional y Bayes
El teorema de Bayes es una fórmula utilizada para calcular la probabilidad de ocurrencia de un evento sabiendo que otro evento, conocido como condicionante, ya ocurrió, denominado probabilidad condicional. Su notação es dada por P(A|B), que significa la probabilidad de A dado que B ya ocurrió y está definido por la siguiente ecuación:
$$P(A|B) = {P(B|A) * P(A) \over P(B)} $$
Donde:

P(B|A): probabilidad de B ocurrir dado que A ya ocurrió;
P(A): probabilidad de A ocurrir;
P(B): probabilidad de B ocurrir.

Hablando de esta forma, puede ser complicado de visualizar en la práctica lo que esto realmente significa. Como el teorema de Bayes involucra la probabilidad, este puede ser aplicado a un sinnúmero de contextos. Vamos a utilizar aquí un ejemplo que tiene que ver con el área de la salud, aunque el teorema podría ser utilizado en otras áreas, también. Considera el siguiente ejemplo:

|  Test         | Covid-19 (5%) | Sin Covid-19 (95%) |
|:--------------|:-------------:|--------------:|
| Test positivo | 85% | 10% |
| Test Negativo | 15% | 90% |

Primeramente, vamos a entender lo que está en la tabla. Sabemos, a través de investigaciones realizadas anteriormente, que hay una probabilidad del 5% de tener Covid-19. Consecuentemente, hay una probabilidad del 95% de no tener. Quien padece la enfermedad está en la columna “Covid-19 (5%)” y posee una probabilidad del 85% de obtener un resultado positivo y 15% de uno negativo. Ahora bien, quien no padece la enfermedad está en la columna “Sin Covid-19 (95%)” y posee una probabilidad del 10% de obtener un resultado positivo y 90% de uno negativo.

Con esta información, vamos a responder a la pregunta: ¿Cuál es el chance de tener Covid-19 dado que el test resultó positivo?

- Evento A: Tener Covid-19

- Evento B: Resultado Positivo del test (evento condicionante)

Utilizando la fórmula presentada anteriormente, necesitamos definir algunas probabilidades, como:

P(B|A): probabilidad de obtener un resultado positivo del test dado que la persona tiene Covid-19, que es del 85% o 0.85, de acuerdo con la tabla.
P(A): probabilidad de tener covid-19. Observando la tabla, tenemos que es del 5%, o 0.05.
P(B): la probabilidad de obtener un resultado positivo del test.
La P(B) no la logramos encontrar directamente en la tabla, pues la probabilidad del test ser positivo puede ocurrir cuando la persona que realiza el test posee o no la enfermedad (conocido también como complementar). Luego, la probabilidad P(B) puede ser calculada como se muestra a continuación:

$$P(B) = {P(B|A) * P(A) + P(B|A^c) * P(A^c)} $$

- P(B|AC): Probabilidad de que el test resulte positivo dado que la persona no padece Covid-19, que es del 10% o 0.1
- P(AC): Probabilidad de que la persona no tenga Covid-19, que es del 95% o 0.95.

Una vez más, la información anterior fue retirada directamente de la tabla. ¡Ya está! Ahora tenemos toda la información que necesitamos. Sustituyendo en la ecuación, tenemos que:
$$P(B) = {0.85 * 0.05 + 0.1 * 0.95} = 0.1375 $$

Y reemplazando este valor en la primera ecuación, tenemos que:
$$P(A|B) = {P(B|A) * P(A) \over P(B)} = {0.85 * 0.05 \over 0.1375} = 0.31$$

Entonces, la probabilidad de tener Covid-19, dado que el test resultó positivo, es del 31%


### Arbol de decisión
El árbol de decisión es uno de los modelos de previsión más sencillos, inspirado en la forma como los seres humanos tomamos las decisiones y posee una alta interpretabilidad, o sea, una comprensión fácil de los pasos que fueron realizados para lograr alcanzar el resultado final. Estos árboles pueden ser utilizados tanto para modelos de regresión, que tienen el objetivo de prever valores numéricos, como para modelos de clasificación, que tienen el objetivo de prever categorías. Una de las principales desventajas del uso del algoritmo del arbol de decisiones es el sobreajuste (overfitting) a los datos de entrenamiento debidoa la profundidad (cantidad de subniveles verticales) de los nodos del arbol.

**Criterio de división de los nodos**

Para lograr identificar cuál es el mejor momento en que un nodo debe ser dividido en dos o más subnodos, el algoritmo del árbol de decisión considera algunos criterios. Los dos principales criterios de división empleados en los árboles de decisión son:

**Índice Gini:**

Este índice informa el grado de heterogeneidad de los datos. El objetivo de él es medir la frecuencia de que un elemento aleatorio de un nodo sea rotulado de manera incorrecta. En otras palabras, este índice es capaz de medir la impureza de un nodo y es determinado por medio del siguiente cálculo:
$$Gini = 1 - {\sum_{i=1}^kP(i)^2|}$$

Donde:
- P(i) representa la frecuencia relativa de las clases en cada uno de los nodos;
- k es el número de clases.
Si el índice Gini es igual a 0, esto indica que el nodo es puro. Sin embargo, si el valor de este se aproxima más a 1, el nodo es impuro.

**Entropía:**

La idea básica de la entropía es medir el desorden de los datos de un nodo por medio de la variable clasificadora. Así como el índice de Gini, esta es utilizada para caracterizar la impureza de los datos y puede ser calculada por medio de la siguiente fórmula:
$$Entropia = {\sum_{i=1}^C -P_{i}*log_{2}(P_{i})}$$

Donde:
- Pi representa la proporción de datos en el conjunto de datos (S), pertenecientes a la clase específica i 
- C es el número de clases.


### Screenshot
![Image text](https://www.united-internet.de/fileadmin/user_upload/Brands/Downloads/Logo_IONOS_by.jpg)
## Technologies
***
A list of technologies used within the project:
* [Python](https://example.com): Version 3.9.x
* [Google Colab](https://example.com): Version x.x
* [Numpy](http://www.numpy.org/): Version 1.20.3
* [Pandas](https://pandas.pydata.org/pandas-docs/stable/visualization.html): Version 1.5.3
* [Seaborn](https://seaborn.pydata.org/): Version 0.12.2
* [Matplotlib](https://matplotlib.org/stable/): Version 3.8.1
* [Plotly](https://matplotlib.org/stable/): Version 5.15.0


## Installation
***
A little intro about the installation. 
```
$ git clone https://example.com
$ cd ../path/to/the/file
$ npm install
$ npm start
```
Side information: To use the application in a special environment use ```lorem ipsum``` to start
## Collaboration
***
Give instructions on how to collaborate with your project.
> Maybe you want to write a quote in this part. 
> It should go over several rows?
> This is how you do it.
## FAQs
***
A list of frequently asked questions
1. **This is a question in bold**
Answer of the first question with _italic words_. 
2. __Second question in bold__ 
To answer this question we use an unordered list:
* First point
* Second Point
* Third point
3. **Third question in bold**
Answer of the third question with *italic words*.
4. **Fourth question in bold**
| Headline 1 in the tablehead | Headline 2 in the tablehead | Headline 3 in the tablehead |
|:--------------|:-------------:|--------------:|
| text-align left | text-align center | text-align right |