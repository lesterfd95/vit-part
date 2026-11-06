Este repositorio contiene una parte de los recursos usados para preparar un experimento en el cual se realiza inferencia con una red neuronal compuesta solo por las capas que forman el primer modulo/bloque de atencion de un modelo ViT-base entrenado con CIFAR10.

A esta red neuronal se le puede hacer referencia en la documentacion de varias formas como: subred, submodelo, red parcial, modelo parcial. Además todos los terminos siguientes se refieren a la misma estructura: modulo de atencion, bloque de atencion, mecanismo de atencion.

Los resutados del experimento en el que se usaron estos recursos pueden ser consultados en: --insert link--


Clonar el repositorio del proyecto en tu area de trabajo

git clone 

En caso de que quieras usar docker:

1. instalar docker 
Consultar web y documentacion:
https://www.docker.com/
https://docs.docker.com/

2. crear la imagen de docker. 
Para ello ejecuta desde dentro del proyecto 
docker build -t vit_ti:1.0 .
La imagen docker creada con nombre vit_ti:1.0 tendrá todas las dependecias necesarias extraidas de requirements.txt
La imagen no contiene la carpeta scripts ni su contenido

3. instanciar un contenedor docker
Ejecutar el script create_container.sh 
Esto instanciará un contenedor a partir de la imagen creada y abrirá la terminal dentro en el directorio app al cual se vinculó
el contenido del directorio scripts. De esta forma se podrá usar y editar todo el contenido del directorio sin necesidad de aumentar el tamaño de 
la imagen con datasets y otros elementos pesados.

4. asegurate de poder usar los recursos de gpu desde dentro del contenedor. Para ello puede que necesites Nvidia Container Toolkit
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

En ocaciones puede haber conflicto entre la version de pytorch instalada en el contenedor y la version de CUDA soportada por los drivers de la GPU. Consulta la version de drivers y CUDA con la herramienta nvidia-smi y consulta la compatibilidad entre versiones CUDA y Pytorch en https://pytorch.org/



Para usar sin docker

Asegurate de instalar python y las dependencias listadas en requirements.txt y trabaja desde el directorio scripts


Descripcion de funcionalidades 

vit.py 
Contiene una clase ViT y clases componentes que permiten construir un modelo ViT. El archivo original se tomó de https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

utility.py 
Contiene funciones que se usan en varios otros scripts, como constuir un modelo ViT a partir de vit.py o costruir dataloaders para entrenamiento y evaluacion.
La funcion build_model podria extenderse para incluir más versiones de ViT. La funcion build_submodel podria extenderse para incluir modelos formados por otros módulos o conjuntos de módulos.

train.py 
Entrena un modelo vit con CIFAR10. Recibe nombre del modelo a entrenar, numero de batches y epochs. Guarda un .pth incluyendo el model_dict con pesos entrenados. 

test.py
Realiza inferencia con un modelo ViT entrenado sobre CIFAR10. Recibe el nombre del modelo y el tamaño de batch. Genera una archivo golden con las salidas (logits) obtenidas para todas las entradas. Genera un dataset para el primer modulo de atencion a partir de capturar con hooks las entradas y salidas del modulo. 
Este script podria extenderse para que genere datasets para otros modulos, no solo el primer módulo de atencion.

model_subset.py
A partir de un archivo .pth con pesos entrenados saca un subset de estos pesos correspondientes a las capas que componen el primer modulo de atencion. Recibe el nombre del modelo. Genera el archivo .pth para la red parcial costituida por el primer modulo de atenciion.
Este script podria extenderse para que genere un subset de pesos para otros modulos, no solo el primer módulo de atencion.

confidence.py
Usando dataset CIFAR10 y un golden de una inferencia sobre este, toma los indices de las imagenes clasificadas correctamente y los distribuye en grupos. Cada grupo corresponde a un rango de confianza de clasificacion. Recibe el nombre del modelo con el que se creo el golden. Genera una lista de listas, cada lista interna contiene los indices de imagenes que corresponden ese rango de confianza. 

data_subset.py
A partir del dataset generado para la subred, extrae un subset basado en un conjunto de indices. El subset a crear puede ser basado en un rango de indices de imagenes en cuyo caso recibe el rango a usar. Puede ser basado en confianza de clasificacion, para ello se usan listas pre-elaboradas de indices distribuidos en grupos de confianza, en ese caso recibe un arreglo de tamaños para saber cuantos grupos de confianza crear y cuantos elementos poner en cada uno. Genera el subset deseado.

attention.py
Permite realizar inferencia sobre la subred entrenada con el dataset preparado y comprobar que la salidas sean las esperadas. Contiene la funcionalidad de inyectar errores en imagenes consecutivas o batches consecutivos. 

Ejemplo de flujo de trabajo:

Vamos a preparar una red que consista en solamente el primer modulo de atencion de un modelo ViT de la variante ViT-base. Además para realizar inferencias con esta red, vamos a generar un dataset compuesto por los valores de entrada y salida del modulo de atención correspondientes a cuando se hace inferencia sobre CIFAR10 con el modelo ViT entero. Para ir un paso más allá, del dataset obtenido vamos a quedarnos con un subset compuesto por datos organizados de acuerdo al nivel de confianza de clasificacion que provocan en la salida.

Paso 1 

Usamos train.py para entrenar el modelo ViT-base 

python3 train.py --model base --batch_size 100 --epochs 100

La ejecucion descargará los subsets de entrenamiento y validación de CIFAR10 si no existen en el directorio de trabajo y los guardará en el directorio data. Realizará un entrenamiento de las epocas indicadas con el tamaño de batch indicado. Otros hiperpámetros usados son learning rate 1e-4, funcion de perdidas cross entropia y optimizador adam, estos u otros parámetros del entrenamiento se pueden cambiar editando el codigo de train.py. 
Por cada epoca de entrenamiento realizara una de validacion y se irá quedando con los pesos que mejor accuracy hayan conseguido. Al final del proceso guardará en el directorio trained_models un archivo vit_base_cifar10.pth. La posibilidad y duración del entrenamiento depende de los hiperparametros seleccionados y el hardware en que se ejecute.

En caso de estar usando docker en un servidor remoto, el script launch_detached.sh puede ser usado para lanzar automáticamente la construccion del contenedor docker y ejecutar el proceso de entrenamiento, además de permitir que continue luego de desconectarse de dicho servidor. En caso de uso, se genera un archivo full_train.log con los resultados de la ejecucion. Edite el script para más detalles y personalizacion.

Paso 2 

Usamos test.py para obtener un golden y un dataset para la subred que vamos a usar.

python3 test.py --model base --batch_size 100

La ejecucion usará el dataset de validacion de CIFAR10, costruirá el modelo ViT-base, cargará los pesos entrenados desde /trained_models/vit_base_cifar10.pth, pondrá hooks para capturar las entradas y salidas del primer módulo de atencion y realizará inferencia sobre los datos. Aquí podras comprobar si el accuracy de la red es el reportado por el entrenamiento. Al final habrá almacenado las salidas del modelo (logits) para todas las entradas y las guardará en el directorio goldens con el nombre golden_base.pt. Las entradas y salidas del primer módulo de atencion las guardara en el directorio module_datasets con el nombre vit_base_attn_dataset.pt.

Paso 3

Usamos model_subset.py para quedarnos solo con los pesos entrenados correspondientes a la subred a usar.

python3 model_subset.py --model base

La ejecucion carga el archivo de pesos entrenados en /trained_models/vit_base_cifar10.pth y extrae los datos correspondientes a las capas que componen el primer modulo de atencion. Guarda el subset de pesos en el directorio trained_models con nombre vit_base_part_att.pth.

Paso 4 

Usamos confidence.py para separar los datos según la confianza de clasificacion.

python3 confidence.py --model base

La ejecución carga de CIFAR10 la clasificacion correcta para cada imágen, luego carga de /goldens/golden_base.pt los logits de salida de una inferencia y de estos saca la prediccion de la red. Compara la prediccion de la red y con la clasificacion correcta y separa los indices de imágenes de predicciones correctas e incorrectas. Usando el golden tambien calcula la confianza de clasificacion. Esta se define como la diferencia en probabilidad entre las dos clases mas probables a la salida de la red. En este punto genera dos ficheros que contienen indices de imagenes y su confianza de clasificacion, un fichero para clasificacion correcta (base_cnf_corr_out.pt) y uno para clasificacion incorrecta (base_cnf_inc_out.pt) en el directorio confidence. El proceso continua con los datos de clasificaciones correctas. Los indices son separados en 4 grupos que cubren los distintos rangos de confianza de clasificacion (0%-25%, 26%-50%, 51%-75%, 76%-100%). Finalmente las listas que constituyen estos grupos son guardadas en base_cnf_corr_out_cat_4.pt en el directorio confidence.

Si no quisieramos que los datos a usar en la inferencia de nuestra subred tengan esta distribucion especial de acuerdo a a confianza de clasificacion, podriamos saltarnos este Paso 4.

Paso 5

Usamos data_subset.py para generar el subset de datos a usar en la inferencia de nuestra subred.

python3 data_subset.py --model base --mode conf --group_sizes 64 64 128 256

La ejecucion carga los valores del dataset creado para nuestra subred desde /module_datasets/vit_base_attn_dataset.pt. Carga las listas de indices de confianza desde /confidence/base_cnf_corr_out_cat_4.pt. Luego reorganiza creado un dataset de 512 valores, primero 64 valores cuyos indices estan en el grupo de 0%-25% de confianza, seguido de 64 valores cuyos indices estan en el grupo de 26%-50%, etc. Los valores 64 64 128 256, se escogieron para sumar 512 elementos en el dataset final. La diferencia de tamaños de los grupos se debe a que no hay suficientes elementos en todos los grupos de confianza para hacerlos iguales, esto es porque hay una tendencia en las predicciones correctas a que haya mas predicciones de alta confianza que de baja confianza. El nuevo dataset reducido y organizado se guarda en /module_datasets/vit_base_attn_dataset_cnf_4.pt

En caso de que no se desee organizar por confianza y solo se quierea obtener un subset de tamaño determinado, se puede usar: 

python3 data_subset.py --model base --mode range --range 0 512

Esto resultaria en un nuevo dataset formado con los primeros 512 elementos (0 a 511) del dataset original. Se guardaria en /module_datasets/vit_base_attn_dataset_0_512.pt

Incluso podria usarse el dataset entero sin recoretar si se desea, eliminando entonces la necesidad del Paso 5.

Paso 6

Ahora ya tenemos los elementos necesarios para realizar la inferencia con nuestra subred:

1. El modelo de subred: se crea con build_submodel() de utility.py usando vit.py
2. Los pesos entrenados para la subred: /trained_models/vit_base_part_att.pth
3. El dataset de entradas y salidas: /module_datasets/vit_base_attn_dataset_cnf_4.pt 

Usamos attention.py para realizar la inferencia y comprobar que la subred se comporta debidamente, esto es que las salidas sean similares a lo esperado.

python3 attention.py --model base --dataset vit_base_attn_dataset_cnf_4.pt --atol 5.0e-6 --batch_size 64 --no-inject

La ejecucion creará la subred, cargará los pesos de /trained_models/vit_base_part_att.pth, cargará el dataset de /module_datasets/vit_base_attn_dataset_cnf_4.pt y realizará la inferencia. En cada batch comprobará que no existan errores. Usar --inject como argumento de la llamada hará que se inyecten algunos errores fijos para probar el mecanismo de comparación. Dependiendo del valor de tolerancia escogido (atol y rtol) podrian encontrarse errores incluso sin inyectarlos, debido a errores de redondeo acumulados y otras incertidumbres numericas en la ejecucion del modelo.

A pesar de que este script es menos complejo que el ambiente de software en que se despliega la inferencia en el experimento real, es una herramienta para jugar con los recursos creados, comprobar bugs, afinar valores de tolerancia, etc.
  












