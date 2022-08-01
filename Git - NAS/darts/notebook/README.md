<h1>Notebook à executer sur Colab</h1>

- Le notebook <b>DARTS-PyTorch.ipynb</b> est responsable de lancement du projet DARTS en PyTorch à l'aide de la library nni à executer sur Colab.

- Le notebook <b>DARTS_v1-TensorFlow.ipynb</b> est responsable de lancement du projet DARTS_v1 en TensorFlow à executer sur Colab.

- Le notebook <b>DARTS_v2-TensorFlow.ipynb</b> est responsable de lancement du projet DARTS_v2 en TensorFlow avec les dernières modifications visant à améliorer les performances à executer sur Colab.

- Le notebook <b>Conversion_of_DARTS_to_tf.ipynb</b> est un notebook commenté qui permet de réaliser la création du modèle tensorflow issue du DARTS à l'aide du plan du modèle généré suite au DARTS. Ce notebook comprend également le code permettant d'élaguer les modèles ("méthode du pruning") et également nos recherches et tentatives d'application de la quantisation sur nos modèles dans le cadre du projet.

- Le notebook <b>Génération_KDARTS.ipynb</b> est un notebook commenté qui permet de générer un modèle tensorflow qui correspond au meilleur modèle possible en utilisant la méthode innovante que nous avons développé en fusionant la CKD et le DARTS : le KDARTS.

- Le notebook <b>generation_of_darts.ipynb</b> est un notebook commenté qui permet de générer le plan de l'architecture du modèle généré grâce à la méthode de DARTS.

- Le dossier <b>project_notebook</b> contient plusieurs notebooks de DARTS, et donc l'ensemble du code source à executer sur Colab.

<h1>Remarques</h1>
Dans les notebooks <b>Conversion_of_DARTS_to_tf.ipynb</b> et <b>generation_of_darts.ipynb</b>, on a des chemins de fichiers qui font références à un environnement de type Kaggle. En effet, nous avons utilisé l'iDE propre à kaggle car cela nous permettait d'avoir entre 30 heures et 40 heures de calcul pour pouvoir entraîner nos modèles. Nous recommendons donc d'utiliser kaggle pour effectuer les gros entraînement (en mettant les fichiers contenant les modèles comme étant des datasets personnels). On peut également utiliser google colab afin de pouvoir accéder à du code sur l'ordinateur de travail mais google colab ne permet dans sa version gratuite que d'avoir accès à 4h d'entraînement par jour.

<br/>
<h1>Création d'un modèle à l'aide d'un plan d'architecture</h1>

Pour générer une architecture, il faut utiliser les notebooks créés à ce sujet (DARTS-PyTorch.ipynb, DARTS_v1-TensorFlowipynb et DARTS_v2-TensorFlow.ipynb) adapté à une utilisation sur Colab. Les versions de DARTS ne sont pas celles compatibles à une utilisation de la distillation de connaissance (POC Data Inference).

Une fois que le plan d'architecture est généré, la version du DARTS n'ayant pas d'importance, la première étape est d'utiliser le notebook Conversion_export_en_normal_reduce.ipynb, toujours sur Colab.

A partir de ce résultat, il est possible de créer un model à l'aide du notebook Création_modèle_via_plan_du_DARTS_v3.ipynb à executer sur Colab.

