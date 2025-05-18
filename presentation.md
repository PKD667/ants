# Simulation évolutive de fourmis

Étant particulièrement intéressé en intelligence artificielle, j'ai développé ce projet comme un exercice, qui me permet de faire émerger les premières dynamiques des algorithmes complexes utilisés aujourd'hui dans le machine learning, en partant d'un modèle simple. 

## Contexte

La simulation est plutôt simpliste, en ce qu'elle consitue un modèle de fourmis qui se déplacent sur une grille, en suivant des règles simples. 

Chaque fourmi dispose d'un *cerveau*, représenté par un ensemble de **poids** et de **biais** qui connecte ses *inputs* sensoriels (une grille 9x9 autour d'elle) à ses *outputs* moteurs (une direction de déplacement).

Ainsi, le fonctionement du cerveau est décrit par une multiplication matricielle entre les différentes couches de neurones, suivie d'une fonction d'activation (ReLU).

Le modèle fonctionne par cycles, d'une durée définie d'étapes, au cours desquels les fourmis se déplacent sur la grille, en suivant les règles de déplacement définies par le modèle, et mangent de la nourriture.

Au terme de chaque cycle, les fourmis sont évaluées en fonction de leur performance, mesurée par la quantité de nourriture qu'elles ont mangée. Les meilleures fourmis sont sélectionnées pour se reproduire et donner naissance à une nouvelle génération de fourmis.

Lors de chaque nouvelle génération, les poids et biais des parents sont mutés pour créer de la diversité génétique dans la population.

## Objectifs

En théorie, suivent un mécanisme de **sélection naturelle** et de **reproduction**, les fourmis devraient être capables d'apprendre à se déplacer sur la grille et à trouver de la nourriture de manière plus efficace au fil des générations.

Ainsi, elle pourraient commencer par se diriger vers la nourriture, puis développer des stratégies pour s'éloigner des murs et de leur congénères.

Finalement, on peu s'attendre a ce que notre systeme reproduisent les dynamiques d'optimisation des matrices vers une fonction optimale de choix des directions, similaire a celle qu'on aurait obtient en utilisant un algorithme de reinforcement learning.

## Résultats

Finalement, j'ai pou observer des résultats intéréssant, qui représentent aussi les enjeux des systems d'apprentissage autonomes. Plusieurs fois, le ceveau des fourmis s'est retrouvé coincé dans un minimul local, caractérisé par un comportement répétitif (tourner en rond, ou se déplacer en ligne droite), et n'a pas été capable de s'en échapper.

