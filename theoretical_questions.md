# Questions Théoriques

Ce document synthétise les questions présentes dans `data/questions.md` et fournit des réponses concises.

## Perceptron multicouches

1. **Types de couches** : un perceptron multicouches (MLP) peut comprendre des couches d'entrée, plusieurs couches cachées entièrement connectées et une couche de sortie. Dans un cadre plus général, on peut également y inclure des couches de normalisation ou de régularisation (Dropout).
2. **Epochs, itérations et batch size** : une *epoch* correspond à un passage complet sur l'ensemble des données d'entraînement. Une *itération* désigne une mise à jour des poids du réseau et se produit après l'exécution d'un seul *batch*. Le *batch size* est le nombre d'échantillons utilisés pour calculer une itération.
3. **Learning rate** : il s'agit du pas de mise à jour lors de l'optimisation. S'il est trop faible, l'apprentissage est très lent et peut rester bloqué dans un minimum local. S'il est trop élevé, l'algorithme peut diverger et ne jamais converger vers un optimum stable.
4. **Batch normalization** : cette technique consiste à normaliser l'activation des couches au sein du réseau pendant l'entraînement afin de stabiliser et d'accélérer l'apprentissage. Elle permet également une meilleure généralisation.
5. **Optimiseur Adam** : Adam est un algorithme d'optimisation stochastique qui combine les avantages d'AdaGrad et RMSProp. Il maintient un taux d'apprentissage adaptatif pour chaque paramètre en utilisant les moyennes mobiles du premier et second moment du gradient.

## Réseaux de neurones convolutifs

1. **Architecture typique** : un CNN est généralement composé d'un enchaînement de blocs `Convolution → Activation (ReLU) → Pooling`, éventuellement suivi de couches entièrement connectées pour la classification finale.
2. **Couche convolutive et filtre** : la couche applique plusieurs filtres de convolution (petites matrices de poids) sur l'image d'entrée afin d'extraire des motifs locaux. Chaque filtre génère une *feature map*.
3. **Application d'un filtre** : le filtre se déplace (convolution) sur l'image et calcule un produit matriciel local. Il en résulte une carte d'activation indiquant la présence du motif appris, utilisée pour détecter des objets ou des textures.
4. **Fonction d'activation** : ReLU est la plus courante car elle limite le problème du gradient qui disparaît et accélère l'apprentissage tout en introduisant la non-linéarité nécessaire.
5. **Effet sur la Feature Map** : l'application de l'activation ReLU met à zéro les valeurs négatives de la carte de caractéristiques et laisse inchangées les valeurs positives, rendant la représentation plus parcimonieuse.
6. **Pooling** : cette couche réduit la dimension spatiale des cartes de caractéristiques. Les opérations courantes sont le *max pooling* et l'*average pooling*.
7. **Avantages du pooling** : il diminue la taille des données tout en conservant les informations essentielles et confère une certaine invariance aux translations de l'entrée.
8. **Couche entièrement connectée finale** : elle prend en entrée les cartes de caractéristiques aplaties et calcule la distribution de probabilité sur les classes à l'aide d'une fonction d'activation (souvent softmax).
9. **Pourquoi préférer un CNN** : les CNN exploitent la structure spatiale des images grâce au partage de poids et aux opérations locales, ce qui réduit fortement le nombre de paramètres à apprendre par rapport à un réseau dense et améliore la capacité de généralisation.

## Expérimentations automatiques

Pour sélectionner le meilleur modèle, le script `train.py` réalise une
recherche par grille (*grid search*) sur plusieurs hyper‑paramètres du CNN
(nombre de filtres, taille de la couche dense et taux d'apprentissage). Les
modèles sont entraînés avec un *early stopping* pour éviter le
surapprentissage. À la fin de la recherche, le modèle affichant la plus haute
exactitude sur l'ensemble de test est sauvegardé dans `model.h5` et évalué à
l'aide d'une matrice de confusion et d'un rapport de classification.
