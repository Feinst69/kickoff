# Réponses aux questions théoriques

## Phase 1 : Perceptron Multicouches

1. **Il existe différents types de couches dans un réseau de neurones artificiels. Quelles sont les types de couches pouvant composer un Perceptron multicouches ?**
   - Un perceptron multicouches (MLP) peut être constitué de couches d'entrée, de couches cachées (denses ou entièrement connectées) et d'une couche de sortie. Les couches cachées peuvent inclure des fonctions d'activation telles que ReLU, sigmoid ou tanh. Des couches de normalisation (BatchNorm) ou de régularisation (Dropout) peuvent également être intégrées.

2. **Définissez et différenciez les notions d'Epochs, d'Iterations et de Batch size.**
   - *Epoch* : un passage complet sur l'ensemble du jeu de données d'entraînement.
   - *Batch size* : le nombre d'échantillons traités avant la mise à jour des poids.
   - *Iteration* : une mise à jour des poids correspondant au traitement d'un batch. Le nombre d'itérations par epoch est donc égal au nombre total d'échantillons divisé par la taille du batch.

3. **Qu'est ce que l'hyper-paramètre learning rate ? Quelles sont les conséquences d'un learning rate trop bas ou trop élevé ?**
   - Le learning rate définit l'amplitude des mises à jour des poids lors de l'optimisation. Un taux trop bas ralentit l'entraînement et peut conduire à un piétinement. Un taux trop élevé risque de provoquer des oscillations voire une divergence de la fonction de coût.

4. **Définissez la Batch normalization et argumentez son utilisation.**
   - La Batch Normalization consiste à normaliser les activations d'une couche pour chaque mini-batch (soustraction de la moyenne et division par l'écart type), puis à appliquer des paramètres d'échelle et de décalage appris. Cela stabilise et accélère l'entraînement en réduisant la covariance interne.

5. **Qu'est-ce que l'algorithme d'optimisation d'Adam ?**
   - Adam (Adaptive Moment Estimation) est un algorithme d'optimisation qui combine la descente de gradient stochastique avec des estimations adaptatives du premier moment (moyenne) et du second moment (variance) des gradients. Il adapte le learning rate pour chaque paramètre et facilite une convergence rapide.

6. **Explorez et testez différentes combinaisons d'architectures et d'hyper-paramètres du PMC. Prenez soin de comparer vos modèles et notez les meilleurs résultats.**
   - Cette étape consiste à varier le nombre de couches, de neurones, les fonctions d'activation, le taux de dropout, etc. On peut par exemple comparer des configurations (784-128-64-10) ou (784-256-128-10) et mesurer la précision sur un jeu de validation afin de retenir la meilleure.

7. **Après l'utilisation d'une couche dense, les données sont transformées. Cela a souvent pour effet de produire des valeurs totalement dispersées. Remédiez à ce problème en ajoutant une couche de normalisation.**
   - On insère une couche de Batch Normalization après la couche dense pour recentrer et réduire la variance des activations, ce qui stabilise l'apprentissage.

8. **Surveillez le surapprentissage de vos modèles en visualisant la loss en fonction des epochs. Qu'est ce que le Early stopping ? S'il y a du surapprentissage, utilisez des couches de régularisation.**
   - On trace la courbe d'erreur sur l'entraînement et la validation pour détecter un écart croissant. Le *Early stopping* consiste à interrompre l'entraînement lorsqu'aucune amélioration sur le jeu de validation n'est observée pendant un certain nombre d'epochs, afin d'éviter le surapprentissage. Des techniques comme le Dropout ou la régularisation L2 permettent également de limiter ce phénomène.

9. **Évaluez vos modèles avec différentes métriques de classification (matrice de confusion et rapport de classification).**
   - On utilise la matrice de confusion pour visualiser les prédictions correctes et erronées par classe, et des indicateurs comme la précision, le rappel et le F1-score pour juger des performances globales et par classe.

10. **Concluez sur cette première tentative. Quel est le modèle construit générant le taux d'erreurs le plus bas ?**
   - En général, un MLP bien réglé peut atteindre une erreur autour de 1 à 2 % sur MNIST. Le modèle obtenant le meilleur compromis entre complexité et performance est conservé (par exemple un réseau avec deux couches cachées et BatchNorm).

## Phase 2 : Réseau neuronal convolutif

1. **Réalisez une veille sur les réseaux de neurones artificiels de type convolutif. Quel est l'architecture typique d'un CNN ?**
   - L'architecture classique d'un CNN alterne des blocs convolution + activation + pooling, suivis d'une ou plusieurs couches entièrement connectées en fin de réseau pour la classification.

2. **Donnez le principe de fonctionnement d'une couche convolutive. Qu'est ce qu'un filtre de convolution ?**
   - Une couche convolutive applique plusieurs filtres (ou noyaux) glissants sur l'image d'entrée. Chaque filtre est une matrice de poids apprise qui extrait une caractéristique locale (bord, motif, etc.). Le résultat est une carte de caractéristiques (feature map).

3. **Comment un filtre de convolution est-il appliqué à une image en entrée ? Qu'est ce qui en résulte ? En quoi est-il utile pour la détection d'objets ?**
   - Le filtre se déplace (convolution) sur l'image, effectuant à chaque position le produit élément par élément puis la somme, donnant ainsi la valeur d'une cellule dans la feature map. Ces cartes mettent en évidence des motifs locaux utiles pour détecter des contours ou des textures, étapes nécessaires pour reconnaître des objets.

4. **Quelle est la fonction d'activation utilisée par un CNN ? Pourquoi est-elle la plus adaptée pour ce type de réseaux de neurones ?**
   - La fonction ReLU (Rectified Linear Unit) est la plus couramment employée car elle évite le problème de gradient vanishing et accélère l'apprentissage tout en conservant la non-linéarité nécessaire.

5. **Qu'est ce qui arrive à la Feature Map lorsque celle-ci est donnée en paramètre à la fonction d'activation d'un CNN ?**
   - Chaque valeur de la feature map est transformée par la fonction d'activation (par exemple, avec ReLU, les valeurs négatives sont mises à zéro). Cela introduit de la non-linéarité et permet d'apprendre des représentations plus complexes.

6. **Donnez le principe de fonctionnement d'une couche de Pooling. Il existe différentes opérations de Pooling, citez-en au moins deux.**
   - Une couche de pooling réduit la dimension spatiale des feature maps en appliquant une opération de résumé sur des régions locales. Les opérations les plus courantes sont le *max pooling* et le *average pooling*.

7. **Quels sont les avantages de l'utilisation d'une couche de Pooling ?**
   - Le pooling diminue la taille des données, réduit le nombre de paramètres, limite le surapprentissage et introduit une invariance locale aux translations de l'image.

8. **La dernière couche d'un CNN est une couche entièrement connectée. Expliquez son fonctionnement. Qu'est ce que reçoit la couche entièrement connectée ?**
   - La couche entièrement connectée reçoit les feature maps aplaties en un vecteur. Chaque neurone de cette couche est connecté à toutes les valeurs de ce vecteur et calcule une combinaison linéaire suivie d'une activation (souvent Softmax pour la sortie) pour produire les scores de classe.

9. **Détaillez les raisons pour lesquelles un réseau de neurones convolutif est préféré à un réseau de neurones dense pour une tâche de classification d'images.**
   - Les CNN exploitent la structure spatiale des images grâce au partage de poids et à la localité des filtres, ce qui réduit drastiquement le nombre de paramètres par rapport à un réseau dense. Ils sont donc plus efficaces, généralisent mieux et sont particulièrement performants pour l'extraction de caractéristiques visuelles.

10. **Explorez et testez différentes architectures et hyper-paramètres d'un CNN. Prenez soin de comparer vos modèles et notez les meilleurs résultats.**
   - On peut varier le nombre de couches convolutionnelles, la taille des filtres, l'utilisation de BatchNorm ou Dropout, et ajuster le learning rate ou l'optimiseur. On compare ensuite la précision de validation pour retenir l'architecture la plus performante.

11. **Évitez une dispersion trop importante de vos données en utilisant une couche de normalisation.**
   - L'ajout de Batch Normalization après les couches convolutionnelles ou denses permet de normaliser les activations et favorise un apprentissage plus stable.

12. **Surveillez le surapprentissage de vos modèles en visualisant la loss en fonction des epochs. S'il y a du surapprentissage, utilisez des couches de régularisation.**
   - On suit les courbes d'apprentissage et on applique, si nécessaire, du Dropout, de la régularisation L2 ou un early stopping pour éviter que le modèle mémorise le jeu d'entraînement.

13. **Évaluez vos modèles avec les différentes métriques de classification (matrice de confusion et rapport de classification).**
   - De même que pour la phase précédente, on calcule la matrice de confusion et les scores (précision, rappel, F1) pour juger la qualité de la prédiction par classe.

14. **Concluez sur cette seconde phase. Quel est le taux d'erreurs le plus bas que vous pouvez obtenir ? Avez-vous obtenu de meilleures performances qu'avec un MLP ?**
   - Les CNN atteignent couramment des taux d'erreur inférieurs à 0,5 % sur MNIST. Avec un réseau correctement paramétré, on peut souvent dépasser la performance d'un MLP, qui reste généralement autour de 1 % d'erreur ou plus.

