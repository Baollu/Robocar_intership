# Mask Generator — Guide pour débutants en IA

Ce guide explique le projet `mask_generator` en supposant que tu sais coder mais que tu n'as jamais touché à l'intelligence artificielle ni au hardware embarqué.

---

## C'est quoi le problème qu'on résout ?

La voiture RC doit rester sur la route. Pour ça, elle a une caméra qui filme en permanence. Mais une image brute c'est juste des pixels — la voiture ne "sait" pas où est la route.

L'objectif de ce projet est d'entraîner un réseau de neurones capable de regarder une image et de répondre **pixel par pixel** : *"est-ce que ce pixel fait partie de la route ou pas ?"*

Le résultat s'appelle un **masque de segmentation** : une image en noir et blanc de la même taille que l'originale, où blanc = route et noir = le reste.

```
Image caméra (couleur)        Masque produit (noir/blanc)
┌─────────────────────┐       ┌─────────────────────┐
│  🌳  route  🌳      │  -->  │  ░░░  ██████  ░░░   │
│  🌳  route  🌳      │       │  ░░░  ██████  ░░░   │
└─────────────────────┘       └─────────────────────┘
```

---

## Notions d'IA à connaître

### Réseau de neurones
Un réseau de neurones est un programme qui **apprend à partir d'exemples**. Au lieu d'écrire des règles à la main ("si le pixel est gris alors c'est la route"), on lui montre des milliers d'exemples (image + bonne réponse) et il trouve lui-même les règles.

Concrètement en code c'est une suite de fonctions mathématiques (des matrices, des multiplications) organisées en couches. PyTorch est la bibliothèque qui gère tout ça.

### Entraînement
L'entraînement c'est la phase d'apprentissage. Le modèle regarde une image, fait une prédiction, on compare avec la bonne réponse (qu'on appelle **ground truth** ou **vérité terrain**), et on corrige légèrement les paramètres du modèle pour qu'il se trompe moins la prochaine fois.

Cette correction s'appelle la **rétropropagation** (backpropagation). En code : `loss.backward()` + `optimizer.step()`.

### Epoch
Une epoch = le modèle a vu une fois l'ensemble du dataset. On répète plusieurs centaines voire milliers d'epochs pour que le modèle s'améliore progressivement. Dans ce projet : 1000 epochs par défaut.

### Loss (perte)
La loss est un nombre qui mesure à quel point le modèle se trompe. Plus la loss est basse, meilleur est le modèle. L'objectif de l'entraînement est de minimiser ce nombre. Tu peux voir la loss descendre dans les logs pendant l'entraînement.

### Checkpoint
Un checkpoint est une sauvegarde des paramètres du modèle à un instant donné. Comme un `git commit` pour les poids du réseau. Ça permet de reprendre l'entraînement plus tard ou de réutiliser le modèle sans ré-entraîner.

---

## Comment l'IA apprend vraiment — la mécanique interne

Oui, c'est bien une boucle. Mais ce qui rend l'IA "intelligente" c'est ce qui se passe **à l'intérieur** de chaque itération. Voici la mécanique réelle.

### Le modèle c'est juste des nombres

Un réseau de neurones n'est rien d'autre qu'une très grande fonction mathématique avec des **millions de paramètres** (appelés **poids** ou **weights**). Au départ, ces poids sont initialisés aléatoirement — le modèle ne sait rien faire.

```python
# Conceptuellement, le modèle c'est ça :
def model(image, poids):
    return calcul_compliqué(image, poids)  # prédit un masque

# Au début :
poids = valeurs_aléatoires()  # résultat = n'importe quoi
```

### La boucle d'apprentissage

```python
for epoch in range(1000):
    # 1. Le modèle fait une prédiction
    prediction = model(image)

    # 2. On mesure l'erreur (la loss)
    erreur = comparer(prediction, masque_correct)

    # 3. On calcule comment modifier les poids pour réduire l'erreur
    erreur.backward()       # ← la magie se passe ici

    # 4. On modifie légèrement les poids dans la bonne direction
    optimizer.step()
```

### Ce que fait `loss.backward()` — la vraie explication

C'est l'étape clé. `backward()` calcule pour **chaque poids** du réseau : *"si j'augmente légèrement ce poids, est-ce que l'erreur monte ou descend ?"*

Ce calcul s'appelle le **gradient**. C'est une dérivée — la même notion qu'en maths : la pente d'une courbe à un point donné.

Imagine que l'erreur est une montagne et que tu veux descendre dans la vallée (erreur minimale). Le gradient te dit dans quelle direction est la pente descendante. `optimizer.step()` fait un petit pas dans cette direction.

```
Erreur
  │
  │   ╲
  │    ╲         ← gradient = pente à ce point
  │     ╲___/‾‾
  │          ↑ minimum (ce qu'on cherche)
  └──────────────── valeur d'un poids
```

### Pourquoi ça marche sur des images ?

PyTorch enregistre **toutes les opérations mathématiques** faites sur les données (c'est ce qu'on appelle le graphe de calcul). Quand tu appelles `backward()`, il remonte ce graphe en sens inverse et calcule le gradient de chaque poids par rapport à l'erreur finale — automatiquement.

C'est pour ça que tu n'as pas à écrire les dérivées à la main : PyTorch le fait tout seul. Ce mécanisme s'appelle la **différentiation automatique**.

### En résumé : ce qui se passe réellement à chaque epoch

```
1. Image entre dans le réseau
2. Chaque couche fait des multiplications avec ses poids → produit un résultat
3. On compare ce résultat au masque correct → on obtient un score d'erreur (loss)
4. backward() : on calcule pour chaque poids "de combien contribue-t-il à l'erreur ?"
5. optimizer.step() : on décale chaque poids d'un tout petit peu dans la direction qui réduit l'erreur
6. Répéter 1000 fois → les poids convergent vers des valeurs qui donnent de bonnes prédictions
```

Après 1000 epochs, les poids ne sont plus aléatoires : ils encodent la connaissance "à quoi ressemble une route".

---

## Architecture SegNet — comment ça marche

Le modèle utilisé ici s'appelle **SegNet**. C'est une architecture en deux parties symétriques :

### L'encodeur — comprendre l'image

L'encodeur "compresse" l'image progressivement pour en extraire le sens. C'est comme zoomer de plus en plus loin pour voir les formes générales plutôt que les détails.

```
Image 256x128 (RGB)
      ↓  Stage 1 : Conv + BN + ReLU  (détecte les bords)
      ↓  MaxPool → 128x64            (réduit la taille de moitié)
      ↓  Stage 2 : Conv + BN + ReLU  (détecte les textures)
      ↓  MaxPool → 64x32
      ↓  Stage 3 : Conv + BN + ReLU  (détecte les formes)
      ↓  MaxPool → 32x16
      ↓  Stage 4 : Conv + BN + ReLU  (détecte les objets)
      ↓  MaxPool → 16x8
      ↓  Stage 5 : Conv + BN + ReLU  (compréhension globale)
      ↓  MaxPool → 8x4
```

**Conv (Convolution)** : filtre qui glisse sur l'image pour détecter des patterns locaux (bords, couleurs, textures). Paramètre `kernel_size=3` = le filtre regarde des zones 3x3 pixels à la fois.

**BN (Batch Normalization)** : normalise les valeurs pour stabiliser l'entraînement. Sans ça, les valeurs peuvent exploser ou disparaître au fil des couches.

**ReLU** : fonction d'activation très simple — met à zéro les valeurs négatives. Elle introduit de la non-linéarité (sans elle le réseau ne serait qu'une multiplication de matrices).

**MaxPool** : réduit la résolution de moitié en gardant uniquement la valeur maximale dans chaque zone 2x2. **Crucial** : les *indices* (la position des maximums) sont sauvegardés pour l'étape suivante.

### Le décodeur — reconstruire le masque

Le décodeur fait le chemin inverse : il reconstruit une image pleine résolution à partir de la représentation compressée.

```
Représentation 8x4
      ↑  MaxUnpool (avec les indices sauvegardés) → 16x8
      ↑  Conv + BN + ReLU
      ↑  MaxUnpool → 32x16
      ↑  ...
      ↑  MaxUnpool → 256x128
Masque 256x128 (2 canaux : non-route / route)
```

**MaxUnpool** : l'inverse du MaxPool. Utilise les indices sauvegardés pendant l'encodage pour replacer les valeurs exactement où elles étaient. C'est pour ça que SegNet est précis au pixel près.

### La sortie

Le réseau produit 2 "canaux" pour chaque pixel :
- Canal 0 : score "ce pixel est NON-route"
- Canal 1 : score "ce pixel est route"

`torch.argmax(output, dim=1)` choisit le canal avec le score le plus élevé → 0 ou 1 par pixel.

---

## Le dataset — comment sont faites les données

### Les images d'entrée
Des photos couleur (RGB) prises depuis la caméra de la voiture dans le simulateur. Rangées dans `DatasetSimuator/ColoredCamera/`.

### Les masques de vérité terrain
Des images en niveaux de gris générées par le simulateur, où chaque pixel indique si c'est la route ou non. Rangées dans `DatasetSimuator/MaskCamera/`.

La conversion en binaire se fait avec un seuil de 120 :
```python
# pixel < 120  → classe 0 (non-route)
# pixel >= 120 → classe 1 (route)
```

### Pourquoi des poids de classe `[1.0, 5.0]` ?
Dans une image, il y a souvent beaucoup plus de pixels "non-route" que "route". Si on traite les deux classes également, le modèle apprend à tout dire "non-route" et obtient quand même un bon score. Les poids forcent le modèle à pénaliser 5x plus fort les erreurs sur la route. C'est le paramètre `cross_entropy_loss_weights` dans `model.json`.

---

## Les hyperparamètres — quoi régler et pourquoi

Les hyperparamètres sont les réglages du modèle, à modifier dans [model.json](model.json).

| Paramètre | Valeur | Effet si on augmente |
|---|---|---|
| `epochs` | 1000 | Plus d'entraînement → meilleur modèle (mais plus long) |
| `learning_rate` | 0.001 | Apprentissage plus rapide mais risque d'être instable |
| `sgd_momentum` | 0.9 | Le modèle "prend de l'élan" → converge plus vite |
| `image_to_load` | 30 | Plus d'images par epoch → plus représentatif mais plus lent |
| `bn_momentum` | 0.5 | Vitesse d'adaptation de la normalisation |

**Learning rate** : c'est le réglage le plus important. Trop grand = le modèle oscille et n'apprend pas. Trop petit = l'entraînement prend une éternité. `0.001` est une valeur classique pour commencer.

**SGD** (Stochastic Gradient Descent) : l'algorithme d'optimisation. C'est lui qui calcule dans quelle direction modifier les paramètres du modèle à chaque étape.

---

## Comment savoir si les paramètres sont bons

La seule façon de le savoir c'est de **regarder la loss pendant l'entraînement**. Voici les 3 cas possibles :

### Cas 1 — Ça marche bien
```
Loss
1.0 │╲
0.7 │  ╲
0.4 │    ╲___
0.2 │        ‾‾──────  ← se stabilise à une valeur basse
    └──────────────── epochs
```
La loss descend régulièrement puis se stabilise. C'est le comportement attendu.

### Cas 2 — Learning rate trop grand
```
Loss
1.0 │╲  ╱╲  ╱╲
0.7 │  ╲╱  ╲╱  ╲╱  ← oscille, n'apprend pas vraiment
0.5 │
    └──────────────── epochs
```
Le modèle fait des pas trop grands et dépasse à chaque fois le minimum. **Solution** : divise le `learning_rate` par 10.

### Cas 3 — Le modèle ne converge pas (loss ne bouge pas)
```
Loss
0.9 │────────────────  ← reste bloquée
    └──────────────── epochs
```
Le modèle n'apprend rien. Causes possibles : learning rate trop petit, dataset trop petit, bug dans les données. **Solution** : multiplie le `learning_rate` par 10, ou vérifie que les chemins vers le dataset sont corrects.

### Règle pratique pour régler les hyperparamètres

Ne change **qu'un seul paramètre à la fois** et observe l'effet sur la loss. Si tu changes tout en même temps tu ne sauras pas ce qui a aidé ou nui.

Ordre recommandé :
1. D'abord vérifie que la loss **descend** (sinon le learning_rate est mauvais)
2. Ensuite vérifie qu'elle descend **assez vite** (augmente `image_to_load` ou `epochs`)
3. En dernier, ajuste `sgd_momentum` et `bn_momentum` si la courbe est instable

---

## Jetson Nano — spécificités hardware

Le Jetson Nano est un mini-ordinateur avec un GPU NVIDIA intégré. Contrairement à un PC classique, il est conçu pour faire tourner des réseaux de neurones avec peu de consommation électrique.

**CUDA** : technologie NVIDIA qui permet d'utiliser le GPU pour les calculs du réseau de neurones. Le GPU peut faire des milliers de multiplications en parallèle — l'entraînement et l'inférence sont beaucoup plus rapides qu'en CPU.

Dans le code, la détection est automatique :
```python
cuda_available = torch.cuda.is_available()
if cuda_available:
    model.cuda()   # déplace le modèle sur le GPU
    images.cuda()  # déplace les données sur le GPU
```

Sur Jetson Nano, `cuda_available` sera `True` → le GPU sera utilisé automatiquement.

**Mode haute performance** (à faire sur le Jetson avant d'entraîner) :
```bash
sudo nvpmodel -m 0    # passe en mode 10W (max performance)
sudo jetson_clocks    # pousse les fréquences CPU/GPU au maximum
```

---

## Structure des fichiers

```
mask_generator/
├── model.py          → Définition du réseau SegNet (l'architecture)
├── train.py          → Boucle d'entraînement
├── test.py           → Inférence sur une image (utiliser le modèle entraîné)
├── parameters.py     → Taille des images (256x128)
├── model.json        → Hyperparamètres réglables sans toucher au code
├── weights/          → Checkpoints sauvegardés après chaque entraînement
└── DatasetSimuator/  → Dataset (à générer avec racing_simulator)
    ├── ColoredCamera/    → Images RGB
    └── MaskCamera/       → Masques de vérité terrain
```

---

## Lancer l'entraînement

```bash
# Depuis la racine du projet
source venv/bin/activate
cd mask_generator
python train.py
```

Tu verras défiler des lignes comme :
```
Epoch 0:
Loss at 0 mini-batch: 0.342 epoch 0
Average loss @ epoch: 0.342
Epoch 1:
Loss at 0 mini-batch: 0.289 epoch 1
...
```

La loss doit descendre au fil des epochs. Si elle stagne ou monte, essaie de baisser le `learning_rate`.

## Tester le modèle entraîné

```bash
python test.py
```
python follow_the_line/camera/camera_output.py
Modifie le chemin de l'image et du checkpoint dans `test.py` avant de lancer. Le résultat s'affiche visuellement : blanc = route détectée, noir = reste.

---

## Lien avec les autres projets

```
racing_simulator  →  génère le dataset (images + masques)
       ↓
mask_generator    →  entraîne le modèle sur ce dataset
       ↓
follow_the_line   →  utilise le modèle entraîné en temps réel sur la voiture
```