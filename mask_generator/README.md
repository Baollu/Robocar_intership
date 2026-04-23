# Mask Generator — Segmentation SegNet pour Robocar

Ce module entraîne et utilise un réseau de neurones **SegNet** pour générer des masques de segmentation sémantique à partir d'images de caméra. L'objectif est de détecter la chaussée (route/trottoir) pixel par pixel, utile pour la navigation autonome du Robocar.

## Architecture

Le modèle est un **SegNet encodeur-décodeur** :
- **5 stages d'encodage** : convolutions + batch norm + ReLU + max pooling (avec sauvegarde des indices)
- **5 stages de décodage** : max unpooling (avec les indices) + convolutions + batch norm + ReLU
- **Sortie** : softmax sur 2 classes — `0` = non-chaussée, `1` = chaussée

```
model.py         → Architecture SegNet
train.py         → Entraînement du modèle
test.py          → Inférence sur une image
parameters.py    → Paramètres image (taille, facteur de réduction)
model.json       → Hyperparamètres (epochs, lr, momentum...)
datasets/        → Répertoire pour les données d'entraînement
weights/         → Répertoire pour les checkpoints
```

## Prérequis

- Python 3.8+
- CUDA recommandé (mais CPU fonctionne)

## Installation

```bash
cd mask_generator

# Créer et activer le virtualenv
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# ou .venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## Configuration

Les hyperparamètres sont dans [model.json](model.json) :

| Paramètre                    | Valeur par défaut | Description                          |
|------------------------------|-------------------|--------------------------------------|
| `epochs`                     | 1000              | Nombre d'époques d'entraînement      |
| `image_to_load`              | 30                | Images chargées par époque           |
| `learning_rate`              | 0.001             | Taux d'apprentissage SGD             |
| `sgd_momentum`               | 0.9               | Momentum SGD                         |
| `bn_momentum`                | 0.5               | Momentum batch normalization         |
| `cross_entropy_loss_weights` | [1.0, 5.0]        | Poids des classes (5x pour chaussée) |
| `in_chn`                     | 3                 | Canaux d'entrée (RGB)                |
| `out_chn`                    | 2                 | Canaux de sortie (2 classes)         |

Les paramètres d'image sont dans [parameters.py](parameters.py) :
- `image_width = 256`, `image_height = 128`

## Dataset

Le script d'entraînement attend les données dans :
```
../DatasetSimuator/ColoredCamera/   ← Images RGB de la caméra
../DatasetSimuator/MaskCamera/      ← Masques de vérité terrain (niveaux de gris)
```

Les masques sont binarisés avec un seuil de 120 : pixels < 120 → classe 0, pixels ≥ 120 → classe 1.

## Entraînement

```bash
source .venv/bin/activate
python train.py
```

Le script :
1. Charge un checkpoint existant (configurer le chemin dans `train.py`)
2. Charge `image_to_load` images par époque depuis le dataset
3. Optimise avec SGD + CrossEntropyLoss pondérée
4. Sauvegarde un checkpoint horodaté dans `weights/` après chaque run

## Inférence / Test

```bash
source .venv/bin/activate
python test.py
```

Par défaut, teste sur `car_pictures/320_180/frame_07775.png` et affiche le masque prédit.

**Usage programmatique :**
```python
from test import transform_image

mask = transform_image("path/to/image.png", debug=False)
# mask : tenseur numpy (height x width) avec classes 0 ou 1
```

## Notes importantes

- Les chemins vers les checkpoints sont **hardcodés** dans `train.py` et `test.py` — à adapter à votre machine.
- Le support GPU (CUDA) est disponible mais nécessite une installation PyTorch avec CUDA.
- Pour changer la résolution de traitement, modifier `parameters.py`.
