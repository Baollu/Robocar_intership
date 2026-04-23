# Racing Simulator - Apprentissage par Imitation

Un simulateur de course utilisant Unity ML-Agents où une IA apprend à conduire en imitant un conducteur humain (behavioral cloning).

## Architecture du Projet

```
racing_simulator/
├── config.json                 # Configuration des agents (FOV, rayons)
├── simulation_parameters.py    # Paramètres globaux (RAY_COUNT)
├── lib/
│   ├── RobocarModel.py        # Réseau de neurones PyTorch
│   ├── robocar_env.py         # Connexion à l'environnement Unity
│   ├── robocar_simulation.py  # Boucle de simulation
│   ├── data_recorder.py       # Enregistrement CSV des données
│   ├── my_keyboard.py         # Gestion des entrées clavier
│   └── Gamepad/               # Support manette
├── main_record_data.py        # Enregistrer des données (clavier)
├── main_record_data_gamepad.py # Enregistrer des données (manette)
├── train_ai.py                # Entraîner le modèle
├── main_ai_drive.py           # Conduite autonome par l'IA
├── main_docker_test.py        # Test de l'environnement
├── models/                    # Modèles sauvegardés
└── datasets/                  # Données d'entraînement
```

## Comment ça Fonctionne

### 1. Perception de l'Environnement

La voiture perçoit son environnement via **50 rayons** (configurable) qui mesurent la distance aux obstacles dans un champ de vision de 180°. Ces rayons, combinés à la vitesse et l'angle de direction actuels, forment l'entrée du modèle.

### 2. Le Modèle de Réseau de Neurones

```
Entrées (52)                    Sorties (2)
─────────────                   ───────────
• 50 rayons      ┌─────────┐    • Vitesse
• Vitesse    ───►│ 200 (Tanh) │    • Direction
• Direction      │ 200 (Tanh) │◄───
                 └─────────┘
```

Le modèle est un perceptron multicouche avec :
- **Entrée** : 52 valeurs (50 rayons + vitesse + direction)
- **Couches cachées** : 2 × 200 neurones avec activation Tanh
- **Sortie** : 2 valeurs (vitesse et direction, entre -1 et 1)

### 3. Pipeline d'Apprentissage

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Conduite    │───►│ Enregistrer  │───►│  Entraîner   │───►│  Conduite    │
│  Manuelle    │    │  Données     │    │  le Modèle   │    │  Autonome    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

## Installation

### Prérequis

```bash
pip install torch pandas mlagents-envs pynput
```

### Simulateur Unity

Placez le build du simulateur Unity dans le dossier du projet :
```
racing_simulator/RacingSimulatorLinux/BuildLinux/RacingSimulator.x86_64
```

## Utilisation

### Étape 1 : Enregistrer des Données de Conduite

Conduisez manuellement pour créer un dataset d'entraînement :

```bash
python main_record_data.py
```

**Contrôles clavier :**
| Touche | Action |
|--------|--------|
| Z | Accélérer |
| S | Freiner / Reculer |
| Q | Tourner à gauche |
| D | Tourner à droite |
| ESC | Sauvegarder et quitter |

Les données sont sauvegardées dans `data.csv`.

**Alternative avec manette :**
```bash
python main_record_data_gamepad.py
```

### Étape 2 : Entraîner le Modèle

Modifiez le nom du fichier CSV dans `train_ai.py` si nécessaire, puis :

```bash
python train_ai.py
```

Paramètres d'entraînement par défaut :
- **Époques** : 2000
- **Batch size** : 10
- **Learning rate** : 1e-5
- **Loss** : L1Loss (MAE)
- **Optimiseur** : Adam

Le modèle est sauvegardé dans `models/model_trained`.

### Étape 3 : Tester l'IA

Copiez/renommez votre modèle entraîné :
```bash
cp models/model_trained models/current_model
```

Lancez la conduite autonome :
```bash
python main_ai_drive.py
```

Vous pouvez reprendre le contrôle à tout moment avec les touches Z/Q/S/D.

## Configuration

### config.json

```json
{
    "agents": [{
        "fov": 180,      // Champ de vision en degrés
        "nbRay": 50      // Nombre de rayons de détection
    }]
}
```

### simulation_parameters.py

```python
RAY_COUNT = 50  # Doit correspondre à config.json
```

### robocar_env.py

```python
SCREEN_WIDTH = 700    # Largeur fenêtre
SCREEN_HEIGHT = 500   # Hauteur fenêtre
TIME_SCALE = 1        # Vitesse de simulation (1 = temps réel)
GRAPHIC_MODE = True   # Affichage graphique
```

## Format des Données

Le CSV d'entraînement contient :

| Colonne | Description |
|---------|-------------|
| user speed | Commande vitesse de l'utilisateur [-1, 1] |
| user steering | Commande direction de l'utilisateur [-1, 1] |
| r0 - r49 | Distances des 50 rayons |
| speed | Vitesse actuelle du véhicule |
| steering | Direction actuelle du véhicule |

## Conseils pour un Bon Entraînement

1. **Conduisez de manière fluide** : évitez les mouvements brusques
2. **Variez les situations** : virage serrés, lignes droites, corrections
3. **Plusieurs tours** : plus de données = meilleur apprentissage
4. **Nettoyez les données** : supprimez les moments d'erreur/crash
5. **Augmentez les époques** si le modèle sous-apprend

## Dépannage

| Problème | Solution |
|----------|----------|
| Simulateur introuvable | Vérifiez le chemin dans `robocar_env.py` |
| Timeout connexion | Augmentez `timeout_wait` dans `robocar_env.py` |
| Modèle diverge | Réduisez le learning rate |
| IA instable | Enregistrez plus de données variées |

## Structure des Classes

### Observation
```python
@dataclass
class Observation:
    rays: list       # 50 distances de rayons
    reward: float    # Récompense (non utilisée ici)
    speed: float     # Vitesse actuelle
    steering: float  # Direction actuelle
    x, y, z: float   # Position 3D
```

### Action
```python
@dataclass
class Action:
    speed: float = 0     # [-1, 1] : recul ← → avance
    steering: float = 0  # [-1, 1] : gauche ← → droite
```
