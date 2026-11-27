# Three-Body Problem: ML Classifier vs. Numerical Solver

A neural network (PyTorch) that predicts whether a three-body system ends in **collision**, **escape** or **bounded motion** from its initial conditions. Training data is generated with a numerical solver (DOP853).

## Files

- `threebodysolver.py`: equations of motion and solver
- `datagenerator_3bp.py`: random initial conditions and simulation ($t = 30$)
- `classifier.py`: labels each trajectory (collision radius 0.1, escape radius 15)
- `model_3bp.py`: features, training and evaluation (10 000 systems, 2000 epochs, dropout 0.2, weight decay $10^{-4}$)

## Usage

```bash
pip install numpy scipy scikit-learn matplotlib torch
python model_3bp.py
```

Run once with `mode="new"`, then use `mode="saved"` to reuse the data.

## Results

Test accuracy: **64.8 %** (collision 60.0 %, escape 67.3 %, bounded 64.1 %). Always guessing the largest class gives 54 %.

## Limitations

The three-body problem is chaotic. Escapes from systems with positive energy are predictable, but for bound systems the outcome depends on chaotic close encounters, which limits how well it can be predicted from the initial state.
