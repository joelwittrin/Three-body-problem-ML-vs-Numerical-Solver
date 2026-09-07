**THREE-BODY PROBLEM - ML CLASSIFIER vs NUMERICAL SOLVER**

The project uses a classical numerical solver for the three-dimensional three-body problem to generate trajectories, which are then classified into three outcomes: collision, escape, or bounded, based on the initial state of the system. A simple neural network model is trained to predict this outcome directly from the initial conditions, without solving the full trajectory. Note that this project is NOT finished at this time.

**RUN ORDER**
- threebodysolver.py (defines the ODE and the numerical solver, DOP853)
- datagenerator_3bp.py (generates training data by simulating random initial states, imports the solver)
- classifier.py (classifies each trajectory as collision, escape, or bounded, imports the data generator)
- model_3bp.py (builds labels, trains and evaluates the neural network model, imports the data generator and classifier)

Run 'python model_3bp.py' to kick off the full pipeline (data generation -> classification -> training -> evaluation).
