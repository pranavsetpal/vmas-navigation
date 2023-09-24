# Navigation with TorchRL and Vmas
This simply follows the [tutorial](https://pytorch.org/rl/tutorials/multiagent_ppo.html) project for MultiAgent TorchRL via PPO (MAPPO)

## Installation
Make sure to activate your venv and then run
```bash
git clone https://github.com/pranavsetpal/vmas-navigation/
cd vmas-navigation
pip install -r requirements.txt
```

## Usage
After cloning the directory, you can train the model via
```bash
python -m src.models.mappo
```

In order the render the trained model via
```bash
python -m src.models.visualize
```

You can change the hyperparameters by editing the `src/models/hyperparameters.py` file.
The trained model is saved in `models/policy-parameters.pt`
