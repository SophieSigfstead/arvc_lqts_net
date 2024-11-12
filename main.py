from comet_ml import Optimizer
import sys
import random
import os
from ecg_classifier import train_and_evaluate, load_model

def get_experiment(experiment, model_name):

    experiment.log_parameter("model_type", f"{model_name}")

    experiment.log_parameter("loss", "sparse_categorical_crossentropy")
    experiment.log_parameter("optimizer", "Adam")

    experiment.log_parameter("validation_frac", 0.15)
    #experiment.log_parameter("max_ECGs_per_patient", 3)
    experiment.log_parameter("deterministic", True)
    experiment.log_parameter("activation_fn", "ReLU")
    return experiment

if __name__ == "__main__":
    comet_ml_key = sys.argv[1]
    opt_metric = "validation_loss"
    config = {"algorithm": "random",

    # Declare  hyperparameters:
    "parameters": {
        #"epochs": {"type": "integer", "scaling": "uniform", "min": 100, "max": 800},
        "batch_size": {"type": "discrete", "values": [ 4,8,16, 32, 64, 128]},
        "epochs":{"type": "discrete", "values": [5,10,20,30,50,100,200]},
        # "epochs":{"type": "discrete", "values": [1]},
        #"dropout": {"type": "discrete", "values": [0.1,0.2,0.3]},
        #"learning_rate": {"type": "discrete", "values": [0.001, 0.01, 0.1]},
         #"epsilon": {"type": "discrete", "values": [1e-8]},
        #"epsilon": {"type": "discrete", "values": [1e-7, 1e-4, 1e-2, 1]}
    },

    # Declare what to optimize, and how:
    "spec": {
        "metric": opt_metric,
        "objective": "minimize",
        "maxCombo": 100
    }}

    opt = Optimizer(config)
    root_dir = "./csv_ecgs"

    for experiment in opt.get_experiments(workspace = "sophiesigfstead", project_name = "lqts-arvc-project-initial-runs", api_key = comet_ml_key):
            # Randomly choose model - will be reported in comet-ml
            models_dir = "./models"
            model_files = [f.replace(".py", "") for f in os.listdir(models_dir) if f.endswith("_model.py")]
            model_name = random.choice(model_files)
            print(f"Randomly selected model: {model_name}")
            model = load_model(model_name)
            if model: 
                experiment_curr = get_experiment(experiment, model_name)
                epochs=experiment_curr.get_parameter("epochs")
                print(epochs)
                batch_size=experiment_curr.get_parameter("batch_size")
                print(batch_size)
                model, history = train_and_evaluate(root_dir, model, epochs, batch_size, experiment)
                opt_metric_value = history.history['val_loss'][-1]
                print(f"Opt Metric Value: {opt_metric_value}")
                experiment.log_metric(opt_metric, opt_metric_value)
                experiment.end()
            else:
                print("Failed:", model_name)