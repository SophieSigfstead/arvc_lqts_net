from comet_ml import Optimizer
import sys
import os
from ecg_classifier import train_and_evaluate, load_model

def get_experiment(experiment, model_name):
    experiment.log_parameter("model_type", f"{model_name}")
    experiment.log_parameter("loss", "sparse_categorical_crossentropy")
    experiment.log_parameter("optimizer", "Adam")
    experiment.log_parameter("validation_frac", 0.15)
    experiment.log_parameter("deterministic", True)
    experiment.log_parameter("activation_fn", "ReLU")
    return experiment

if __name__ == "__main__":
    comet_ml_key = sys.argv[1]
    opt_metric = "validation_loss"

    # Define the hyperparameter search configuration
    config = {
        "algorithm": "random",
        "parameters": {
            "batch_size": {"type": "discrete", "values": [4, 8]},
            "epochs": {"type": "discrete", "values": [6,7,8,9,11,12,13,14,16,17,18,19,22]},
        },
        "spec": {
            "metric": opt_metric,
            "objective": "minimize",
            "maxCombo": 100,
        }
    }

    root_dir = "./csv_ecgs"
    models_dir = "./models"
    model_files = [f.replace(".py", "") for f in os.listdir(models_dir) if f.endswith("_model.py")]

    # Iterate over each model file and run a separate optimizer for each model
    for model_name in model_files:
        print(f"Running optimizer for model: {model_name}")

        # Load the model
        model = load_model(model_name)
        if not model:
            print(f"Failed to load model: {model_name}")
            continue

        # Create a new Optimizer instance for each model
        opt = Optimizer(config)

        # Iterate over each experiment configuration provided by the optimizer
        for experiment in opt.get_experiments(workspace="sophiesigfstead", project_name="lqts-arvc-project-initial-runs", api_key=comet_ml_key):
            try:
                # Log the experiment parameters
                experiment_curr = get_experiment(experiment, model_name)

                # Retrieve hyperparameters from the experiment configuration
                epochs = experiment_curr.get_parameter("epochs")
                batch_size = experiment_curr.get_parameter("batch_size")
                print(f"Training with epochs: {epochs}, batch_size: {batch_size}")

                # Train and evaluate the model
                model, history = train_and_evaluate(root_dir, model, epochs, batch_size, experiment)

                # Log the validation loss to the optimizer
                opt_metric_value = history.history['val_loss'][-1]
                print(f"Validation Loss: {opt_metric_value}")
                experiment.log_metric(opt_metric, opt_metric_value)

            except Exception as e:
                print(f"Error during experiment for model {model_name}: {e}")

            # End the current experiment
            experiment.end()
