from ecg_classifier import train_and_evaluate, load_model

if __name__ == "__main__":
    root_dir = "./csv_ecgs"
    model_name = input("Enter the model to use (e.g., 'gru_model', 'lstm_model', 'bigru_model', 'bilstm_model'): ")
    
    print(f"Using model: {model_name}")
    model = load_model(model_name)
    
    if model:
        train_and_evaluate(root_dir, model)