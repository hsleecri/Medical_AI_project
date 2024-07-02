from imports import *

def train_initial_model(X_train, y_train, model_type='XGBoost', seed=42, num_epochs=100, verbose=False):
    """
    Trains an initial model.

    Args:
    X_train (DataFrame): Training features.
    y_train (Series): Training labels.
    model_type (str): The type of model to use ('XGBoost' or 'MLP').
    seed (int): Seed for reproducibility.
    num_epochs (int): Number of epochs to train (for MLP).
    verbose (bool): If True, prints training details.

    Returns:
    model: Trained model.
    """

    if model_type == 'XGBoost':
        model = XGBClassifier(random_state=seed, eval_metric='logloss')
        eval_set = [(X_train, y_train)]
        model.fit(X_train, y_train, eval_set=eval_set, verbose=verbose)
        if verbose:
            eval_results = model.evals_result()
            print("XGBoost training log loss per epoch:")
            for epoch, loss in enumerate(eval_results['validation_0']['logloss']):
                print(f"Epoch {epoch+1}: Log Loss = {loss:.4f}")
    elif model_type == 'MLP':
        class PrintLoss(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs is not None:
                    print(f"Epoch {epoch+1}: Loss = {logs['loss']:.4f}")
        
        model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, warm_start=True, random_state=seed)
        for epoch in range(num_epochs):
            model.fit(X_train, y_train)
            if verbose:
                y_pred_proba = model.predict_proba(X_train)
                loss = log_loss(y_train, y_pred_proba)
                print(f"Epoch {epoch+1}: Log Loss = {loss:.4f}")
    else:
        raise ValueError("model_type must be either 'XGBoost' or 'MLP'")
    
    return model