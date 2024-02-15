# <div align="center">训练模型</div>

``` python
import time
import gc
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset

kfold = True
n_splits = 1 if not kfold else 10
random_state = 2023
random_state_list = [2140]  # used by split_data [71]
n_estimators = 9999  # 9999
early_stopping_rounds = 200
verbose = False
device = 'cpu'

splitter = Splitter(kfold = kfold, n_splits = n_splits)

# Initialize an array for storing test predictions
test_predss = np.zeros(X_test.shape[0])
ensemble_score = []
weights = []
trained_models = {'ann': [], 'xgb': [], 'lgb': [], 'cat': [], 'cat_sym': [], 'cat_dep': [], 'lg': [], 'rf':  [], 'hgb': [], 'gbm': [], 'svc': [], 'knn': [], 'mlp': [], 'et': [], 'dt': [], 'ada': [], 'nb': []}

for i, (X_train_, X_val, y_train_, y_val) in enumerate(splitter.split_data(X_train, y_train, random_state_list = random_state_list)):
    n = i % n_splits
    m = i // n_splits
    
    # Get a set of Regressor models
    classifier = Classifier(n_estimators, device, random_state)
    models = classifier.models
    
    # Initialize lists to store oof and test predictions for each base model
    oof_preds = []
    test_preds = []
    
    # Loop over each base model and fit it to the training data, evaluate on validation data, and store 
    # predictions
    for name, model in models.items( ):
        start = time.time( )
        if ('cat' in name) or ("lgb" in name) or ("xgb" in name):
            model.fit(X_train_, y_train_, eval_set = [(X_val, y_val)], early_stopping_rounds = early_stopping_rounds, verbose = verbose)
        elif name in 'ann':
            epochs = 50
            batch_size = 5
            
            X_train_tensor = torch.Tensor(np.array(X_train_)).to(device)
            y_train_tensor = torch.Tensor(np.array(y_train_)).to(device)
            X_val_tensor = torch.Tensor(np.array(X_val)).to(device)
            y_val_tensor = torch.Tensor(np.array(y_val)).to(device)
            X_test_tensor = torch.Tensor(np.array(X_test)).to(device)
            
            # Create PyTorch datasets
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            
            # Create DataLoader for training and validation
            train_loader = DataLoader(train_dataset, batch_size = 5, shuffle = True)
            val_loader = DataLoader(val_dataset, batch_size = 5, shuffle = False)
            
            model.train( )
            
            for epoch in range(epochs):
                for batch_X, batch_y in train_loader:
                    # Transfer data to GPU if available
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    # Forward pass, backward pass, and optimization
                    optimizer.zero_grad( )
                    outputs = model(batch_X)
                    
                    outputs = outputs.view(-1)
                    batch_y = batch_y.view(outputs.shape)
                    
                    loss = criterion(outputs, batch_y)
                    loss.backward( )
                    optimizer.step( )
                
                # Validation
                model.eval( )
                with torch.no_grad( ):
                    for val_batch_X, val_batch_y in val_loader:
                        val_batch_X, val_batch_y = val_batch_X.to(device), val_batch_y.to(device)
                        val_outputs = model(val_batch_X)
                        
                        val_outputs = val_outputs.view(-1)
                        val_batch_y = val_batch_y.view(val_outputs.shape)
                        
                        val_loss = criterion(val_outputs, val_batch_y)
        
        else:
            model.fit(X_train_, y_train_)
        
        if name in 'ann':
            with torch.no_grad( ):
                model.eval( )  # Set the model to evaluation mode
                test_pred = model(X_test_tensor).cpu( ).numpy( )[:, 0]
                y_val_pred = model(X_val_tensor).cpu( ).numpy( )[:, 0]
        else:
            test_pred = model.predict_proba(X_test)[:, 1]
            y_val_pred = model.predict_proba(X_val)[:, 1]
        
        score = accuracy_score(y_val, acc_cutoff_class(y_val, y_val_pred))
        print(f'{name} [FOLD-{n} SEED-{random_state_list[m]}] Accuracy score: {score:.5f} Accuracy time: {time.time( ) - start:.2f}s')
        
        oof_preds.append(y_val_pred)
        test_preds.append(test_pred)
        
        if name in trained_models.keys( ):
            trained_models[f'{name}'].append(deepcopy(model))
    
    # Use Optuna to find the best ensemble weights
    optweights = OptunaWeights(random_state = random_state)
    y_val_pred = optweights.fit_predict(y_val.values, oof_preds)
    
    score = accuracy_score(y_val, acc_cutoff_class(y_val, y_val_pred))
    print(f'Ensemble [FOLD-{n} SEED-{random_state_list[m]}] Accuracy score {score:.5f}')
    
    ensemble_score.append(score)
    weights.append(optweights.weights)
    
    test_predss += optweights.predict(test_preds) / (n_splits * len(random_state_list))
    
    gc.collect( )
```