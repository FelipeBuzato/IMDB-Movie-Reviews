import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

class Trainer:
    def __init__(self, optimizer="adam", lr=1e-3, epochs=10, batch_size=256, criterion="cross entropy", random_state=42, device="gpu"):
        self.optimizer = optimizer
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = criterion
        self.random_state = random_state

        if(device == "gpu"):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif(device == "cpu"):
            self.device = torch.device("cpu")
        else: raise ValueError(f"device type {device} not found.")
        self.set_seed()

        self.display = True


    def set_seed(self):
        torch.manual_seed(self.random_state)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def get_criterion(self):
        if(self.criterion == "cross entropy"):
            return nn.CrossEntropyLoss()
        
        raise ValueError(f"Criterion {self.criterion} not found.")
    

    def get_optimizer(self, model):
        if(self.optimizer == "adam"):
            return optim.Adam(model.parameters(), lr=self.lr)
        
        raise ValueError(f"Optimizer {self.optimizer} not found.")
    

    def full_gradient_descent(self, model, X, y, optimizer, criterion):
        loss_curve = []
        
        for epoch in range(self.epochs): 
            # reset gradients so they won't accumulate from previous iteration   
            optimizer.zero_grad(set_to_none=True)
            
            # forward propagation and loss computation
            output = model(X)
            loss = criterion(output, y)
    
            # back propagation
            loss.backward()
            epoch_loss = loss.item()
    
            # optimization step
            optimizer.step()
        
            loss_curve.append(epoch_loss)
            
            if(epoch%20 == 0 and self.display):
                print(f"Loss after Epoch {epoch+1}: {epoch_loss:.4f}.")
        
        return loss_curve
    

    def stochastic_gradient_descent(self, model, X, y, optimizer, criterion):
        loss_curve = []
        print("Model device:", next(model.parameters()).device)
        
        for epoch in range(self.epochs):
            # shuffle every epoch
            indices = torch.randperm(X.shape[0], device=self.device)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
        
            epoch_loss = 0
            
            i = 0
            # train in batches:
            for X_batch, y_batch in zip(X_shuffled.split(self.batch_size), y_shuffled.split(self.batch_size)): 
                i += 1
                if(i == 1 and epoch == 0):
                    print("X device:", X_batch.device)
                    print("y device:", y_batch.device)

                # reset gradients so they won't accumulate from previous iteration
                optimizer.zero_grad(set_to_none=True)
                
                # forward propagation and loss computation
                output = model(X_batch)
                loss = criterion(output, y_batch)
        
                # back propagation
                loss.backward()
                epoch_loss += loss.item() * X_batch.shape[0]

                if(i==1 and epoch == 0):
                    print("fc:", model.fc.weight.grad.norm().item())
                    print("rnn:", model.lstm.weight_ih_l0.grad.norm().item())

                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            print(name, param.grad.norm().item())
        
                # optimization step
                optimizer.step()
        
            epoch_loss /= X.shape[0]
            loss_curve.append(epoch_loss)
            
            if(epoch%20 == 0 and self.display):
                print(f"Loss after Epoch {epoch+1}: {epoch_loss:.4f}.")
        
        return loss_curve
    

    def fit(self, model, X, y):
        start_training_time = time.time()

        # send model, X and y to device
        model = model.to(self.device)
        y = torch.as_tensor(y.values.copy(), dtype=torch.long, device=self.device)
        X = X.to(self.device)
            
        #y = torch.as_tensor(y.values.copy())

        model.train()
        if(self.display): print(f"Training Activated ? {model.training}")
        
        # define loss function and optimizer
        criterion = self.get_criterion()
        optimizer = self.get_optimizer(model=model)

        if(self.batch_size is not None):
            loss_curve = self.stochastic_gradient_descent(model, X, y, optimizer=optimizer, criterion=criterion)
        else:
            loss_curve = self.full_gradient_descent(model, X, y, optimizer=optimizer, criterion=criterion)

        training_time = time.time() - start_training_time
        
        if(self.display): 
            print(f"Loss after gradient descent: {loss_curve[-1]:.6f}.")
            print(f"Total Training time: {training_time}.")

        model.loss_curve_ = loss_curve

        return self


    def predict(self, model, X):
        model.eval()
        if(self.display): print(f"Training Activated ? {model.training}")

        X = X.to(self.device)
        print("predict X device: ", X.device)

        preds = []
        with torch.no_grad():
            for X_batch in X.split(self.batch_size):
                output = model(X_batch)
                preds.append(output.argmax(dim=1))

        return torch.cat(preds).cpu().numpy()
    

    def predict_proba(self, model, X):
        model.eval()
        if(self.display): print(f"Training Activated ? {model.training}")
        
        X = X.to(self.device)

        probs = []
        with torch.no_grad():
            for X_batch in X.split(self.batch_size):
                output = model(X_batch)
                probs.append(torch.softmax(output, dim=1))

        return probs.cpu().numpy()