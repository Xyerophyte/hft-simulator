"""
Machine learning models for price prediction.
Implements LSTM-based models for time series forecasting.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
import pickle


class PriceLSTM(nn.Module):
    """
    LSTM model for price direction prediction.
    Binary classification: up (1) or down (0).
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layer
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
        """
        super(PriceLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Fully connected layers
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc1 = nn.Linear(fc_input_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Predictions of shape (batch_size, 1)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out
    
    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10
    ) -> dict:
        """
        Train the model with given data.
        
        Args:
            X: Training features of shape (n_samples, sequence_length, n_features)
            y: Training labels of shape (n_samples,)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            learning_rate: Learning rate for optimizer
            early_stopping_patience: Stop if no improvement for N epochs
            
        Returns:
            Dictionary with training history
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Normalize features
        scaler = StandardScaler()
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_val_flat = X_val.reshape(-1, X_val.shape[-1])
        scaler.fit(X_train_flat)
        
        X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape)
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train_scaled).to(device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
        X_val_t = torch.FloatTensor(X_val_scaled).to(device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)
        
        # Setup training
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training on {device} with {len(X_train)} samples...")
        
        for epoch in range(epochs):
            # Training
            self.train()
            total_loss = 0
            
            indices = torch.randperm(len(X_train_t))
            for i in range(0, len(X_train_t), batch_size):
                batch_idx = indices[i:i + batch_size]
                X_batch = X_train_t[batch_idx]
                y_batch = y_train_t[batch_idx]
                
                optimizer.zero_grad()
                outputs = self(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            train_loss = total_loss / (len(X_train_t) // batch_size)
            
            # Validation
            self.eval()
            with torch.no_grad():
                val_outputs = self(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
                val_preds = (val_outputs > 0.5).float()
                val_accuracy = (val_preds == y_val_t).float().mean().item()
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        self._scaler = scaler
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features of shape (n_samples, sequence_length, n_features)
            
        Returns:
            Predictions as numpy array
        """
        device = next(self.parameters()).device
        
        # Scale if scaler exists
        if hasattr(self, '_scaler'):
            X_flat = X.reshape(-1, X.shape[-1])
            X_scaled = self._scaler.transform(X_flat).reshape(X.shape)
        else:
            X_scaled = X
        
        X_t = torch.FloatTensor(X_scaled).to(device)
        
        self.eval()
        with torch.no_grad():
            outputs = self(X_t)
        
        return outputs.cpu().numpy()


class ModelTrainer:
    """
    Trains and evaluates LSTM models.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        device: str = None
    ):
        """
        Initialize trainer.
        
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            device: Device to train on (cuda/cpu)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = PriceLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )
        
        # Scaler for input normalization
        self.scaler = StandardScaler()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def prepare_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: int = 20,
        fit_scaler: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare sequences for LSTM training.
        
        Args:
            X: Feature array
            y: Target array
            sequence_length: Length of sequences
            fit_scaler: Whether to fit scaler on data
            
        Returns:
            Tuple of (X_sequences, y_sequences) as tensors
        """
        # Scale features
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Create sequences
        X_seq = []
        y_seq = []
        
        for i in range(len(X_scaled) - sequence_length):
            X_seq.append(X_scaled[i:i + sequence_length])
            y_seq.append(y[i + sequence_length])
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(np.array(X_seq)).to(self.device)
        y_tensor = torch.FloatTensor(np.array(y_seq)).unsqueeze(1).to(self.device)
        
        return X_tensor, y_tensor
    
    def train_epoch(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        batch_size: int = 32
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            X_train: Training features
            y_train: Training labels
            batch_size: Batch size
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Create batches
        num_samples = X_train.shape[0]
        indices = torch.randperm(num_samples)
        
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def evaluate(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Evaluate model on validation set.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(X_val)
            loss = self.criterion(outputs, y_val)
            
            # Calculate accuracy
            predictions = (outputs > 0.5).float()
            correct = (predictions == y_val).sum().item()
            accuracy = correct / y_val.shape[0]
        
        return loss.item(), accuracy
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sequence_length: int = 20,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10
    ) -> dict:
        """
        Train model with early stopping.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            sequence_length: Sequence length for LSTM
            epochs: Number of epochs
            batch_size: Batch size
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history dictionary
        """
        print(f"\nPreparing sequences (length={sequence_length})...")
        X_train_seq, y_train_seq = self.prepare_sequences(
            X_train, y_train, sequence_length, fit_scaler=True
        )
        X_val_seq, y_val_seq = self.prepare_sequences(
            X_val, y_val, sequence_length, fit_scaler=False
        )
        
        print(f"Training samples: {X_train_seq.shape[0]}")
        print(f"Validation samples: {X_val_seq.shape[0]}")
        print(f"\nStarting training for {epochs} epochs...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(X_train_seq, y_train_seq, batch_size)
            
            # Evaluate
            val_loss, val_accuracy = self.evaluate(X_val_seq, y_val_seq)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_accuracy:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_model('best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Final validation accuracy: {val_accuracy:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': best_val_loss
        }
    
    def predict(
        self,
        X: np.ndarray,
        sequence_length: int = 20
    ) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features
            sequence_length: Sequence length
            
        Returns:
            Predictions array
        """
        self.model.eval()
        
        # Prepare sequences
        X_seq, _ = self.prepare_sequences(
            X, np.zeros(len(X)), sequence_length, fit_scaler=False
        )
        
        with torch.no_grad():
            outputs = self.model(X_seq)
            predictions = outputs.cpu().numpy()
        
        return predictions
    
    def save_model(self, filepath: str):
        """Save model and scaler."""
        torch.save({
            'model_state': self.model.state_dict(),
            'scaler': self.scaler,
            'train_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_accuracies': self.val_accuracies
            }
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model and scaler."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.scaler = checkpoint['scaler']
        if 'train_history' in checkpoint:
            history = checkpoint['train_history']
            self.train_losses = history['train_losses']
            self.val_losses = history['val_losses']
            self.val_accuracies = history['val_accuracies']


# Example usage
if __name__ == "__main__":
    # Create dummy data for testing
    np.random.seed(42)
    
    # Features and labels
    n_samples = 1000
    n_features = 20
    X = np.random.randn(n_samples, n_features)
    y = (np.random.randn(n_samples) > 0).astype(float)
    
    # Train/val split
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Initialize trainer
    trainer = ModelTrainer(
        input_size=n_features,
        hidden_size=32,
        num_layers=2,
        learning_rate=0.001
    )
    
    # Train model
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        sequence_length=20,
        epochs=20,
        batch_size=32
    )
    
    print("\nTraining completed!")
    print(f"Final validation accuracy: {history['val_accuracies'][-1]:.4f}")