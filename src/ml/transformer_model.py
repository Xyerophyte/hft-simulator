"""
Transformer-based model for price prediction.
Uses self-attention mechanism for capturing patterns in time series.
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Tuple, Optional, Dict
from sklearn.preprocessing import StandardScaler


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PriceTransformer(nn.Module):
    """
    Transformer model for price direction prediction.
    
    Uses multi-head self-attention to capture temporal patterns.
    
    Parameters:
        input_size: Number of input features
        d_model: Transformer model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer encoder layers
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_len: int = 100
    ):
        super().__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.fc1 = nn.Linear(d_model, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # For training
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, input_size)
            
        Returns:
            Predictions (batch, 1)
        """
        # Project to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Use last timestep output
        x = x[:, -1, :]
        
        # Classification head
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x
    
    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10
    ) -> Dict:
        """
        Train the transformer model.
        
        Args:
            X: Features (n_samples, seq_len, n_features)
            y: Labels (n_samples,)
            epochs: Number of epochs
            batch_size: Batch size
            validation_split: Validation fraction
            learning_rate: Learning rate
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history dict
        """
        self.to(self.device)
        
        # Split data
        n_samples = len(X)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)
        
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]
        
        # Scale features
        X_shape = X.shape
        X_flat = X.reshape(-1, X_shape[-1])
        X_scaled = self.scaler.fit_transform(X_flat).reshape(X_shape)
        
        X_train = torch.FloatTensor(X_scaled[train_idx]).to(self.device)
        y_train = torch.FloatTensor(y[train_idx]).to(self.device)
        X_val = torch.FloatTensor(X_scaled[val_idx]).to(self.device)
        y_val = torch.FloatTensor(y[val_idx]).to(self.device)
        
        # Training setup
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training Transformer on {self.device} with {len(train_idx)} samples...")
        
        for epoch in range(epochs):
            self.train()
            train_losses = []
            
            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size].unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_losses.append(loss.item())
            
            # Validation
            self.eval()
            with torch.no_grad():
                val_outputs = self(X_val)
                val_loss = criterion(val_outputs, y_val.unsqueeze(1)).item()
                val_preds = (val_outputs > 0.5).float()
                val_acc = (val_preds.squeeze() == y_val).float().mean().item()
            
            train_loss = np.mean(train_losses)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        self.eval()
        self.to(self.device)
        
        # Scale
        X_shape = X.shape
        X_flat = X.reshape(-1, X_shape[-1])
        X_scaled = self.scaler.transform(X_flat).reshape(X_shape)
        
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            outputs = self(X_tensor)
        
        return outputs.cpu().numpy()


class EnsembleModel:
    """
    Ensemble of LSTM and Transformer models.
    
    Combines predictions using weighted average.
    """
    
    def __init__(
        self,
        input_size: int,
        lstm_weight: float = 0.5,
        transformer_weight: float = 0.5
    ):
        from .models import PriceLSTM
        
        self.lstm = PriceLSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=2
        )
        
        self.transformer = PriceTransformer(
            input_size=input_size,
            d_model=64,
            nhead=4,
            num_layers=2
        )
        
        # Normalize weights
        total = lstm_weight + transformer_weight
        self.lstm_weight = lstm_weight / total
        self.transformer_weight = transformer_weight / total
        
        self.is_trained = False
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 30,
        **kwargs
    ) -> Dict:
        """Train both models."""
        
        print("=" * 50)
        print("Training LSTM...")
        print("=" * 50)
        lstm_history = self.lstm.train_model(X, y, epochs=epochs, **kwargs)
        
        print("\n" + "=" * 50)
        print("Training Transformer...")
        print("=" * 50)
        transformer_history = self.transformer.train_model(X, y, epochs=epochs, **kwargs)
        
        self.is_trained = True
        
        return {
            'lstm': lstm_history,
            'transformer': transformer_history
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble predictions."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        lstm_preds = self.lstm.predict(X)
        transformer_preds = self.transformer.predict(X)
        
        # Weighted average
        ensemble = (self.lstm_weight * lstm_preds + 
                   self.transformer_weight * transformer_preds)
        
        return ensemble
    
    def get_model_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get individual model predictions."""
        return {
            'lstm': self.lstm.predict(X),
            'transformer': self.transformer.predict(X),
            'ensemble': self.predict(X)
        }
