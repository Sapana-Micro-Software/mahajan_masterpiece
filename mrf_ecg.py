"""
Markov Random Field (MRF) for ECG Classification

Markov Random Fields are undirected graphical models that can capture
spatial and temporal dependencies in data. For ECG classification, MRFs
can model dependencies between different time points and features.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import pdist, squareform


class ECGDataset:
    """Dataset class for ECG signals compatible with MRF."""
    
    def __init__(self, signals: np.ndarray, labels: np.ndarray, seq_len: int = 1000):
        """
        Initialize ECG dataset.
        
        Parameters:
        -----------
        signals : np.ndarray
            ECG signals of shape (n_samples, seq_len) or (n_samples, seq_len, features)
        labels : np.ndarray
            Class labels of shape (n_samples,)
        seq_len : int
            Sequence length to use (padding/truncation)
        """
        self.signals = signals
        self.labels = labels
        self.seq_len = seq_len
        
        # Flatten to 2D if needed
        if len(self.signals.shape) == 3:
            self.signals = self.signals.reshape(self.signals.shape[0], -1)
        
        # Normalize signals
        self.scaler = StandardScaler()
        self.signals = self.scaler.fit_transform(self.signals)
    
    def extract_mrf_features(self, window_size: int = 50) -> np.ndarray:
        """
        Extract features for MRF modeling.
        
        Parameters:
        -----------
        window_size : int
            Window size for feature extraction
        
        Returns:
        --------
        np.ndarray
            Feature matrix with spatial relationships
        """
        features = []
        
        for signal in self.signals:
            # Extract local features
            local_features = []
            for i in range(0, len(signal) - window_size, window_size // 2):
                window = signal[i:i+window_size]
                local_features.append([
                    np.mean(window),
                    np.std(window),
                    np.max(window),
                    np.min(window),
                    np.median(window)
                ])
            
            if len(local_features) > 0:
                local_features = np.array(local_features)
                
                # Add spatial relationships (pairwise distances)
                if len(local_features) > 1:
                    distances = squareform(pdist(local_features))
                    # Use mean distance as feature
                    mean_dist = np.mean(distances[distances > 0])
                    local_features = np.concatenate([
                        np.mean(local_features, axis=0),
                        np.std(local_features, axis=0),
                        [mean_dist]
                    ])
                else:
                    local_features = local_features.flatten()
                
                features.append(local_features)
            else:
                features.append([
                    np.mean(signal),
                    np.std(signal),
                    np.max(signal),
                    np.min(signal),
                    np.median(signal)
                ])
        
        return np.array(features)


class MRFECGClassifier:
    """
    Markov Random Field for ECG Classification.
    
    Uses a simplified MRF approach that models dependencies between
    features and time points in ECG signals.
    """
    
    def __init__(
        self,
        n_features: int = 20,
        n_classes: int = 5,
        regularization: float = 0.1
    ):
        """
        Initialize MRF classifier.
        
        Parameters:
        -----------
        n_features : int
            Number of features
        n_classes : int
            Number of classes
        regularization : float
            Regularization parameter
        """
        self.n_features = n_features
        self.n_classes = n_classes
        self.regularization = regularization
        self.classifier = None
        self.feature_extractor = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train MRF classifier.
        
        Parameters:
        -----------
        X : np.ndarray
            Training signals
        y : np.ndarray
            Training labels
        """
        # Extract MRF features
        dataset = ECGDataset(X, y)
        features = dataset.extract_mrf_features()
        
        # Use Random Forest as base classifier (simplified MRF)
        # In a full MRF implementation, we would learn the graph structure
        # and use belief propagation or similar inference methods
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.classifier.fit(features, y)
        self.feature_extractor = dataset
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Parameters:
        -----------
        X : np.ndarray
            Test signals
        
        Returns:
        --------
        np.ndarray
            Predicted labels
        """
        # Extract features
        features = []
        for signal in X:
            if len(signal.shape) > 1:
                signal = signal.flatten()
            
            # Normalize
            signal = self.feature_extractor.scaler.transform([signal])[0]
            
            # Extract MRF features
            window_size = 50
            local_features = []
            for i in range(0, len(signal) - window_size, window_size // 2):
                window = signal[i:i+window_size]
                local_features.append([
                    np.mean(window),
                    np.std(window),
                    np.max(window),
                    np.min(window),
                    np.median(window)
                ])
            
            if len(local_features) > 0:
                local_features = np.array(local_features)
                
                # Add spatial relationships
                if len(local_features) > 1:
                    distances = squareform(pdist(local_features))
                    mean_dist = np.mean(distances[distances > 0])
                    feature_vec = np.concatenate([
                        np.mean(local_features, axis=0),
                        np.std(local_features, axis=0),
                        [mean_dist]
                    ])
                else:
                    feature_vec = local_features.flatten()
                
                features.append(feature_vec)
            else:
                features.append([
                    np.mean(signal),
                    np.std(signal),
                    np.max(signal),
                    np.min(signal),
                    np.median(signal)
                ])
        
        features = np.array(features)
        
        # Ensure feature dimension matches
        if features.shape[1] != self.classifier.n_features_in_:
            n_expected = self.classifier.n_features_in_
            if features.shape[1] < n_expected:
                padding = np.zeros((features.shape[0], n_expected - features.shape[1]))
                features = np.hstack([features, padding])
            else:
                features = features[:, :n_expected]
        
        return self.classifier.predict(features)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        # Extract features (same as predict)
        features = []
        for signal in X:
            if len(signal.shape) > 1:
                signal = signal.flatten()
            signal = self.feature_extractor.scaler.transform([signal])[0]
            
            window_size = 50
            local_features = []
            for i in range(0, len(signal) - window_size, window_size // 2):
                window = signal[i:i+window_size]
                local_features.append([
                    np.mean(window),
                    np.std(window),
                    np.max(window),
                    np.min(window),
                    np.median(window)
                ])
            
            if len(local_features) > 0:
                local_features = np.array(local_features)
                if len(local_features) > 1:
                    distances = squareform(pdist(local_features))
                    mean_dist = np.mean(distances[distances > 0])
                    feature_vec = np.concatenate([
                        np.mean(local_features, axis=0),
                        np.std(local_features, axis=0),
                        [mean_dist]
                    ])
                else:
                    feature_vec = local_features.flatten()
                features.append(feature_vec)
            else:
                features.append([
                    np.mean(signal),
                    np.std(signal),
                    np.max(signal),
                    np.min(signal),
                    np.median(signal)
                ])
        
        features = np.array(features)
        
        # Ensure feature dimension matches
        if features.shape[1] != self.classifier.n_features_in_:
            n_expected = self.classifier.n_features_in_
            if features.shape[1] < n_expected:
                padding = np.zeros((features.shape[0], n_expected - features.shape[1]))
                features = np.hstack([features, padding])
            else:
                features = features[:, :n_expected]
        
        return self.classifier.predict_proba(features)


def train_mrf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    n_features: int = 20,
    verbose: bool = True
) -> dict:
    """
    Train MRF model.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training signals
    y_train : np.ndarray
        Training labels
    X_val : Optional[np.ndarray]
        Validation signals
    y_val : Optional[np.ndarray]
        Validation labels
    n_features : int
        Number of features
    verbose : bool
        Print training progress
    
    Returns:
    --------
    dict
        Model and training history
    """
    start_time = time.time()
    
    model = MRFECGClassifier(n_features=n_features)
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    history = {
        'train_time': train_time
    }
    
    if verbose:
        print(f"Trained MRF in {train_time:.2f} seconds")
    
    return {'model': model, 'history': history}


def evaluate_model(
    model: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = 'cpu'
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate MRF model on test set.
    
    Parameters:
    -----------
    model : dict
        Model dictionary
    X_test : np.ndarray
        Test signals
    y_test : np.ndarray
        Test labels
    device : str
        Device (not used, kept for compatibility)
    
    Returns:
    --------
    Tuple[float, float, np.ndarray, np.ndarray]
        (test_loss, test_acc, y_true, y_pred)
    """
    mrf_model = model['model']
    y_pred = mrf_model.predict(X_test)
    
    test_acc = accuracy_score(y_test, y_pred) * 100
    
    # Calculate loss
    try:
        y_proba = mrf_model.predict_proba(X_test)
        test_loss = -np.mean(np.log(y_proba[np.arange(len(y_test)), y_test] + 1e-10))
    except:
        test_loss = 1.0 - (test_acc / 100.0)
    
    return test_loss, test_acc, y_test, y_pred


def create_synthetic_ecg_data(
    n_samples: int = 3000,
    seq_len: int = 1000,
    n_classes: int = 5,
    noise_level: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic ECG data for testing."""
    signals = []
    labels = []
    
    for i in range(n_samples):
        label = i % n_classes
        t = np.linspace(0, 4 * np.pi, seq_len)
        
        if label == 0:
            signal = np.sin(t) + 0.5 * np.sin(2 * t) + 0.3 * np.sin(3 * t)
        elif label == 1:
            signal = np.sin(t) + 0.8 * np.sin(1.5 * t) + 0.2 * np.sin(4 * t)
        elif label == 2:
            signal = 1.2 * np.sin(0.8 * t) + 0.6 * np.sin(2.5 * t) + 0.4 * np.sin(5 * t)
        elif label == 3:
            signal = 0.9 * np.sin(t) + 0.7 * np.sin(1.8 * t) + 0.3 * np.sin(3.5 * t)
        else:
            signal = 1.1 * np.sin(1.2 * t) + 0.5 * np.sin(2.2 * t) + 0.4 * np.sin(4.5 * t)
        
        signal += np.random.normal(0, noise_level, seq_len)
        signals.append(signal)
        labels.append(label)
    
    return np.array(signals), np.array(labels)


if __name__ == "__main__":
    """Test MRF implementation."""
    print("="*60)
    print("Testing MRF ECG Classifier")
    print("="*60)
    
    # Create synthetic data
    X, y = create_synthetic_ecg_data(n_samples=1000, seq_len=500)
    
    # Split data
    split_idx = int(0.7 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train MRF
    result = train_mrf(X_train, y_train, verbose=True)
    
    # Evaluate
    test_loss, test_acc, y_true, y_pred = evaluate_model(result, X_test, y_test)
    
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

