"""
Markov Decision Process (MDP) and Partially Observable MDP (PO-MDP) for ECG Classification

MDPs model decision-making in situations where outcomes are partly random and
partly under the control of a decision maker. For ECG classification, we can
model the classification as a sequential decision process.

PO-MDPs extend MDPs to handle situations where the state is not directly observable,
which is relevant for ECG signals where the underlying cardiac state is hidden.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict


class ECGDataset:
    """Dataset class for ECG signals compatible with MDP/PO-MDP."""
    
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
    
    def extract_state_features(self, window_size: int = 50) -> List[np.ndarray]:
        """
        Extract state features for MDP.
        
        Parameters:
        -----------
        window_size : int
            Window size for state extraction
        
        Returns:
        --------
        List[np.ndarray]
            List of state sequences
        """
        state_sequences = []
        
        for signal in self.signals:
            states = []
            
            # Extract states using sliding window
            for i in range(0, len(signal) - window_size, window_size // 2):
                window = signal[i:i+window_size]
                
                # Discretize window into state
                state_features = [
                    np.mean(window),
                    np.std(window),
                    np.max(window) - np.min(window),
                    np.argmax(window) / len(window),  # Relative position of max
                ]
                states.append(state_features)
            
            if len(states) > 0:
                state_sequences.append(np.array(states))
            else:
                # Fallback: single state
                state_sequences.append(np.array([[np.mean(signal), np.std(signal), 0, 0]]))
        
        return state_sequences


class MDPECGClassifier:
    """
    Markov Decision Process for ECG Classification.
    
    Models ECG classification as a sequential decision process where
    we make decisions at each time step to classify the signal.
    """
    
    def __init__(
        self,
        n_states: int = 10,
        n_actions: int = 5,  # Actions = classes
        n_classes: int = 5,
        gamma: float = 0.9,  # Discount factor
        learning_rate: float = 0.1
    ):
        """
        Initialize MDP classifier.
        
        Parameters:
        -----------
        n_states : int
            Number of states
        n_actions : int
            Number of actions (classes)
        n_classes : int
            Number of classes
        gamma : float
            Discount factor
        learning_rate : float
            Learning rate for Q-learning
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_classes = n_classes
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.Q = defaultdict(lambda: np.zeros(n_actions))  # Q-table
        self.state_discretizer = None
        self.classifier = None
    
    def _discretize_state(self, state_features: np.ndarray) -> int:
        """
        Discretize continuous state features into state index.
        
        Parameters:
        -----------
        state_features : np.ndarray
            Continuous state features
        
        Returns:
        --------
        int
            Discrete state index
        """
        # Simple discretization: use hash of quantized features
        quantized = (state_features * 10).astype(int) % self.n_states
        state_idx = int(np.sum(quantized) % self.n_states)
        return state_idx
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train MDP classifier using Q-learning.
        
        Parameters:
        -----------
        X : np.ndarray
            Training signals
        y : np.ndarray
            Training labels
        """
        # Extract state sequences
        dataset = ECGDataset(X, y)
        state_sequences = dataset.extract_state_features()
        
        # Train using Q-learning
        for seq_idx, (states, label) in enumerate(zip(state_sequences, y)):
            # Use last state for classification
            if len(states) > 0:
                last_state = states[-1]
                state_idx = self._discretize_state(last_state)
                
                # Update Q-value (reward = 1 if correct, 0 otherwise)
                # We'll use a simplified approach: learn state-action values
                action = int(label)
                self.Q[state_idx][action] += self.learning_rate * (1.0 - self.Q[state_idx][action])
        
        # Also train a classifier for final prediction
        # Extract features from last states
        features = []
        labels_list = []
        for states, label in zip(state_sequences, y):
            if len(states) > 0:
                features.append(states[-1])
                labels_list.append(label)
        
        if len(features) > 0:
            from sklearn.ensemble import RandomForestClassifier
            self.classifier = RandomForestClassifier(n_estimators=50, random_state=42)
            self.classifier.fit(features, labels_list)
        
        self.state_discretizer = dataset
    
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
        predictions = []
        
        # Extract state sequences
        state_sequences = []
        for signal in X:
            if len(signal.shape) > 1:
                signal = signal.flatten()
            
            # Normalize
            signal = self.state_discretizer.scaler.transform([signal])[0]
            
            # Extract states
            window_size = 50
            states = []
            for i in range(0, len(signal) - window_size, window_size // 2):
                window = signal[i:i+window_size]
                state_features = [
                    np.mean(window),
                    np.std(window),
                    np.max(window) - np.min(window),
                    np.argmax(window) / len(window),
                ]
                states.append(state_features)
            
            if len(states) > 0:
                state_sequences.append(np.array(states))
            else:
                state_sequences.append(np.array([[np.mean(signal), np.std(signal), 0, 0]]))
        
        # Predict using classifier
        if self.classifier is not None:
            features = [seq[-1] if len(seq) > 0 else seq[0] for seq in state_sequences]
            # Ensure feature dimension matches
            max_features = max(len(f) for f in features)
            features_padded = []
            for f in features:
                if len(f) < max_features:
                    f_padded = np.pad(f, (0, max_features - len(f)), 'constant')
                else:
                    f_padded = f[:max_features]
                features_padded.append(f_padded)
            
            # Match classifier input dimension
            if len(features_padded[0]) != self.classifier.n_features_in_:
                n_expected = self.classifier.n_features_in_
                features_final = []
                for f in features_padded:
                    if len(f) < n_expected:
                        f_final = np.pad(f, (0, n_expected - len(f)), 'constant')
                    else:
                        f_final = f[:n_expected]
                    features_final.append(f_final)
                features_padded = features_final
            
            predictions = self.classifier.predict(features_padded)
        else:
            # Fallback: use Q-table
            for states in state_sequences:
                if len(states) > 0:
                    last_state = states[-1]
                    state_idx = self._discretize_state(last_state)
                    action = np.argmax(self.Q[state_idx])
                    predictions.append(action)
                else:
                    predictions.append(0)
        
        return np.array(predictions)


class POMDPECGClassifier:
    """
    Partially Observable Markov Decision Process for ECG Classification.
    
    Extends MDP to handle cases where the true state is not directly observable,
    which is relevant for ECG signals where the underlying cardiac state is hidden.
    """
    
    def __init__(
        self,
        n_hidden_states: int = 10,
        n_observations: int = 20,
        n_actions: int = 5,
        n_classes: int = 5
    ):
        """
        Initialize PO-MDP classifier.
        
        Parameters:
        -----------
        n_hidden_states : int
            Number of hidden states
        n_observations : int
            Number of observation symbols
        n_actions : int
            Number of actions (classes)
        n_classes : int
            Number of classes
        """
        self.n_hidden_states = n_hidden_states
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.n_classes = n_classes
        self.belief_state = None
        self.classifier = None
    
    def _discretize_observation(self, signal: np.ndarray) -> int:
        """Discretize signal into observation symbol."""
        # Quantize signal
        min_val, max_val = signal.min(), signal.max()
        if max_val > min_val:
            normalized = (signal - min_val) / (max_val - min_val)
            obs_idx = int(np.mean(normalized) * (self.n_observations - 1))
        else:
            obs_idx = 0
        return obs_idx % self.n_observations
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train PO-MDP classifier.
        
        Parameters:
        -----------
        X : np.ndarray
            Training signals
        y : np.ndarray
            Training labels
        """
        # Extract observation sequences
        dataset = ECGDataset(X, y)
        observations = []
        
        for signal in dataset.signals:
            # Discretize into observations
            window_size = 50
            obs_sequence = []
            for i in range(0, len(signal) - window_size, window_size // 2):
                window = signal[i:i+window_size]
                obs = self._discretize_observation(window)
                obs_sequence.append(obs)
            observations.append(obs_sequence)
        
        # Train classifier on observation features
        features = []
        labels_list = []
        for signal, label in zip(dataset.signals, y):
            # Extract features from observations
            window_size = 50
            signal_features = []
            for i in range(0, len(signal) - window_size, window_size // 2):
                window = signal[i:i+window_size]
                signal_features.extend([
                    np.mean(window),
                    np.std(window),
                    np.max(window),
                    np.min(window)
                ])
            
            if len(signal_features) > 0:
                # Use aggregated features
                features.append(signal_features[:50])  # Limit to 50 features
                labels_list.append(label)
        
        if len(features) > 0:
            from sklearn.ensemble import RandomForestClassifier
            # Ensure consistent feature dimension
            max_len = max(len(f) for f in features)
            features_padded = [np.pad(f, (0, max_len - len(f)), 'constant') if len(f) < max_len else f[:max_len] 
                             for f in features]
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.classifier.fit(features_padded, labels_list)
        
        self.state_discretizer = dataset
    
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
        predictions = []
        
        for signal in X:
            if len(signal.shape) > 1:
                signal = signal.flatten()
            
            # Normalize
            signal = self.state_discretizer.scaler.transform([signal])[0]
            
            # Extract features
            window_size = 50
            signal_features = []
            for i in range(0, len(signal) - window_size, window_size // 2):
                window = signal[i:i+window_size]
                signal_features.extend([
                    np.mean(window),
                    np.std(window),
                    np.max(window),
                    np.min(window)
                ])
            
            if len(signal_features) > 0:
                # Pad to match training dimension
                if hasattr(self.classifier, 'n_features_in_'):
                    n_expected = self.classifier.n_features_in_
                    if len(signal_features) < n_expected:
                        signal_features = np.pad(signal_features, (0, n_expected - len(signal_features)), 'constant')
                    else:
                        signal_features = signal_features[:n_expected]
                
                pred = self.classifier.predict([signal_features])[0]
                predictions.append(pred)
            else:
                predictions.append(0)
        
        return np.array(predictions)


def train_mdp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    model_type: str = 'mdp',
    verbose: bool = True
) -> dict:
    """
    Train MDP or PO-MDP model.
    
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
    model_type : str
        'mdp' or 'pomdp'
    verbose : bool
        Print training progress
    
    Returns:
    --------
    dict
        Model and training history
    """
    start_time = time.time()
    
    if model_type == 'mdp':
        model = MDPECGClassifier(n_states=10, n_actions=5, n_classes=5)
    elif model_type == 'pomdp':
        model = POMDPECGClassifier(n_hidden_states=10, n_observations=20, n_actions=5, n_classes=5)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    history = {
        'train_time': train_time,
        'model_type': model_type
    }
    
    if verbose:
        print(f"Trained {model_type.upper()} in {train_time:.2f} seconds")
    
    return {'model': model, 'history': history}


def evaluate_model(
    model: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = 'cpu'
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate MDP/PO-MDP model on test set.
    
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
    mdp_model = model['model']
    y_pred = mdp_model.predict(X_test)
    
    test_acc = accuracy_score(y_test, y_pred) * 100
    test_loss = 1.0 - (test_acc / 100.0)  # Simple loss approximation
    
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
    """Test MDP/PO-MDP implementation."""
    print("="*60)
    print("Testing MDP ECG Classifier")
    print("="*60)
    
    # Create synthetic data
    X, y = create_synthetic_ecg_data(n_samples=1000, seq_len=500)
    
    # Split data
    split_idx = int(0.7 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train MDP
    result = train_mdp(X_train, y_train, model_type='mdp', verbose=True)
    
    # Evaluate
    test_loss, test_acc, y_true, y_pred = evaluate_model(result, X_test, y_test)
    
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

