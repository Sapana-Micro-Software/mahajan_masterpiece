"""
Hidden Markov Model (HMM) and Hierarchical HMM for ECG Classification

HMMs are probabilistic models that assume the system being modeled is a Markov process
with unobserved (hidden) states. They are widely used in time-series analysis and
signal processing, making them suitable for ECG classification.

Hierarchical HMMs extend standard HMMs by modeling multiple levels of temporal
structure, allowing for more complex pattern recognition in ECG signals.
"""

import numpy as np
from typing import Tuple, Optional, List
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    print("Warning: hmmlearn not available. Install with: pip install hmmlearn")


class ECGDataset:
    """Dataset class for ECG signals compatible with HMM."""
    
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
        elif len(self.signals.shape) == 2:
            pass  # Already 2D
        else:
            raise ValueError(f"Unexpected signal shape: {self.signals.shape}")
        
        # Normalize signals
        self.scaler = StandardScaler()
        self.signals = self.scaler.fit_transform(self.signals)
        
        # Discretize signals for HMM (convert to observation symbols)
        self.n_symbols = 20  # Number of discrete observation symbols
        self._discretize_signals()
    
    def _discretize_signals(self):
        """Discretize continuous ECG signals into observation symbols."""
        # Quantize signals into discrete symbols
        min_val = self.signals.min()
        max_val = self.signals.max()
        bins = np.linspace(min_val, max_val, self.n_symbols + 1)
        self.discrete_signals = np.digitize(self.signals, bins) - 1
        self.discrete_signals = np.clip(self.discrete_signals, 0, self.n_symbols - 1)
    
    def get_sequences(self) -> List[np.ndarray]:
        """Get list of observation sequences for HMM training."""
        sequences = []
        for i in range(len(self.discrete_signals)):
            seq = self.discrete_signals[i].astype(int)
            sequences.append(seq.reshape(-1, 1))
        return sequences


class HMMECGClassifier:
    """
    Hidden Markov Model for ECG Classification.
    
    Uses a separate HMM for each class, then classifies new sequences
    by computing the likelihood under each model.
    """
    
    def __init__(
        self,
        n_states: int = 5,
        n_symbols: int = 20,
        n_classes: int = 5,
        n_iter: int = 100
    ):
        """
        Initialize HMM classifier.
        
        Parameters:
        -----------
        n_states : int
            Number of hidden states
        n_symbols : int
            Number of observation symbols
        n_classes : int
            Number of classes
        n_iter : int
            Maximum number of EM iterations
        """
        if not HMMLEARN_AVAILABLE:
            raise ImportError("hmmlearn is required. Install with: pip install hmmlearn")
        
        self.n_states = n_states
        self.n_symbols = n_symbols
        self.n_classes = n_classes
        self.n_iter = n_iter
        self.models = {}
        self.classes = None
    
    def fit(self, X: List[np.ndarray], y: np.ndarray):
        """
        Train HMM models for each class.
        
        Parameters:
        -----------
        X : List[np.ndarray]
            List of observation sequences
        y : np.ndarray
            Class labels
        """
        self.classes = np.unique(y)
        
        for class_label in self.classes:
            # Get sequences for this class
            class_sequences = [X[i] for i in range(len(X)) if y[i] == class_label]
            
            if len(class_sequences) == 0:
                continue
            
            # Concatenate sequences for training
            if len(class_sequences) > 0:
                # Train HMM for this class
                model = hmm.MultinomialHMM(n_components=self.n_states, n_iter=self.n_iter)
                model.fit(np.vstack(class_sequences), lengths=[len(seq) for seq in class_sequences])
                self.models[class_label] = model
    
    def predict(self, X: List[np.ndarray]) -> np.ndarray:
        """
        Predict class labels for sequences.
        
        Parameters:
        -----------
        X : List[np.ndarray]
            List of observation sequences
        
        Returns:
        --------
        np.ndarray
            Predicted class labels
        """
        predictions = []
        
        for seq in X:
            scores = {}
            for class_label, model in self.models.items():
                try:
                    score = model.score(seq)
                    scores[class_label] = score
                except:
                    scores[class_label] = -np.inf
            
            # Predict class with highest likelihood
            if scores:
                predicted_class = max(scores, key=scores.get)
            else:
                predicted_class = self.classes[0] if len(self.classes) > 0 else 0
            
            predictions.append(predicted_class)
        
        return np.array(predictions)
    
    def predict_proba(self, X: List[np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : List[np.ndarray]
            List of observation sequences
        
        Returns:
        --------
        np.ndarray
            Class probabilities
        """
        probabilities = []
        
        for seq in X:
            scores = {}
            for class_label, model in self.models.items():
                try:
                    score = model.score(seq)
                    scores[class_label] = score
                except:
                    scores[class_label] = -np.inf
            
            # Convert log-likelihoods to probabilities
            if scores:
                max_score = max(scores.values())
                exp_scores = {k: np.exp(v - max_score) for k, v in scores.items()}
                total = sum(exp_scores.values())
                probs = [exp_scores.get(c, 0) / total if total > 0 else 1.0 / len(self.classes) 
                        for c in self.classes]
            else:
                probs = [1.0 / len(self.classes)] * len(self.classes)
            
            probabilities.append(probs)
        
        return np.array(probabilities)


class HierarchicalHMMECGClassifier:
    """
    Hierarchical Hidden Markov Model for ECG Classification.
    
    Models ECG signals at multiple temporal scales using a hierarchical
    structure with super-states and sub-states.
    """
    
    def __init__(
        self,
        n_super_states: int = 3,
        n_sub_states: int = 5,
        n_symbols: int = 20,
        n_classes: int = 5,
        n_iter: int = 100
    ):
        """
        Initialize Hierarchical HMM.
        
        Parameters:
        -----------
        n_super_states : int
            Number of high-level states
        n_sub_states : int
            Number of low-level states per super-state
        n_symbols : int
            Number of observation symbols
        n_classes : int
            Number of classes
        n_iter : int
            Maximum number of EM iterations
        """
        if not HMMLEARN_AVAILABLE:
            raise ImportError("hmmlearn is required. Install with: pip install hmmlearn")
        
        self.n_super_states = n_super_states
        self.n_sub_states = n_sub_states
        self.n_symbols = n_symbols
        self.n_classes = n_classes
        self.n_iter = n_iter
        self.models = {}
        self.classes = None
    
    def _create_hierarchical_features(self, sequences: List[np.ndarray]) -> List[np.ndarray]:
        """
        Create hierarchical features by segmenting sequences.
        
        Parameters:
        -----------
        sequences : List[np.ndarray]
            Input sequences
        
        Returns:
        --------
        List[np.ndarray]
            Hierarchical feature sequences
        """
        hierarchical_sequences = []
        
        for seq in sequences:
            # Segment sequence into chunks for hierarchical modeling
            chunk_size = max(1, len(seq) // self.n_super_states)
            chunks = []
            
            for i in range(0, len(seq), chunk_size):
                chunk = seq[i:i+chunk_size]
                if len(chunk) > 0:
                    # Aggregate chunk (mean, std, etc.)
                    chunk_features = np.array([
                        np.mean(chunk),
                        np.std(chunk),
                        np.max(chunk),
                        np.min(chunk)
                    ])
                    chunks.append(chunk_features)
            
            if len(chunks) > 0:
                # Discretize hierarchical features
                hierarchical_seq = np.array(chunks)
                hierarchical_seq = (hierarchical_seq - hierarchical_seq.min()) / (hierarchical_seq.max() - hierarchical_seq.min() + 1e-8)
                hierarchical_seq = (hierarchical_seq * (self.n_symbols - 1)).astype(int)
                hierarchical_sequences.append(hierarchical_seq.reshape(-1, 1))
            else:
                hierarchical_sequences.append(seq)
        
        return hierarchical_sequences
    
    def fit(self, X: List[np.ndarray], y: np.ndarray):
        """
        Train Hierarchical HMM models for each class.
        
        Parameters:
        -----------
        X : List[np.ndarray]
            List of observation sequences
        y : np.ndarray
            Class labels
        """
        self.classes = np.unique(y)
        
        # Create hierarchical features
        hierarchical_X = self._create_hierarchical_features(X)
        
        for class_label in self.classes:
            # Get sequences for this class
            class_sequences = [hierarchical_X[i] for i in range(len(hierarchical_X)) if y[i] == class_label]
            
            if len(class_sequences) == 0:
                continue
            
            # Train HMM for this class with more states (hierarchical structure)
            total_states = self.n_super_states * self.n_sub_states
            model = hmm.MultinomialHMM(n_components=total_states, n_iter=self.n_iter)
            model.fit(np.vstack(class_sequences), lengths=[len(seq) for seq in class_sequences])
            self.models[class_label] = model
    
    def predict(self, X: List[np.ndarray]) -> np.ndarray:
        """
        Predict class labels for sequences.
        
        Parameters:
        -----------
        X : List[np.ndarray]
            List of observation sequences
        
        Returns:
        --------
        np.ndarray
            Predicted class labels
        """
        hierarchical_X = self._create_hierarchical_features(X)
        predictions = []
        
        for seq in hierarchical_X:
            scores = {}
            for class_label, model in self.models.items():
                try:
                    score = model.score(seq)
                    scores[class_label] = score
                except:
                    scores[class_label] = -np.inf
            
            if scores:
                predicted_class = max(scores, key=scores.get)
            else:
                predicted_class = self.classes[0] if len(self.classes) > 0 else 0
            
            predictions.append(predicted_class)
        
        return np.array(predictions)


def train_hmm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    model_type: str = 'hmm',
    n_states: int = 5,
    n_iter: int = 100,
    verbose: bool = True
) -> dict:
    """
    Train HMM or Hierarchical HMM model.
    
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
        'hmm' or 'hierarchical_hmm'
    n_states : int
        Number of hidden states
    n_iter : int
        Maximum EM iterations
    verbose : bool
        Print training progress
    
    Returns:
    --------
    dict
        Training history
    """
    if not HMMLEARN_AVAILABLE:
        raise ImportError("hmmlearn is required. Install with: pip install hmmlearn")
    
    # Prepare datasets
    train_dataset = ECGDataset(X_train, y_train)
    train_sequences = train_dataset.get_sequences()
    
    # Initialize model
    if model_type == 'hmm':
        model = HMMECGClassifier(n_states=n_states, n_symbols=train_dataset.n_symbols, n_iter=n_iter)
    elif model_type == 'hierarchical_hmm':
        model = HierarchicalHMMECGClassifier(
            n_super_states=3,
            n_sub_states=n_states,
            n_symbols=train_dataset.n_symbols,
            n_iter=n_iter
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Train model
    start_time = time.time()
    model.fit(train_sequences, y_train)
    train_time = time.time() - start_time
    
    history = {
        'train_time': train_time,
        'model_type': model_type
    }
    
    if verbose:
        print(f"Trained {model_type.upper()} in {train_time:.2f} seconds")
    
    return {'model': model, 'history': history, 'scaler': train_dataset.scaler}


def evaluate_model(
    model: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = 'cpu'
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate HMM model on test set.
    
    Parameters:
    -----------
    model : dict
        Model dictionary with 'model' and 'scaler' keys
    X_test : np.ndarray
        Test signals
    y_test : np.ndarray
        Test labels
    device : str
        Device (not used for HMM, kept for compatibility)
    
    Returns:
    --------
    Tuple[float, float, np.ndarray, np.ndarray]
        (test_loss, test_acc, y_true, y_pred)
    """
    # Prepare test dataset
    test_dataset = ECGDataset(X_test, y_test)
    test_sequences = test_dataset.get_sequences()
    
    # Predict
    hmm_model = model['model']
    y_pred = hmm_model.predict(test_sequences)
    
    # Calculate accuracy
    test_acc = accuracy_score(y_test, y_pred) * 100
    
    # Calculate loss (negative log-likelihood)
    test_loss = 0.0
    for i, seq in enumerate(test_sequences):
        try:
            scores = {}
            for class_label, hmm_model_class in hmm_model.models.items():
                score = hmm_model_class.score(seq)
                scores[class_label] = score
            
            if scores:
                max_score = max(scores.values())
                test_loss -= max_score  # Negative log-likelihood
        except:
            pass
    
    test_loss = test_loss / len(test_sequences) if len(test_sequences) > 0 else 0.0
    
    return test_loss, test_acc, y_test, y_pred


def create_synthetic_ecg_data(
    n_samples: int = 3000,
    seq_len: int = 1000,
    n_classes: int = 5,
    noise_level: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic ECG data for testing.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    seq_len : int
        Sequence length
    n_classes : int
        Number of classes
    noise_level : float
        Noise level
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (signals, labels)
    """
    signals = []
    labels = []
    
    for i in range(n_samples):
        label = i % n_classes
        t = np.linspace(0, 4 * np.pi, seq_len)
        
        # Generate different patterns for different classes
        if label == 0:  # Normal
            signal = np.sin(t) + 0.5 * np.sin(2 * t) + 0.3 * np.sin(3 * t)
        elif label == 1:  # APC
            signal = np.sin(t) + 0.8 * np.sin(1.5 * t) + 0.2 * np.sin(4 * t)
        elif label == 2:  # VPC
            signal = 1.2 * np.sin(0.8 * t) + 0.6 * np.sin(2.5 * t) + 0.4 * np.sin(5 * t)
        elif label == 3:  # Fusion
            signal = 0.9 * np.sin(t) + 0.7 * np.sin(1.8 * t) + 0.3 * np.sin(3.5 * t)
        else:  # Other
            signal = 1.1 * np.sin(1.2 * t) + 0.5 * np.sin(2.2 * t) + 0.4 * np.sin(4.5 * t)
        
        # Add noise
        signal += np.random.normal(0, noise_level, seq_len)
        
        signals.append(signal)
        labels.append(label)
    
    return np.array(signals), np.array(labels)


if __name__ == "__main__":
    """Test HMM implementation."""
    if not HMMLEARN_AVAILABLE:
        print("Please install hmmlearn: pip install hmmlearn")
        exit(1)
    
    print("="*60)
    print("Testing HMM ECG Classifier")
    print("="*60)
    
    # Create synthetic data
    X, y = create_synthetic_ecg_data(n_samples=1000, seq_len=500)
    
    # Split data
    split_idx = int(0.7 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train HMM
    result = train_hmm(X_train, y_train, model_type='hmm', n_states=5, verbose=True)
    model = result['model']
    
    # Evaluate
    test_loss, test_acc, y_true, y_pred = evaluate_model(result, X_test, y_test)
    
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"\nClassification Report:")
    from sklearn.metrics import classification_report
    print(classification_report(y_true, y_pred))

