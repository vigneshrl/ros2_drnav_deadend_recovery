import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from collections import deque
import torch

@dataclass
class RecoveryPoint:
    """Class to store information about a recovery point"""
    index: int  # Position in the path
    open_directions: List[int]  # [front, left, right] status (1=open, 0=closed)
    rank: int  # Number of open directions (1-3)
    timestamp: float  # When this point was recorded
    confidence: List[float]  # Model's confidence scores for each direction

class RecoveryPointManager:
    def __init__(self, max_stored_points: int = 2, confidence_threshold: float = 0.5):
        """
        Initialize the recovery point manager
        
        Args:
            max_stored_points: Maximum number of recent recovery points to store
            confidence_threshold: Threshold for considering a path as open (0-1)
        """
        self.recovery_points: List[RecoveryPoint] = []  # All recovery points
        self.recent_points = deque(maxlen=max_stored_points)  # Last N recovery points
        self.current_index = 0
        self.timestamp = 0.0
        self.confidence_threshold = confidence_threshold

    def process_model_output(self, model_output: Dict[str, torch.Tensor]) -> Optional[RecoveryPoint]:
        """
        Process model output to determine if this is a recovery point
        
        Args:
            model_output: Dictionary containing model predictions
                - 'path_status': Tensor of shape [batch_size, 3] with probabilities
                - 'is_dead_end': Tensor of shape [batch_size, 1] with dead end probability
        
        Returns:
            RecoveryPoint if this is a recovery point, None otherwise
        """
        # Get path status probabilities - handle batch dimension
        path_status_tensor = torch.sigmoid(model_output['path_status']).cpu()
        if path_status_tensor.dim() > 1:
            path_probs = path_status_tensor[0].numpy()  # Take first batch element
        else:
            path_probs = path_status_tensor.numpy()
        
        # Convert probabilities to binary decisions using threshold
        open_directions = (path_probs > self.confidence_threshold).astype(int)
        
        # Count number of open directions
        num_open = int(np.sum(open_directions))  # Use np.sum to avoid ambiguity
        
        # If at least one direction is open, this is a recovery point
        if num_open > 0:
            point = RecoveryPoint(
                index=self.current_index,
                open_directions=open_directions.flatten().tolist(),
                rank=num_open,
                timestamp=self.timestamp,
                confidence=path_probs.flatten().tolist()
            )
            
            # Add to list of all recovery points
            self.recovery_points.append(point)
            
            # Add to recent points queue
            self.recent_points.append(point)
            
            return point
        
        return None

    def is_dead_end(self, model_output: Dict[str, torch.Tensor]) -> bool:
        """
        Check if current position is a dead end based on model output
        
        Args:
            model_output: Dictionary containing model predictions
        
        Returns:
            True if all directions are closed (dead end), False otherwise
        """
        # Get path status probabilities - handle batch dimension
        path_status_tensor = torch.sigmoid(model_output['path_status']).cpu()
        if path_status_tensor.dim() > 1:
            path_probs = path_status_tensor[0].numpy()  # Take first batch element
        else:
            path_probs = path_status_tensor.numpy()
        
        # Check if all directions are closed using numpy operations
        return bool(np.all(path_probs <= self.confidence_threshold))

    def get_recovery_points(self) -> Tuple[Optional[RecoveryPoint], Optional[RecoveryPoint]]:
        """
        Get the last two recovery points
        
        Returns:
            Tuple of (second_last_point, last_point)
        """
        if len(self.recent_points) == 0:
            return None, None
        elif len(self.recent_points) == 1:
            return None, self.recent_points[-1]
        else:
            return self.recent_points[-2], self.recent_points[-1]

    def get_best_recovery_point(self) -> Optional[RecoveryPoint]:
        """
        Get the recovery point with highest rank (most open directions)
        
        Returns:
            RecoveryPoint with highest rank, or None if no recovery points exist
        """
        if not self.recovery_points:
            return None
        return max(self.recovery_points, key=lambda x: x.rank)

    def update(self, model_output: Dict[str, torch.Tensor], timestamp: float) -> Optional[RecoveryPoint]:
        """
        Update the manager with new model output
        
        Args:
            model_output: Dictionary containing model predictions
            timestamp: Current timestamp
            
        Returns:
            RecoveryPoint if this is a recovery point, None otherwise
        """
        self.current_index += 1
        self.timestamp = timestamp
        return self.process_model_output(model_output)

    def get_recovery_strategy(self, model_output: Dict[str, torch.Tensor]) -> dict:
        """
        Get recovery strategy based on model output
        
        Args:
            model_output: Dictionary containing model predictions
        
        Returns:
            Dictionary containing recovery strategy information
        """
        if self.is_dead_end(model_output):
            # Get last two recovery points
            second_last, last = self.get_recovery_points()
            
            # Get best recovery point (highest rank)
            best = self.get_best_recovery_point()
            
            # Get confidence scores for current position - handle batch dimension
            path_status_tensor = torch.sigmoid(model_output['path_status']).cpu()
            dead_end_tensor = torch.sigmoid(model_output['is_dead_end']).cpu()
            
            if path_status_tensor.dim() > 1:
                path_probs = path_status_tensor[0].numpy()
            else:
                path_probs = path_status_tensor.numpy()
                
            if dead_end_tensor.dim() > 1:
                dead_end_prob = dead_end_tensor[0].numpy()
            else:
                dead_end_prob = dead_end_tensor.numpy()
            
            return {
                "is_dead_end": True,
                "dead_end_confidence": float(dead_end_prob),
                "path_confidences": path_probs.tolist(),
                "recovery_points": {
                    "second_last": second_last,
                    "last": last,
                    "best": best
                },
                "recommended_action": "Return to last recovery point" if last else "No recovery points available"
            }
        else:
            return {
                "is_dead_end": False,
                "recovery_points": None,
                "recommended_action": "Continue current path"
            }

def process_batch_predictions(model_outputs: List[Dict[str, torch.Tensor]], timestamps: List[float]) -> List[dict]:
    """
    Process a batch of model predictions to identify recovery points and dead ends
    
    Args:
        model_outputs: List of model output dictionaries
        timestamps: List of timestamps corresponding to each prediction
    
    Returns:
        List of recovery strategies for each prediction
    """
    manager = RecoveryPointManager()
    strategies = []
    
    for output, timestamp in zip(model_outputs, timestamps):
        # Update manager with current prediction
        point = manager.update(output, timestamp)
        
        # Get recovery strategy
        strategy = manager.get_recovery_strategy(output)
        strategies.append(strategy)
    
    return strategies

# Example of how to use with model predictions
if __name__ == "__main__":
    # This would be replaced with actual model predictions
    sample_output = {
        'path_status': torch.tensor([[0.8, 0.2, 0.9]]),  # [front, left, right] probabilities
        'is_dead_end': torch.tensor([[0.1]])  # Dead end probability
    }
    
    manager = RecoveryPointManager()
    point = manager.update(sample_output, timestamp=1.0)
    
    if point:
        print(f"Recovery point found at index {point.index}:")
        print(f"  Open directions: Front={point.open_directions[0]}, Left={point.open_directions[1]}, Right={point.open_directions[2]}")
        print(f"  Rank: {point.rank}")
        print(f"  Confidence scores: {point.confidence}")
    
    strategy = manager.get_recovery_strategy(sample_output)
    print("\nRecovery strategy:")
    print(f"  Is dead end: {strategy['is_dead_end']}")
    print(f"  Recommended action: {strategy['recommended_action']}") 