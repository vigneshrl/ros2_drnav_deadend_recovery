import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
import random
import matplotlib
matplotlib.use('Agg')  
from matplotlib import pyplot as plt
import wandb
# Fix torcheval import
from torcheval.metrics import functional as metrics_functional
import torch.nn.functional as F

# Fix import for DeadEndDetectionModel - direct import since we're in the same directory
try:
    from model_CA import DeadEndDetectionModel
except ImportError:
    print("Warning: Could not import DeadEndDetectionModel, will need to import later")
    DeadEndDetectionModel = None  # Initialize as None to avoid reference errors

# Set CUDA memory management options to help with fragmentation 
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import gc
import torch

def optimize_cuda_memory():
    """Configure CUDA for optimal memory usage"""
    # Empty CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        
        # Set memory allocation options
        torch.cuda.set_per_process_memory_fraction(0.85)  # Use at most 85% of available VRAM
        
        # Use deterministic algorithms for better memory consistency
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Print memory status
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Initial allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Initial cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    else:
        print("CUDA not available, using CPU")

def memory_cleanup():
    """Release memory after operations"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"After cleanup - Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

class DeadEndDataset(Dataset):
    def __init__(self, data_root, split='train', transform=None, num_points=4096, train_ratio=0.8, val_ratio=0.2, inference_mode=True):
        """
        Dataset for dead end detection with multimodal data
        
        Args:
            data_root: Path to dataset root directory containing bag folders
            split: 'train', 'val', or 'test'
            transform: Image transformations
            num_points: Number of points to sample from LiDAR data
            train_ratio: Ratio of data to use for training
            val_ratio: Ratio of data to use for validation
            inference_mode: If True, don't load annotations (use for new data
        """
        self.data_root = data_root
        self.split = split
        self.transform = transform if transform else self.get_default_transform()
        self.num_points = num_points
        self.inference_mode = inference_mode
        self.annotations ={}
        # self.annotation_file = ""
        
        # Find all bag directories
        self.bag_dirs = [d for d in glob.glob(os.path.join(data_root, "bag*"))]
        if not self.bag_dirs:
            raise FileNotFoundError(f"No bag directories found in {data_root}")
        
        # Initialize list to store all sample paths
        self.all_samples = []
        # random.shuffle(self.all_samples)
        # total_samples = len(self.all_samples)
        # val_end = int(total_samples * 0.1)
        # if split == 'train':
        #     self.samples = self.all_samples[val_end:]
        # elif split == 'val':
        #     self.samples = self.all_samples[:val_end]
        self.sample_annotations = {}
        
        # Load samples and annotations from each bag
        for bag_dir in self.bag_dirs:
            bag_name = os.path.basename(bag_dir)
            
            # Find all sample directories in this bag's images folder
            images_dir = os.path.join(bag_dir, "images")
            if not os.path.exists(images_dir):
                print(f"Warning: Images directory not found in {bag_dir}")
                continue
                
            sample_dirs = glob.glob(os.path.join(images_dir, "sample_id_*"))
            sample_ids = [os.path.basename(d) for d in sample_dirs]
            
            # Load annotations for this bag
            if not inference_mode:
                annotations_file = os.path.join(bag_dir, "annotations.json")
                if os.path.exists(annotations_file):
                    with open(annotations_file, 'r') as f:
                        bag_annotations = json.load(f)
                    
                        # Add bag name to sample_id for global uniqueness
                    for sample_id in sample_ids:
                        if sample_id in bag_annotations:
                            # Store with full path reference for later retrieval
                            sample_info = {
                                'sample_id': sample_id,
                                'bag_dir': bag_dir,
                                'annotation': bag_annotations[sample_id]
                            }
                            self.all_samples.append(sample_info)
                            self.sample_annotations[sample_id] = bag_annotations[sample_id]
                else:
                    print(f"Warning: Annotations file not found at {annotations_file}")
            
            #if inference mode, create empty annotations
            else:
                for sample_id in sample_ids:
                    sample_info = {
                        'sample_id': sample_id,
                        'bag_dir': bag_dir,
                        'annotation': {}
                    }
                    self.all_samples.append(sample_info)

        if not inference_mode:
        # Split dataset into train/val/test
            random.seed(42)  # For reproducibility
            random.shuffle(self.all_samples)
            
            total_samples = len(self.all_samples)
            train_end = int(total_samples * train_ratio)
            val_end = int(total_samples * val_ratio)
            
            if split == 'train':
                self.samples = self.all_samples[:train_end]
            elif split == 'val':
                self.samples = self.all_samples[:val_end]
            else:
                raise ValueError(f"Invalid split: {split}")

        else:
            self.samples = self.all_samples
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def get_default_transform(self):
        """Default image transformations"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def read_lidar_bin(self, bin_path):
        """Read LiDAR point cloud from binary file"""
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)  # x, y, z, intensity
        try:
            # Sample points if needed
            if points.shape[0] > self.num_points:
                indices = np.random.choice(points.shape[0], self.num_points, replace=False)
                points = points[indices, :]
        except:
            print(f"Warning: No points found in {bin_path}")
            return None
        return points[:, :3]  # Return only x, y, z
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # idx is the index of the sample in the dataset
        sample_info = self.samples[idx]
        sample_id = sample_info['sample_id']
        bag_dir = sample_info['bag_dir']
        annotation = sample_info['annotation']
        
        # Load images (front, right, left cameras)
        front_img_path = os.path.join(bag_dir, "images", sample_id, "front.jpg")
        right_img_path = os.path.join(bag_dir, "images", sample_id, "side_right.jpg")
        left_img_path = os.path.join(bag_dir, "images", sample_id, "side_left.jpg")
        
        front_img = Image.open(front_img_path).convert('RGB')
        right_img = Image.open(right_img_path).convert('RGB')
        left_img = Image.open(left_img_path).convert('RGB')
        
        if self.transform:
            front_img = self.transform(front_img)
            right_img = self.transform(right_img)
            left_img = self.transform(left_img)
        
        # Load LiDAR data
        front_lidar_path = os.path.join(bag_dir, "lidar", sample_id, "front.bin")
        right_lidar_path = os.path.join(bag_dir, "lidar", sample_id, "side_right.bin")
        left_lidar_path = os.path.join(bag_dir, "lidar", sample_id, "side_left.bin")
        
        front_lidar = torch.from_numpy(self.read_lidar_bin(front_lidar_path)).float()
        right_lidar = torch.from_numpy(self.read_lidar_bin(right_lidar_path)).float()
        left_lidar = torch.from_numpy(self.read_lidar_bin(left_lidar_path)).float()
        
        # Transpose to get shape [3, num_points] for model input
        front_lidar = front_lidar.transpose(0, 1)
        right_lidar = right_lidar.transpose(0, 1)
        left_lidar = left_lidar.transpose(0, 1)

        result = {
            'front_img': front_img,
            'right_img': right_img,
            'left_img': left_img,
            'front_lidar': front_lidar, 
            'right_lidar': right_lidar,
            'left_lidar': left_lidar,
            'sample_id': sample_id
        }

        if annotation is not None:
            # Get annotations
            # Path status (1 if open, 0 if blocked)
            front_open = annotation.get("front_open", 0)
            left_open = annotation.get("side_left_open", 0)
            right_open = annotation.get("side_right_open", 0)
        
            # Is it a dead end? (1 if dead end, 0 if not)
            is_dead_end = annotation.get("is_dead_end", 0)
            
            # # Direction vectors (normalized to unit vectors)
            front_direction = torch.tensor(annotation.get("front_direction", [0.0, 0.0, 0.0]), dtype=torch.float)
            left_direction = torch.tensor(annotation.get("left_direction", [0.0, 0.0, 0.0]), dtype=torch.float)
            right_direction = torch.tensor(annotation.get("right_direction", [0.0, 0.0, 0.0]), dtype=torch.float)
            
            # # # Confidence scores
            # front_confidence = annotation.get("front_confidence", 0.0)
            # left_confidence = annotation.get("left_confidence", 0.0)
            # right_confidence = annotation.get("right_confidence", 0.0)
            
            # # Stack all direction vectors
            direction_vectors = torch.stack([front_direction, left_direction, right_direction])
            
            result.update({
                'path_status': torch.tensor([front_open, left_open, right_open], dtype=torch.float),
                'is_dead_end': torch.tensor([is_dead_end], dtype=torch.float),
                'direction_vectors': direction_vectors,
                # 'confidence_scores': torch.tensor([front_confidence, left_confidence, right_confidence], dtype=torch.float),
        })

        return result

#sample_id is the id of the sample in the dataset 
#but a sample_id is like this sample_id_1059
#so we need to add the _1059 part back to the sample_id as 1059 is the index of the sample in the dataset
#not working 
# def visualize_test_results(model_path, data_root, batch_size=16, num_samples=5, device='cuda'):
#     """
#     Visualize and evaluate model predictions on test data
    
#     Args:
#         model_path: Path to saved model weights
#         data_root: Path to dataset root directory
#         batch_size: Batch size for testing
#         num_samples: Number of samples to visualize
#         device: Device to run inference on (cuda/cpu)
#     """
#     # Set device
#     device = torch.device(device if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
    
#     #Environment setup
#     os.environ['QT_QPA_PLATFORM'] = 'xcb'
#     os.environ['DISPLAY'] = ':0'
    
#     # Load model
#     try:
#         # Import the DeadEndDetectionModel class if not already imported
#         global DeadEndDetectionModel
#         if DeadEndDetectionModel is None:
#             try:
#                 from model_CA import DeadEndDetectionModel
#                 print("Successfully imported DeadEndDetectionModel")
#             except ImportError as e:
#                 print(f"Error importing DeadEndDetectionModel: {e}")
#                 print("Attempting to continue by dynamically loading the module...")
#                 import importlib.util
#                 spec = importlib.util.spec_from_file_location("model_CA", 
#                                                           os.path.join(os.path.dirname(__file__), "model_CA.py"))
#                 model_module = importlib.util.module_from_spec(spec)
#                 spec.loader.exec_module(model_module)
#                 DeadEndDetectionModel = model_module.DeadEndDetectionModel
        
#         # Create the model
#         model = DeadEndDetectionModel()
#         model.load_state_dict(torch.load(model_path, map_location=device))
#         model.to(device)
#         model.eval()
#         print(f"Model loaded successfully from: {model_path}")
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         import traceback
#         traceback.print_exc()
#         return
    
#     # Get test data loader
#     try:
#         test_dataset = DeadEndDataset(data_root, split='test', inference_mode=True)
        
#         test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
#         print(f"Test dataset loaded with {len(test_dataset)} samples")
#     except Exception as e:
#         print(f"Error loading test dataset: {e}")
#         return
    
#     # Setup matplotlib for visualization
#     plt.ion()
#     fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5*num_samples))
    
#     # Initialize metrics
#     correct = 0
#     total = 0
#     results = []
    
#     # Visualize predictions
#     try:
#         with torch.no_grad():
#             sample_count = 0
#             for batch in test_loader:
#                 if sample_count >= num_samples:
#                     break
                
#                 # Get inputs for the model
#                 try:
#                     inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    
#                     # Forward pass
#                     outputs = model(
#                         inputs['front_img'], inputs['right_img'], inputs['left_img'],
#                         inputs['front_lidar'], inputs['right_lidar'], inputs['left_lidar']
#                     )
                    
#                     # Process outputs
#                     path_status = outputs['path_status']
#                     is_dead_end = outputs['is_dead_end']
                    
#                     # For visualization, we'll focus on the 'is_dead_end' output
#                     dead_end_preds = (torch.sigmoid(is_dead_end) > 0.5).cpu().numpy()
                    
#                     # In inference mode, we might not have labels, so handle accordingly
#                     if 'is_dead_end' in inputs:
#                         labels = inputs['is_dead_end'].cpu().numpy()
#                     else:
#                         # If no ground truth, just use predictions for visualization
#                         labels = dead_end_preds
                    
#                     # Visualize sample_count samples
#                     for i in range(min(batch_size, inputs['front_img'].size(0))):
#                         if sample_count >= num_samples:
#                             break
                        
#                         # Denormalize image for visualization
#                         img = inputs['front_img'][i].cpu().permute(1, 2, 0).numpy()
#                         img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
#                         img = np.clip(img, 0, 1)
                        
#                         # Get prediction and sample ID
#                         pred = "Dead End" if dead_end_preds[i][0] == 1 else "Not Dead End"
#                         sample_id = batch['sample_id'][i] if 'sample_id' in batch else f"Sample_{sample_count}"
                        
#                         # Get ground truth if available
#                         if 'is_dead_end' in inputs:
#                             gt = "Dead End" if labels[i][0] == 1 else "Not Dead End"
#                             correct_pred = dead_end_preds[i][0] == labels[i][0]
#                         else:
#                             gt = "Unknown"
#                             correct_pred = None
                        
#                         # Plot
#                         if num_samples == 1:
#                             ax1, ax2 = axes
#                         else:
#                             ax1, ax2 = axes[sample_count]
                        
#                         ax1.imshow(img)
#                         ax1.set_title(f"Sample ID: {sample_id}")
#                         ax1.axis('off')
                        
#                         # Create text plot
#                         ax2.text(0.1, 0.6, f"Prediction: {pred}", fontsize=12)
#                         ax2.text(0.1, 0.2, f"Correct: {correct_pred}", fontsize=12, 
#                                  color='green' if correct_pred else 'red')
#                         ax2.axis('off')
                        
#                         # Save result
#                         results.append({
#                             'sample_id': sample_id,
#                             'prediction': pred,
#                             'correct': correct_pred
#                         })
                        
#                         sample_count += 1
        
#         # Calculate accuracy
#         accuracy = 100 * correct / total if total > 0 else 0
#         print(f"Test Accuracy: {accuracy:.2f}%")
        
#         # Display confusion matrix
#         if results:
#             # y_true = [r['ground_truth'] for r in results]
#             y_pred = [r['prediction'] for r in results]
            
#             # plt.figure(figsize=(8, 6))
#             # # cm = confusion_matrix(y_true, y_pred)
#             # # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Dead End", "Dead End"])
#             # # disp.plot(cmap=plt.cm.Blues)
#             # plt.title("Confusion Matrix")
#             # plt.tight_layout()
#             # plt.show()
            
#     except Exception as e:
#         print(f"Error during visualization: {e}")
#         import traceback
#         traceback.print_exc()
    
#     plt.ioff()
#     plt.show()
    
#     return results
#testtin a new code :::::::

def visualize_test_results(model_path, data_root, batch_size=16, num_samples=5, device='cuda', output_dir=None):
    """
    Visualize and evaluate model predictions on test data
    
    Args:
        model_path: Path to saved model weights
        data_root: Path to dataset root directory
        batch_size: Batch size for testing
        num_samples: Number of samples to visualize
        device: Device to run inference on (cuda/cpu)
        output_dir: Directory to save visualizations (if None, just display)
    
    Returns:
        Dictionary with evaluation results
    """
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(f"Using device: {device}")
    
    # Create output directory if specified
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Load model
    try:
        # Import the DeadEndDetectionModel class if not already imported
        global DeadEndDetectionModel
        if DeadEndDetectionModel is None:
            try:
                from model_CA import DeadEndDetectionModel
                print("Successfully imported DeadEndDetectionModel")
            except ImportError as e:
                print(f"Error importing DeadEndDetectionModel: {e}")
                print("Attempting to continue by dynamically loading the module...")
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "model_CA", 
                    os.path.join(os.path.dirname(__file__), "model_CA.py")
                )
                model_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(model_module)
                DeadEndDetectionModel = model_module.DeadEndDetectionModel
        
        # Create and load the model
        model = DeadEndDetectionModel()
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Determine if we're in inference mode based on directory structure
    annotations_exist = any(os.path.exists(os.path.join(root, "annotations.json")) 
                            for root, _, _ in os.walk(data_root))
    inference_mode = not annotations_exist
    
    if inference_mode:
        print("Running in inference mode (no ground truth data)")
    else:
        print("Running in evaluation mode (with ground truth data)")
    
    # Create dataset and dataloader
    try:
        test_dataset = DeadEndDataset(data_root, split='test', inference_mode=inference_mode)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2,
            persistent_workers=False
        )
        print(f"Test dataset loaded with {len(test_dataset)} samples")
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Initialize metrics
    all_preds = []
    all_labels = []
    results = {}
    
    # Try to import visualization libraries
    try:
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        has_sklearn = True
    except ImportError:
        print("Warning: scikit-learn not available for confusion matrix display")
        has_sklearn = False

    # Function to plot direction vectors
    def plot_direction_vectors(ax, img, direction_vectors, path_probs,view_type='front', scale=50):
        """
        Plot direction vectors as arrows on the image
        Args:
            ax: matplotlib axis
            img: image array
            direction_vectors: [3, 3] tensor with x,y,z directions for front, left, right
            path_status: [3] tensor with binary open/closed status
            scale: scaling factor for arrow length
        """
        h, w = img.shape[:2]
        center = np.array([w/2, h/2])
        view_map = {
        'front': (0, 'blue'),
        'left': (1, 'green'),
        'right': (2, 'red')
        }

        if view_type not in view_map:
            raise ValueError(f'Invalid view_type: {view_type}')   
        
        idx, color = view_map[view_type]
        # prob = float(path_probs[idx])
            # Colors for different directions
        # colors = ['b', 'g', 'r']  # blue for front, green for left, red for right
        # labels = ['Front', 'Left', 'Right']
        if torch.is_tensor(direction_vectors):
            direction_vectors = direction_vectors.detach().cpu().numpy()
        # if torch.is_tensor(path_probs):
            # path_probs = path_probs.detach().cpu().numpy()
        
        # path_probs = path_probs.flatten() #now 1D
        # print(f"After flatten: {path_probs}")
        vec = direction_vectors[idx] 
        prob = float(path_probs[idx])
        # print(f"prob: {prob}")
        # print(prob)
        # for i, (vec, is_open, color, label) in enumerate(zip(direction_vectors, path_probs, colors, labels)):
        if prob > 0.5:  # Only plot arrows for open paths
            # Convert 3D vector to 2D by using x,y components
            direction = np.array([vec[0], vec[2]])
            
            # Normalize and scale the vector
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction) * scale
                
                # Draw arrow
                ax.arrow(center[0], center[1], 
                        direction[0], direction[1],
                        head_width=5, head_length=10, 
                        fc=color, ec=color, 
                        label=f"{view_type.capitalize()}")
        
        ax.legend()
    # Create figure for visualization
    plt.figure(figsize=(12, 6 * num_samples))
    
    # Process batches
    try:
        with torch.no_grad():
            sample_count = 0
            for batch_idx, batch in enumerate(test_loader):
                if sample_count >= num_samples:
                    break
                
                # Move tensors to device
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in batch.items()}
                
                # Forward pass
                outputs = model(
                    inputs['front_img'], inputs['right_img'], inputs['left_img'],
                    inputs['front_lidar'], inputs['right_lidar'], inputs['left_lidar']
                )
                
                # Get predictions for dead end and path status
                # dead_end_probs = (outputs['is_dead_end']).cpu()
                # dead_end_preds = (dead_end_probs > 0.5).float()
                current_batch_size = inputs['front_img'].size(0)
                path_probs = (outputs['path_status']).cpu()
                for i in range(current_batch_size):
                # Get the path probabilities for the current sample
                    path_perce = outputs['path_status'][i]  # shape: [3]
                    # # If you want as numpy:
                    if torch.is_tensor(path_perce):
                        path_perce = path_perce.detach().cpu().numpy()
                    # # Now you can get:
                    front_open_prob = path_perce[0]
                    left_open_prob = path_perce[1]
                    right_open_prob = path_perce[2]
                # print(f"Before flatten: {path_probs}")
                path_preds = (path_probs > 0.5).long()
                dead_end_preds = (path_probs < 0.50).all(dim=1).float()
                # Check if ground truth exists
                has_labels = 'is_dead_end' in inputs and not inference_mode
                
                # If we have ground truth, get labels
                if has_labels:
                    dead_end_labels = inputs['is_dead_end'].cpu().long()
                    path_labels = inputs['path_status'].cpu().long()
                    all_labels.extend(dead_end_labels.view(-1).tolist())
                    all_preds.extend(dead_end_preds.view(-1).tolist())
                
                # Process each sample in the batch
                
                for i in range(current_batch_size):
                    if sample_count >= num_samples:
                        break
                    
                    # Get sample ID
                    sample_id = batch['sample_id'][i]
                    
                    # Convert image for visualization (denormalize)
                    front_img = inputs['front_img'][i].cpu().numpy().transpose(1, 2, 0)
                    front_img = front_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    front_img = np.clip(front_img, 0, 1)
                    
                    right_img = inputs['right_img'][i].cpu().numpy().transpose(1, 2, 0)
                    right_img = right_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    right_img = np.clip(right_img, 0, 1)
                    
                    left_img = inputs['left_img'][i].cpu().numpy().transpose(1, 2, 0)
                    left_img = left_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    left_img = np.clip(left_img, 0, 1)
                    
                    # Get prediction information
                    # is_dead_end = dead_end_preds[i][0].item()
                    is_dead_end = dead_end_preds[i].item()
                    
                    #direction vectors
                    direction_vecs = outputs['direction_vectors'][i]
                    # fig = plt.figure(figsize=(20, 5))
                    # Path status predictions
                    front_open_prob = path_probs[i][0].item()
                    left_open_prob = path_probs[i][1].item()
                    right_open_prob = path_probs[i][2].item()
                    
                    front_open = path_preds[i][0].item()
                    left_open = path_preds[i][1].item()
                    right_open = path_preds[i][2].item()
                    # print(f"front_open_prob: {front_open_prob}, left_open_prob: {left_open_prob}, right_open_prob: {right_open_prob}")
                    
                    # Create result dictionary
                    results[sample_id] = {
                        'front_open': front_open,
                        'side_left_open': left_open,
                        'side_right_open': right_open,
                        'is_dead_end': is_dead_end,
                    }
                    
                    # # Add ground truth if available
                    # if has_labels:
                    #     result.update({
                    #         'dead_end_gt': dead_end_labels[i][0].item(),
                    #         'front_open_gt': path_labels[i][0].item(),
                    #         'left_open_gt': path_labels[i][1].item(),
                    #         'right_open_gt': path_labels[i][2].item(),
                    #         'is_correct': dead_end_preds[i][0] == dead_end_labels[i][0]
                    #     })
                    
                    # results.append(result)
                    
                    # Create individual figure for each sample
                    fig  =plt.figure(figsize=(15, 5))  # Width, Height in inches

                    # Front camera
                    ax1 = plt.subplot(1, 3, 1)  # 1 row, 3 columns, position 1
                    # plt.imshow(front_img)
                    front_img = inputs['front_img'][i].cpu().numpy().transpose(1, 2, 0)
                    front_img = front_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    front_img = np.clip(front_img, 0, 1)
                    ax1.imshow(front_img)
                    plot_direction_vectors(ax1, front_img, direction_vecs, path_perce, view_type='front')
                    ax1.set_title(f"Front - Open: {front_open_prob:.2f}\nNot Open: {1-front_open_prob:.2f}")
                    ax1.axis('off')

                    # Left camera
                    ax2 = plt.subplot(1, 3, 2)
                    left_img = inputs['left_img'][i].cpu().numpy().transpose(1, 2, 0)
                    left_img = left_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    left_img = np.clip(left_img, 0, 1)
                    ax2.imshow(left_img)
                    plot_direction_vectors(ax2, left_img, direction_vecs, path_perce, view_type='left')
                    ax2.set_title(f"Left - Open: {left_open_prob:.2f}\nNot Open: {1-left_open_prob:.2f}")
                    ax2.axis('off')

                    # Right camera 
                    ax3 = plt.subplot(1, 3, 3)
                    right_img = inputs['right_img'][i].cpu().numpy().transpose(1, 2, 0)
                    right_img = right_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    right_img = np.clip(right_img, 0, 1)
                    ax3.imshow(right_img)
                    plot_direction_vectors(ax3, right_img, direction_vecs, path_perce, view_type='right')
                    ax3.set_title(f"Right - Open: {right_open_prob:.2f}\nNot Open: {1-right_open_prob:.2f}")
                    ax3.axis('off')

                    # Add prediction text
                    plt.suptitle(f"Sample {sample_id} - Dead End/OpenPath: {is_dead_end:}", y=0.95)

                    # Save individual figure
                    output_path = os.path.join(output_dir, f'sample_{sample_id}.png')
                    plt.savefig(output_path, bbox_inches='tight')
                    plt.close()  # Critical - prevents memory leaks

                    
            # Adjust layout
            # plt.tight_layout()
            
            # Save or show figure
            if output_dir:
                plt.savefig(os.path.join(output_dir, 'visualization.png'))
                print(f"Visualization saved to {os.path.join(output_dir, 'visualization.png')}")
                #saved all the outputs as json file which as all the samples 
                # 
                predictions_file = os.path.join(output_dir, 'predictions.json')

                try:
                    with open(predictions_file, 'w') as f:
                        json.dump(results, f, indent=4)
                    print(f"Predictions saved to {predictions_file}")
                except Exception as e:
                    print(f"Error saving predictions: {e}")
            
            # plt.show()
            
            # If we have labels, display metrics
            if all_labels and has_sklearn and not inference_mode:
                # Calculate accuracy
                accuracy = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_labels)
                print(f"Accuracy: {accuracy:.4f}")
                
                # Create confusion matrix
                plt.figure(figsize=(8, 6))
                cm = confusion_matrix(all_labels, all_preds)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Dead End", "Dead End"])
                disp.plot(cmap=plt.cm.Blues)
                plt.title("Dead End Detection - Confusion Matrix")
                
                if output_dir:
                    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
                
                plt.tight_layout()
                # plt.show()
                
                # Add metrics to results
                results_summary = {
                    'accuracy': accuracy,
                    'confusion_matrix': cm.tolist(),
                }
                
                # Save results to JSON if output_dir is specified
                if output_dir:
                    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
                        # Filter results to make JSON serializable
                        serializable_results = []
                        for r in results:
                            serializable_r = {k: float(v) if isinstance(v, (torch.Tensor, np.floating)) else v
                                            for k, v in r.items()}
                            serializable_results.append(serializable_r)
                        
                        json.dump({
                            'summary': results_summary,
                            'samples': serializable_results
                        }, f, indent=2)
                        
                    print(f"Results saved to {os.path.join(output_dir, 'results.json')}")
            
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
    
    return results



# Training utilities
# def train_model(model, train_loader, val_loader, num_epochs=30, lr=1e-4, device='cuda', 
#                 save_dir='/home/vicky/IROS2025/DRaM/codes/data/saved_models'):
#     """Train the dead end detection model with metrics tracking"""
#     optimize_cuda_memory()

#     use_amp = device == 'cuda'
    
#     # Initialize Wandb
#     wandb.init(project="dead-end-detection", config={
#         "learning_rate": lr,
#         "epochs": num_epochs,
#         "batch_size": train_loader.batch_size,
#         "mixed_precision" : use_amp
#     })
    
#     model.to(device)
    
#     # Optimizer and scheduler with better stability
#     optimizer = torch.optim.AdamW(
#         model.parameters(), 
#         lr=lr,
#         weight_decay=1e-4,
#         eps=1e-8,  # Prevent division by zero
#         betas=(0.9, 0.999)  # Default, but explicit for clarity
#     )
    
#     # Gradual warmup helps with initial stability
#     scheduler = torch.optim.lr_scheduler.OneCycleLR(
#         optimizer, 
#         max_lr=lr,
#         steps_per_epoch=len(train_loader),
#         epochs=num_epochs,
#         pct_start=0.3,  # Spend 30% of time warming up
#         div_factor=25,  # Initial lr = max_lr/25
#         final_div_factor=10000,  # Final lr = max_lr/10000
#         anneal_strategy='cos'  # Cosine annealing
#     )
    
#     scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

#     # Loss functions
#     bce_loss = torch.nn.BCEWithLogitsLoss()  # Better for numerical stability
#     mse_loss = torch.nn.MSELoss()

#     best_val_f1 = 0.0  # Track best model
#     grad_norm_tracking = []
    
#     # Setup loss tracking for early stopping
#     patience = 5
#     patience_counter = 0
#     best_val_loss = float('inf')

#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0.0
#         path_f1 = 0.0
#         dead_end_f1 = 0.0
#         memory_cleanup()
        
#         # Reset gradient statistics for this epoch
#         grad_norm_tracking = []
#         nan_or_inf_detected = False

#         for batch_idx, batch in enumerate(train_loader):
#             try:
#                 # Skip this batch if NaN/Inf were detected in the previous batch
#                 if nan_or_inf_detected:
#                     nan_or_inf_detected = False
#                     continue
                    
#                 with torch.cuda.amp.autocast(enabled=use_amp):
#                     # Move data to device
#                     inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    
#                     # Forward pass
#                     outputs = model(
#                         inputs['front_img'], inputs['right_img'], inputs['left_img'],
#                         inputs['front_lidar'], inputs['right_lidar'], inputs['left_lidar']
#                     )
                    
#                     # Check for NaN/Inf in model outputs (usually indicates numerical issues)
#                     if any(torch.isnan(v).any() or torch.isinf(v).any() for v in outputs.values() if isinstance(v, torch.Tensor)):
#                         print(f"Warning: NaN/Inf detected in model outputs at batch {batch_idx}. Skipping batch.")
#                         nan_or_inf_detected = True
#                         continue
                    
#                     # Calculate losses
#                     path_loss = bce_loss(outputs['path_status'], inputs['path_status'])
#                     dead_end_loss = bce_loss(outputs['is_dead_end'], inputs['is_dead_end'])
                    
#                     # Direction loss with masking
#                     open_mask = inputs['path_status'].unsqueeze(2)
#                     direction_loss = mse_loss(
#                         outputs['direction_vectors'] * open_mask,
#                         inputs['direction_vectors'] * open_mask
#                     )
                    
#                     total_loss = path_loss + dead_end_loss + direction_loss

#                 # Backward pass
#                 scaler.scale(total_loss).backward()
                
#                 # Clip gradients more aggressively for stability
#                 grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
#                 grad_norm_tracking.append(grad_norm.item())

#                 # Step optimizer with the clipped gradients
#                 scaler.step(optimizer)
#                 scaler.update()
#                 optimizer.zero_grad(set_to_none=True)
#                 scheduler.step()

#                 # Calculate metrics
#                 with torch.no_grad():
#                     # Get predictions (apply sigmoid and threshold)
#                     path_preds = torch.sigmoid(outputs['path_status']) > 0.5
#                     dead_end_preds = torch.sigmoid(outputs['is_dead_end']) > 0.5
                    
#                     # Use binary F1 score for path status (calculated separately for each path)
#                     batch_path_f1 = 0.0
#                     for i in range(inputs['path_status'].size(1)):  # For each path position
#                         # Extract predictions and targets for this path
#                         path_pred = path_preds[:, i].long()
#                         path_target = inputs['path_status'][:, i].long()
                        
#                         # Use binary F1 score (only if we have samples from both classes)
#                         try:
#                             path_pos = metrics_functional.binary_f1_score(
#                                 path_pred, 
#                                 path_target
#                             )
#                             batch_path_f1 += path_pos
#                         except Exception as e:
#                             # If error (e.g., only one class present), just continue
#                             pass
                    
#                     # Average across paths
#                     path_f1 += (batch_path_f1 / inputs['path_status'].size(1)) if batch_path_f1 > 0 else 0
                    
#                     # Use binary F1 for dead end prediction
#                     try:
#                         dead_end_f1 += metrics_functional.binary_f1_score(
#                             dead_end_preds.long().flatten(),
#                             inputs['is_dead_end'].long().flatten()
#                         )
#                     except Exception as e:
#                         # If only one class is present in this batch, skip
#                         pass

#                 train_loss += total_loss.item()

#                 # Log batch metrics
#                 if batch_idx % 10 == 0:
#                     wandb.log({
#                         "batch_loss": total_loss.item(),
#                         "learning_rate": scheduler.get_last_lr()[0],
#                         "grad_norm": grad_norm.item(),
#                         "memory_allocated": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
#                     })
                
#                 # Detect gradient explosion
#                 if grad_norm > 5.0:
#                     # Don't print every time to avoid flooding logs
#                     if batch_idx % 5 == 0:
#                         print(f"WARNING: Large gradient norm detected: {grad_norm:.2f} at batch {batch_idx}")
                        
#                     # If gradients are extremely large, reduce learning rate temporarily
#                     if grad_norm > 20.0:
#                         for param_group in optimizer.param_groups:
#                             param_group['lr'] = param_group['lr'] * 0.5
#                         print(f"ALERT: Extremely large gradients ({grad_norm:.2f}). Reducing learning rate temporarily.")
                
#                 # Periodic memory cleanup during epoch
#                 if batch_idx % 50 == 0 and batch_idx > 0:
#                     memory_cleanup()

#             except torch.cuda.OutOfMemoryError:
#                 print(f"CUDA OOM at batch {batch_idx}. Cleaning memory and skipping batch...")
#                 memory_cleanup()
#                 continue

#         # Epoch metrics
#         train_loss /= len(train_loader)
#         path_f1 /= len(train_loader)
#         dead_end_f1 /= len(train_loader)

#         # Validation
#         val_metrics = evaluate_model(model, val_loader, device)
        
#         # Calculate average validation metrics
#         val_loss = val_metrics['val_loss']
#         path_accuracy = val_metrics['path_f1']
#         dead_end_accuracy = val_metrics['dead_end_f1']
        
#         # Check for early stopping
#         if val_metrics['val_loss'] < best_val_loss:
#             best_val_loss = val_metrics['val_loss']
#             patience_counter = 0
#         else:
#             patience_counter += 1
            
#         if patience_counter >= patience:
#             print(f"Early stopping triggered after {epoch+1} epochs")
#             break
        
#         # Save best model
#         if val_metrics['path_f1'] > best_val_f1:
#             best_val_f1 = val_metrics['path_f1']
#             torch.save(model.state_dict(), os.path.join(save_dir, 'model_best.pth'))
#             print(f"Saved new best model with F1 score: {best_val_f1:.4f}")

#         # Log epoch metrics
#         wandb.log({
#             "epoch": epoch,
#             "train_loss": train_loss,
#             "val_loss": val_metrics['val_loss'],
#             "path_f1": path_f1,
#             "dead_end_f1": dead_end_f1,
#             "val_path_f1": val_metrics['path_f1'],
#             "val_dead_end_f1": val_metrics['dead_end_f1'],
#             "learning_rate": scheduler.get_last_lr()[0],
#             "gradient_mean": sum(grad_norm_tracking) / len(grad_norm_tracking) if grad_norm_tracking else 0,
#             "gradient_max": max(grad_norm_tracking) if grad_norm_tracking else 0
#         })

#         # Print statistics
#         print(f'Epoch {epoch+1}/{num_epochs}')
#         print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_metrics["val_loss"]:.4f}')
#         print(f'Path F1: {path_f1:.4f} | Val Path F1: {val_metrics["path_f1"]:.4f}')
#         print(f'DeadEnd F1: {dead_end_f1:.4f} | Val DeadEnd F1: {val_metrics["dead_end_f1"]:.4f}')
#         print(f'Grad Norm - Mean: {sum(grad_norm_tracking) / len(grad_norm_tracking) if grad_norm_tracking else 0:.2f}, Max: {max(grad_norm_tracking) if grad_norm_tracking else 0:.2f}')

#         # Save checkpoint
#         if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'scheduler_state_dict': scheduler.state_dict(),
#                 'scaler_state_dict': scaler.state_dict(),
#                 'metrics': val_metrics,
#                 'best_val_f1': best_val_f1,
#                 'best_val_loss': best_val_loss,
#                 'patience_counter': patience_counter
#             }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

#         memory_cleanup()

#     wandb.finish()
#     return model
# def focal_loss_dead_end(logits, targets, eps=1e-8, alpha=0.50, gamma=2.0, reduction='mean'):
#     """
#     logits: raw model outputs (no sigmoid)
#     targets: ground truth (0 or 1)
#     """
#     # device = 'cuda'
#     logits = torch.clamp(logits, min=-100, max=100)
#     pos_weight = torch.tensor([0.84], device=logits.device)
#     bce_loss = F.binary_cross_entropy_with_logits(
#         logits, targets, reduction='none', pos_weight=pos_weight)
#     # pt = torch.exp(-bce_loss) 
#     pt = torch.clamp(torch.sigmoid(-bce_loss), min=eps, max=1-eps) 
#     # focal_weight = alpha * (1 - pt) ** gamma
#     # loss = focal_weight * bce_loss
#     return (alpha * (1 - pt)**gamma * bce_loss).mean()
#     # if reduction == 'mean':
#     #     return loss.mean()
#     # elif reduction == 'sum':
#     #     return loss.sum()
#     # else:
#     #     return loss

def focal_loss_dead_end(logits, targets, eps=1e-7, alpha=0.50, gamma=2.0):
    logits = torch.clamp(logits, min=-5.0, max=5.0)
    pos_weight = torch.tensor([0.84], device=logits.device)
    bce_loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction='none', pos_weight=pos_weight
    )
    probs = torch.sigmoid(logits)
    pt = torch.where(targets == 1, probs, 1 - probs)
    pt = torch.clamp(pt, min=eps, max=1.0 - eps)
    focal_weight = (1 - pt) ** gamma
    loss = alpha * focal_weight * bce_loss
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        return None
    return loss.mean()

def train_model(model, train_loader, val_loader, num_epochs=20, lr=1e-5, device='cuda', 
                save_dir='/home/vicky/IROS2025/DRaM/codes/data/saved_models'):
    """Train the dead end detection model with metrics tracking"""
    optimize_cuda_memory()

    use_amp = device == 'cuda'
    
    # Initialize Wandb (optional - you can comment this out if not using wandb)
    try:
        wandb.init(project="dead-end-detection", config={
            "learning_rate": lr,
            "epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "mixed_precision": use_amp
        })
    except Exception as e:
        print(f"Warning: Couldn't initialize wandb: {e}")
        use_wandb = False
    else:
        use_wandb = True
    
    model.to(device)
    
    # Optimizer with more conservative settings
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr,
        weight_decay=1e-4,
    )
    
    # Use a simpler scheduler that's more stable
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    # pos_weight = 3.15 #no of open paths/no of deadends 
    # Loss functions
    bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.84], device=device))
    # focal_loss = focal_loss_dead_end
    mse_loss = torch.nn.MSELoss()

    best_val_f1 = 0.0
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    # Disable logging for torcheval warnings
    import logging
    logging.getLogger("root").setLevel(logging.ERROR)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        path_f1 = 0.0
        dead_end_f1 = 0.0
        memory_cleanup()
        
        grad_norm_tracking = []
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Move data to device
                inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                try:
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        # Forward pass
                        outputs = model(
                            inputs['front_img'], inputs['right_img'], inputs['left_img'],
                            inputs['front_lidar'], inputs['right_lidar'], inputs['left_lidar']
                        )
                        # print(inputs['path_status'])
                        # print(inputs['is_dead_end'])
                        
                    # Debug path loss
                    try:
                        path_loss = bce_loss(outputs['path_logits'], inputs['path_status'])
                        # print(f"Path loss: {path_loss.item():.4f}")
                    except Exception as e:
                        print(f"Error in path loss: {e}")
                        path_loss = None

                    # Debug dead end mask
                    # try:
                    #     dead_end_mask = ~(inputs['path_status'].any(dim=1))
                    #     # print(f"Dead end mask sum: {dead_end_mask.sum().item()}/{dead_end_mask.size(0)}")
                    #     # print(f"Dead end logits range: [{outputs['dead_end_logits'][dead_end_mask].min().item():.4f}, {outputs['dead_end_logits'][dead_end_mask].max().item():.4f}]")
                    # except Exception as e:
                    #     print(f"Error in dead end mask: {e}")
                    #     dead_end_mask = None

                    # Debug focal loss
                    # try:
                    dead_end_loss = focal_loss_dead_end(
                            outputs['dead_end_logits'],
                            inputs['is_dead_end']
                        )
                    open_mask = inputs['path_status'].unsqueeze(2)
                    direction_loss = mse_loss(
                        outputs['direction_vectors'] * open_mask,
                        inputs['direction_vectors'] * open_mask
                    )
                    # if dead_end_mask is not None and dead_end_mask.any():
                    #     dead_end_loss = focal_loss_dead_end(
                    #         outputs['dead_end_logits'][dead_end_mask],
                    #         inputs['is_dead_end'][dead_end_mask]
                    #     )
                    #     # print(f"Focal loss: {dead_end_loss.item() if dead_end_loss is not None else 'None'}")
                    # else:
                    #     dead_end_loss = torch.tensor(0.0, device=device)
                    #     print("No dead ends in batch")
                    # except Exception as e:
                    #     print(f"Error in focal loss: {e}")
                    #     dead_end_loss = None
                    total_loss = path_loss + dead_end_loss + direction_loss
                    # Calculate total loss with checks
                    if all(x is not None for x in [path_loss, dead_end_loss]):
                        total_loss = path_loss + 0.1 * dead_end_loss
                        if not torch.isfinite(total_loss):
                            print("Non-finite loss breakdown:")
                            # print(f"- Path loss: {path_loss.item()}")
                            # print(f"- Dead end loss: {dead_end_loss.item()}")
                    else:
                        print("Skipping batch due to None losses")
                        continue
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue
                
                # Backward pass
                scaler.scale(total_loss).backward()
                
                # More conservative gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5, error_if_nonfinite=False)
                
                # Skip step if gradients are invalid
                if not torch.isfinite(grad_norm):
                    print(f"Warning: Non-finite gradients detected at batch {batch_idx}. Skipping optimizer step.")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                
                grad_norm_tracking.append(grad_norm.item())
                
                # Step optimizer
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                train_loss += total_loss.item()
                
                # Calculate metrics without warnings
                with torch.no_grad():
                    path_preds = torch.sigmoid(outputs['path_status']) > 0.5
                    dead_end_preds = torch.sigmoid(outputs['is_dead_end']) > 0.5
                    
                    batch_path_f1 = 0.0
                    for i in range(inputs['path_status'].size(1)):
                        path_pred = path_preds[:, i].long()
                        path_target = inputs['path_status'][:, i].long()
                        
                        # Only calculate F1 if both classes are present
                        if torch.unique(path_target).numel() >= 2:
                            try:
                                f1 = metrics_functional.binary_f1_score(path_pred, path_target)
                                batch_path_f1 += f1
                            except:
                                pass
                    
                    # Average across paths
                    path_f1 += (batch_path_f1 / inputs['path_status'].size(1)) if batch_path_f1 > 0 else 0
                    
                    # Only calculate dead end F1 if both classes are present
                    if torch.unique(inputs['is_dead_end']).numel() >= 2:
                        try:
                            dead_end_f1 += metrics_functional.binary_f1_score(
                                dead_end_preds.long().flatten(),
                                inputs['is_dead_end'].long().flatten()
                            )
                        except:
                            pass
                
                # Log batch metrics
                if use_wandb and batch_idx % 10 == 0:
                    wandb.log({
                        "batch_loss": total_loss.item(),
                        "grad_norm": grad_norm.item() if torch.isfinite(grad_norm) else 0.0,
                        "memory_allocated": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                    })
                
                # Periodic memory cleanup
                if batch_idx % 50 == 0 and batch_idx > 0:
                    memory_cleanup()
                    
                # Print progress
                if batch_idx % 20 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {total_loss.item():.4f}")

            except torch.cuda.OutOfMemoryError:
                print(f"CUDA OOM at batch {batch_idx}. Cleaning memory and skipping batch...")
                memory_cleanup()
                continue
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

        # Calculate epoch metrics
        train_loss /= max(1, len(train_loader))
        path_f1 /= max(1, len(train_loader))
        dead_end_f1 /= max(1, len(train_loader))

        # Validate model
        val_metrics = evaluate_model(model, val_loader, device)
        
        # LR scheduling based on validation loss
        scheduler.step(val_metrics['val_loss'])
        
        # Early stopping
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save best model
        if val_metrics['path_f1'] > best_val_f1:
            best_val_f1 = val_metrics['path_f1']
            torch.save(model.state_dict(), os.path.join(save_dir, 'model_best.pth'))
            print(f"Saved new best model with F1 score: {best_val_f1:.4f}")

        # Log epoch metrics
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "dead_end_loss":dead_end_loss,
                "val_loss": val_metrics['val_loss'],
                "path_f1": path_f1,
                "dead_end_f1": dead_end_f1,
                "val_path_f1": val_metrics['path_f1'],
                "val_dead_end_f1": val_metrics['dead_end_f1'],
                "gradient_mean": sum(grad_norm_tracking) / max(1, len(grad_norm_tracking)),
                "gradient_max": max(grad_norm_tracking) if grad_norm_tracking else 0
            })

        # Print statistics
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_metrics["val_loss"]:.4f}')
        print(f'Path F1: {path_f1:.4f} | Val Path F1: {val_metrics["path_f1"]:.4f}')
        print(f'DeadEnd F1: {dead_end_f1:.4f} | Val DeadEnd F1: {val_metrics["dead_end_f1"]:.4f}')
        # train_dead_ends = sum([sample['annotation'].get('is_dead_end', 0) for sample in train_dataset.samples])
        # val_dead_ends = sum([sample['annotation'].get('is_dead_end', 0) for sample in val_dataset.samples])

        # Save checkpoint
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': val_metrics,
                'best_val_f1': best_val_f1
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

        memory_cleanup()

    if use_wandb:
        wandb.finish()
    return model

# def evaluate_model(model, dataloader, device):
#     """Enhanced evaluation with F1 scores"""
#     model.eval()
#     val_loss = 0.0
    
#     # Initialize metrics
#     path_f1_sum = 0.0
#     dead_end_f1_sum = 0.0
#     batch_count = 0
    
#     bce_loss = torch.nn.BCEWithLogitsLoss()
#     mse_loss = torch.nn.MSELoss()

#     with torch.no_grad():
#         for batch in dataloader:
#             batch_count += 1
#             inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
#             outputs = model(
#                 inputs['front_img'], inputs['right_img'], inputs['left_img'],
#                 inputs['front_lidar'], inputs['right_lidar'], inputs['left_lidar']
#             )

#             # Calculate losses
#             path_loss = bce_loss(outputs['path_status'], inputs['path_status'])
#             dead_end_loss = bce_loss(outputs['is_dead_end'], inputs['is_dead_end'])
#             total_loss = path_loss + dead_end_loss
#             val_loss += total_loss.item()

#             # Get predictions
#             path_preds = torch.sigmoid(outputs['path_status']) > 0.5
#             dead_end_preds = torch.sigmoid(outputs['is_dead_end']) > 0.5
            
#             # Calculate F1 scores for each path position separately
#             batch_path_f1 = 0.0
#             for i in range(inputs['path_status'].size(1)):  # For each path position
#                 path_pred = path_preds[:, i].long()
#                 path_target = inputs['path_status'][:, i].long()
                
#                 # Use binary F1 score (only if we have samples from both classes)
#                 try:
#                     path_pos = metrics_functional.binary_f1_score(
#                         path_pred, 
#                         path_target
#                     )
#                     batch_path_f1 += path_pos
#                 except Exception as e:
#                     # If error (e.g., only one class present), just continue
#                     if torch.unique(batch_path_f1).numel() < 2:
#                         return 0.0
#             # Average F1 across paths
#             if inputs['path_status'].size(1) > 0:
#                 path_f1_sum += (batch_path_f1 / inputs['path_status'].size(1)) if batch_path_f1 > 0 else 0
            
#             # Calculate F1 score for dead end prediction
#             try:
#                 dead_end_f1 = metrics_functional.binary_f1_score(
#                     dead_end_preds.long().flatten(),
#                     inputs['is_dead_end'].long().flatten()
#                 )
#                 dead_end_f1_sum += dead_end_f1
#             except Exception as e:
#                 # If only one class is present in this batch, skip
#                 if torch.unique(dead_end_f1).numel() < 2:
#                     return 0.0

#     # Calculate final metrics
#     return {
#         'val_loss': val_loss / len(dataloader),
#         'path_f1': path_f1_sum / batch_count if batch_count > 0 else 0,
#         'dead_end_f1': dead_end_f1_sum / batch_count if batch_count > 0 else 0
#     }

def evaluate_model(model, dataloader, device):
    """Enhanced evaluation with F1 scores"""
    model.eval()
    val_loss = 0.0
    
    # Initialize metrics
    path_f1_sum = 0.0
    dead_end_f1_sum = 0.0
    batch_count = 0
    
    bce_loss = torch.nn.BCEWithLogitsLoss()
    mse_loss = torch.nn.MSELoss()

    with torch.no_grad():
        for batch in dataloader:
            try:
                batch_count += 1
                inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                outputs = model(
                    inputs['front_img'], inputs['right_img'], inputs['left_img'],
                    inputs['front_lidar'], inputs['right_lidar'], inputs['left_lidar']
                )

                # Calculate losses
                path_loss = bce_loss(outputs['path_logits'], inputs['path_status'])
                dead_end_loss = focal_loss_dead_end(outputs['dead_end_logits'], inputs['is_dead_end'])

                # print(f"path{inputs['path_status']}, dead_end{inputs['is_dead_end']}")
                
                # Skip direction loss in evaluation for simplicity
                total_loss = path_loss + dead_end_loss
                val_loss += total_loss.item()

                # Get predictions
                path_preds = torch.sigmoid(outputs['path_status']) > 0.5
                dead_end_preds = torch.sigmoid(outputs['is_dead_end']) > 0.5
                
                # Calculate F1 scores without warnings
                batch_path_f1 = 0.0
                for i in range(inputs['path_status'].size(1)):
                    path_pred = path_preds[:, i].long()
                    path_target = inputs['path_status'][:, i].long()
                    
                    # Only calculate F1 if both classes present
                    if torch.unique(path_target).numel() >= 2:
                        try:
                            f1 = metrics_functional.binary_f1_score(path_pred, path_target)
                            batch_path_f1 += f1
                        except:
                            pass
                
                if inputs['path_status'].size(1) > 0:
                    path_f1_sum += (batch_path_f1 / inputs['path_status'].size(1)) if batch_path_f1 > 0 else 0
                
                # Only calculate dead end F1 if both classes present
                if torch.unique(inputs['is_dead_end']).numel() >= 2:
                    try:
                        dead_end_f1 = metrics_functional.binary_f1_score(
                            dead_end_preds.long().flatten(),
                            inputs['is_dead_end'].long().flatten()
                        )
                        dead_end_f1_sum += dead_end_f1
                    except:
                        pass
            except Exception as e:
                print(f"Error in evaluation batch: {e}")
                continue

    # Calculate final metrics
    return {
        'val_loss': val_loss / max(1, len(dataloader)),
        'path_f1': path_f1_sum / max(1, batch_count),
        'dead_end_f1': dead_end_f1_sum / max(1, batch_count)
    }

def get_memory_efficient_data_loaders(data_root, batch_size=16, prefetch_factor=2):
    """
    Create memory-efficient data loaders
    
    Args:
        data_root: Path to dataset root directory
        batch_size: Batch size for training
        prefetch_factor: How many batches to prefetch per worker
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Smaller size uses less memory
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = DeadEndDataset(data_root, split='train', transform=train_transform)
    val_dataset = DeadEndDataset(data_root, split='val', transform=val_transform)
    # test_dataset = DeadEndDataset(data_root, split='test', transform=val_transform)
    
    # Persistent workers and pin memory for better performance
    # Lower num_workers to reduce memory usage
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,  # Reduce if needed
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        drop_last=True  # Drops incomplete last batch to avoid shape mismatches
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor
    )
    
    # test_loader = DataLoader(
    #     test_dataset, 
    #     batch_size=batch_size, 
    #     shuffle=False, 
    #     num_workers=2,
    #     pin_memory=True,
    #     persistent_workers=True,
    #     prefetch_factor=prefetch_factor
    # )
    
    return train_loader, val_loader


if __name__ == "__main__":
    inference = True
    if not inference:
        data_root = "/gammascratch/vigneshr/train_bags"
        save_dir = '/gammascratch/vigneshr/saved_models'
        #training the model #####################################################
        # Initialize memory optimization
        optimize_cuda_memory()
        
        # Determine optimal batch size based on available memory
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            # Conservative estimate - adjust based on your model's memory footprint
            if total_memory > 8:
                batch_size = 8
            # elif total_memory > 8:
            #     batch_size = 8
            elif total_memory > 4:
                batch_size = 4
            else:
                batch_size = 2
        else:
            batch_size = 4  # Default for CPU
            
        print(f"Using batch size: {batch_size}")
        
        # Choose device with memory considerations
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        try:
            # Import model - ensure this is available
            # from model.model_CA import DeadEndDetectionModel
            model = DeadEndDetectionModel()
            
            # Create data loaders with memory efficiency
            train_loader, val_loader = get_memory_efficient_data_loaders(
                data_root, 
                batch_size=batch_size
            )
            
            # Train with memory optimization
            model = train_model(
                model, 
                train_loader, 
                val_loader, 
                num_epochs=20, 
                device=device,
                save_dir=save_dir
            )
            
            # Final cleanup before evaluation
            memory_cleanup()
            
            # Evaluate model
            # metrics = evaluate_model(model, test_loader, device=device)
            # print("Final metrics:", metrics)
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA OOM error: {e}")
            memory_cleanup()
            
            # Try again with smaller batch size
            try:
                batch_size = max(1, batch_size // 2)
                print(f"Retrying with smaller batch size: {batch_size}")
                
                train_loader, val_loader = get_memory_efficient_data_loaders(
                    data_root, 
                    batch_size=batch_size
                )
                
                model = DeadEndDetectionModel()
                model = train_model(
                    model, 
                    train_loader, 
                    val_loader, 
                    num_epochs=20, 
                    device=device,
                    save_dir=save_dir
                )
                
                # metrics = evaluate_model(model, test_loader, device=device)
                
            except Exception as e:
                print(f"Error after batch size reduction: {e}")
                print("Falling back to CPU...")
                
                device = torch.device("cpu")
                batch_size = 2
                
                train_loader, val_loader = get_memory_efficient_data_loaders(
                    data_root, 
                    batch_size=batch_size
                )
                
                model = DeadEndDetectionModel()
                model = train_model(
                    model, 
                    train_loader, 
                    val_loader, 
                    num_epochs=5,  # Fewer epochs for CPU
                    device=device,
                    save_dir=save_dir
                )

    #testing the model #####################################################
    else:
        new_data_root = "/gammascratch/vigneshr/test_bags/test_seen_data/"
        model_path = "/gammascratch/vigneshr/saved_models/model_best.pth"
        output_dir = "/gammascratch/vigneshr/v2"
        # # Run inference visualization on new data
        visualize_test_results(
            model_path=model_path,
            data_root=new_data_root,
            batch_size=16,
            num_samples=50,
            output_dir=output_dir  # Adjust as neededa
        )
            