import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import ast
import wfdb
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
# import seaborn as sns
import time
from scipy import signal
from scipy.signal import stft, istft
from fastdtw import fastdtw
from imblearn.over_sampling import SMOTE
import random
import copy
from collections import defaultdict
import json 

# Set up device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the dataset class with lazy loading
class PTBXLDataSet(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, indices=None, task='binary'):
        """
        Args:
            csv_file: Path to the PTB-XL database CSV
            root_dir: Directory containing the ECG files
            transform: Optional transform to apply to the data
            indices: Optional indices to filter the dataset
            task: Task to train on. Options:
                - 'binary': Normal vs abnormal
                - 'MI': Myocardial Infarction detection
                - 'HYP': Hypertrophy detection
                - 'STTC': ST/T Change detection
                - 'CD': Conduction Disturbance detection
                - 'NORM': Normal ECG detection
        """
        # Load and process the main database file
        self.annotations = pd.read_csv(csv_file)
        
        # Apply indices filter if provided (for train/test split)
        if indices is not None:
            self.annotations = self.annotations.iloc[indices]
        
        # Convert string representations of SCP codes to dictionaries
        self.annotations.scp_codes = self.annotations.scp_codes.apply(lambda x: ast.literal_eval(x))
        
        self.root_dir = root_dir
        self.transform = transform
        self.task = task  # 'binary', 'MI', 'HYP', 'STTC', 'CD', or 'NORM'
        
        # Determine which file path to use based on sampling rate
        if '500' in root_dir:
            self.file_column = 'filename_hr'
        else:
            self.file_column = 'filename_lr'
            
        # Load SCP statements to map SCP codes to diagnostic classes
        self.scp_df = self.load_scp_statements()
        
        # Print distribution of the selected task
        self._print_task_distribution()
    
    def load_scp_statements(self):
        """Load SCP statements to map SCP codes to diagnostic classes"""
        scp_file = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/scp_statements.csv'
        scp_df = pd.read_csv(scp_file, index_col=0)
        return scp_df
    
    def _print_task_distribution(self):
        """Print distribution of positive vs negative samples for the selected task"""
        count_pos = 0
        count_neg = 0
        
        for i in range(len(self.annotations)):
            label = self._get_label_for_idx(i)
            if label == 1:
                count_pos += 1
            else:
                count_neg += 1
        
        print(f"Task: {self.task}")
        print(f"Positive samples: {count_pos}")
        print(f"Negative samples: {count_neg}")
        print(f"Positive ratio: {count_pos / (count_pos + count_neg):.2%}")
    
    def has_diagnostic_class(self, scp_codes, target_class):
        """Check if any of the SCP codes belong to the target diagnostic class"""
        for code in scp_codes.keys():
            if code in self.scp_df.index:
                if self.scp_df.loc[code].diagnostic_class == target_class:
                    return True
        return False
    
    def _get_label_for_idx(self, idx):
        """Get label for a specific sample based on the selected task"""
        scp_codes = self.annotations.iloc[idx].scp_codes
        
        if self.task == 'binary':
            # Binary classification (normal vs. abnormal)
            return 1 if 'NORM' in scp_codes else 0
        
        elif self.task == 'MI':
            # Myocardial Infarction detection
            return 1 if self.has_diagnostic_class(scp_codes, 'MI') else 0
        
        elif self.task == 'HYP':
            # Hypertrophy detection
            return 1 if self.has_diagnostic_class(scp_codes, 'HYP') else 0
        
        elif self.task == 'STTC':
            # ST/T Change detection
            return 1 if self.has_diagnostic_class(scp_codes, 'STTC') else 0
        
        elif self.task == 'CD':
            # Conduction Disturbance detection
            return 1 if self.has_diagnostic_class(scp_codes, 'CD') else 0
        
        elif self.task == 'NORM':
            # Normal ECG detection
            return 1 if 'NORM' in scp_codes else 0
        
        else:
            raise ValueError(f"Unknown task: {self.task}")
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get file path for the ECG record without file extension
        row = self.annotations.iloc[idx]
        file_path = row[self.file_column]
        ecg_path = os.path.join(self.root_dir, file_path)
        
        # Load data using wfdb - only when the item is requested
        ecg_data, _ = wfdb.rdsamp(ecg_path)
        
        # ECG data shape is [sequence_length, 12] for 12-lead ECG
        # Convert to torch tensor with shape [12, sequence_length] for 1D convolution
        ecg_data = torch.from_numpy(ecg_data).float().transpose(0, 1)
        
        # Get label based on the selected task
        label = torch.tensor(self._get_label_for_idx(idx))
        
        if self.transform:
            ecg_data = self.transform(ecg_data)

        # Note: We don't move to device here because DataLoader does it for us
        return ecg_data, label

# Define custom ResNet for ECG

def conv_block(in_channels, out_channels, kernel_size=15, stride=1, padding=7):
    """Custom convolution block with batch normalization and activation"""
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                  stride=stride, padding=padding, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True)
    )

class ECGBlock(nn.Module):
    """Residual block for ECG classification"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ECGBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=15, 
                               stride=stride, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Second convolutional layer
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=15,
                               stride=1, padding=7, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        # First convolutional layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second convolutional layer
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply downsample if needed
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add residual connection and apply ReLU
        out += identity
        out = self.relu(out)
        
        return out

class ECG1DResNet(nn.Module):
    """1D ResNet for ECG classification"""
    def __init__(self, block, layers, num_classes=1):
        super(ECG1DResNet, self).__init__()
        
        # Initial parameters
        self.inplanes = 32
        
        # First convolutional layer
        self.conv1 = nn.Conv1d(12, self.inplanes, kernel_size=15, 
                               stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        
        # Global average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        
        # Create downsample layer if stride != 1 or if input/output channels differ
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )
        
        layers = []
        # First block may have downsampling
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        
        # Rest of the blocks maintain dimensions
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.fc(x)
        
        return x

def ecg_resnet18(num_classes=1):
    """Create a ResNet-18 model for ECG classification"""
    return ECG1DResNet(ECGBlock, [2, 2, 2, 2], num_classes)

# Functions to calculate metrics for binary classification
def calculate_binary_metrics(y_true, y_pred_prob, threshold=0.5):
    """Calculate metrics for binary classification with sigmoid outputs"""
    # Convert pytorch tensors to numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred_prob, torch.Tensor):
        y_pred_prob = y_pred_prob.cpu().numpy()
    
    # Apply threshold to get binary predictions
    y_pred_binary = (y_pred_prob >= threshold).astype(int)
    
    # Calculate standard metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_binary, average='binary')
    
    accuracy = accuracy_score(y_true, y_pred_binary)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Calculate AUROC
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auroc = auc(fpr, tpr)
    
    # Calculate AUPRC
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_prob)
    auprc = average_precision_score(y_true, y_pred_prob)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'auroc': auroc,
        'auprc': auprc,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'fpr': fpr,
        'tpr': tpr,
        'precision_curve': precision_curve,
        'recall_curve': recall_curve
    }

def plot_confusion_matrix(y_true, y_pred_prob, threshold=0.5, task_name='Task'):
    """Plot confusion matrix for binary classification"""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred_prob, torch.Tensor):
        y_pred_prob = y_pred_prob.cpu().numpy()
    
    # Apply threshold
    y_pred_binary = (y_pred_prob >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred_binary)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {task_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks([0.5, 1.5], ['Negative', 'Positive'])
    plt.yticks([0.5, 1.5], ['Negative', 'Positive'])
    
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{task_name}.png')
    plt.close()

def plot_roc_curve(fpr, tpr, auroc, task_name='Task'):
    """Plot ROC curve with AUROC value"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUROC = {auroc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {task_name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'roc_curve_{task_name}.png')
    plt.close()

def plot_precision_recall_curve(precision, recall, auprc, task_name='Task'):
    """Plot Precision-Recall curve with AUPRC value"""
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUPRC = {auprc:.3f})')
    plt.axhline(y=sum(precision)/len(precision), color='navy', linestyle='--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {task_name}')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'pr_curve_{task_name}.png')
    plt.close()

# Updated evaluation function with AUROC metrics
def evaluate_binary(model, test_loader, task_name='Task'):
    model.eval()
    all_labels = []
    all_outputs = []
    
    eval_start_time = time.time()
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move inputs to device
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)  # Apply sigmoid for binary classification
            
            # Move outputs back to CPU for metric calculation
            outputs = outputs.cpu()
            
            all_labels.append(labels)
            all_outputs.append(outputs)
    
    # Concatenate all batches
    all_labels = torch.cat(all_labels, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)
    
    # Reshape for metrics calculation
    all_labels = all_labels.view(-1).cpu().numpy()
    all_outputs = all_outputs.view(-1).cpu().numpy()
    
    # Calculate metrics
    metrics = calculate_binary_metrics(all_labels, all_outputs)
    
    eval_time = time.time() - eval_start_time
    
    # Print metrics
    print(f"===== {task_name} Classification Metrics =====")
    print(f"Evaluation time: {eval_time:.2f} seconds")
    print(f"AUROC: {metrics['auroc']:.4f}")  # Highlight AUROC as primary metric
    print(f"AUPRC: {metrics['auprc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall (Sensitivity): {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"True Positives: {metrics['tp']}")
    print(f"False Positives: {metrics['fp']}")
    print(f"True Negatives: {metrics['tn']}")
    print(f"False Negatives: {metrics['fn']}")
    
    # Plot ROC curve
    # plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['auroc'], task_name)
    
    # # Plot Precision-Recall curve
    # plot_precision_recall_curve(metrics['precision_curve'], metrics['recall_curve'], 
    #                           metrics['auprc'], task_name)
    
    # # Plot confusion matrix
    # plot_confusion_matrix(all_labels, all_outputs, task_name=task_name)
    
    return metrics

# Define augmentation techniques
class ECGAugmentations:
    """Class containing different ECG augmentation methods"""
    
    @staticmethod
    def time_masking(ecg_signal, mask_ratio=0.1):
        """
        Apply time masking to ECG signal
        
        Args:
            ecg_signal: ECG signal of shape [channels, time_steps]
            mask_ratio: Fraction of signal to mask
        
        Returns:
            Augmented ECG signal
        """
        # Make a defensive copy
        masked_signal = ecg_signal.clone()
        
        try:
            # Get signal dimensions
            channels, signal_length = masked_signal.shape
            
            # Calculate mask size
            mask_size = max(1, int(signal_length * mask_ratio))
            
            # Generate random start point for the mask
            if signal_length > mask_size:
                mask_start = random.randint(0, signal_length - mask_size - 1)
                
                # Create binary mask: 1 for keep, 0 for mask
                mask = torch.ones_like(masked_signal)
                mask[:, mask_start:mask_start + mask_size] = 0
                
                # Apply mask
                masked_signal = masked_signal * mask
        except Exception as e:
            print(f"Time masking error: {e} - Using original signal")
            # Just return the original signal on error
            pass
            
        return masked_signal
    
    @staticmethod
    def spec_augment(ecg_signal, time_mask_ratio=0.1, freq_mask_ratio=0.1):
        """
        Apply SpecAugment to ECG signal using STFT
        
        Args:
            ecg_signal: ECG signal of shape [channels, time_steps]
            time_mask_ratio: Fraction of time bins to mask
            freq_mask_ratio: Fraction of frequency bins to mask
        
        Returns:
            Augmented ECG signal
        """
        # Make a copy to avoid modifying the original
        signal_np = ecg_signal.clone().numpy()
        augmented = np.zeros_like(signal_np)
        
        # Process each channel separately
        for ch in range(signal_np.shape[0]):
            # Calculate STFT with parameters that ensure proper reconstruction
            nperseg = 64  # Window size
            noverlap = 32  # Overlap between windows
            
            try:
                # Ensure the signal is properly padded to make STFT and ISTFT compatible
                n_samples = signal_np.shape[1]
                padding = (nperseg - (n_samples % nperseg)) % nperseg
                if padding > 0:
                    padded_signal = np.pad(signal_np[ch], (0, padding))
                else:
                    padded_signal = signal_np[ch]
                
                # Calculate STFT
                f, t, Zxx = stft(padded_signal, nperseg=nperseg, noverlap=noverlap)
                
                # Calculate mask sizes
                time_mask_size = max(1, int(len(t) * time_mask_ratio))
                freq_mask_size = max(1, int(len(f) * freq_mask_ratio))
                
                # Create a copy of the STFT
                Z_masked = Zxx.copy()
                
                # Apply time masking
                if time_mask_size > 0 and len(t) > time_mask_size:
                    time_start = random.randint(0, len(t) - time_mask_size - 1)
                    Z_masked[:, time_start:time_start + time_mask_size] = 0 + 0j
                
                # Apply frequency masking
                if freq_mask_size > 0 and len(f) > freq_mask_size:
                    freq_start = random.randint(0, len(f) - freq_mask_size - 1)
                    Z_masked[freq_start:freq_start + freq_mask_size, :] = 0 + 0j
                
                # Calculate inverse STFT
                _, reconstructed = istft(Z_masked, nperseg=nperseg, noverlap=noverlap)
                
                # Truncate to original length
                augmented[ch] = reconstructed[:signal_np.shape[1]]
                
            except Exception as e:
                # Fallback to original signal on error
                print(f"SpecAugment error: {e} - Using original signal for channel {ch}")
                augmented[ch] = signal_np[ch]
        
        return torch.from_numpy(augmented).float()
    
    @staticmethod
    def guided_warping(ecg_signal, reference_signals, dtw_radius=10):
        """
        Apply Discriminative Guided Warping to ECG signal
        
        Args:
            ecg_signal: ECG signal to augment
            reference_signals: Dictionary of reference signals by class
            dtw_radius: DTW radius parameter
        
        Returns:
            Augmented ECG signal
        """
        # If no reference signals, return original
        if reference_signals is None or len(reference_signals) == 0:
            return ecg_signal.clone()
        
        # Convert to numpy for DTW
        signal_np = ecg_signal.clone().numpy()
        
        # Pick a random reference class different from the input
        available_classes = list(reference_signals.keys())
        if len(available_classes) <= 1:
            return torch.from_numpy(signal_np).float()
        
        target_class = random.choice(available_classes)
        
        # Pick a random reference signal from the target class
        if len(reference_signals[target_class]) == 0:
            return torch.from_numpy(signal_np).float()
        
        reference = random.choice(reference_signals[target_class])
        ref_np = reference.clone().numpy()
        
        # Check if shapes match
        if signal_np.shape != ref_np.shape:
            # Simple interpolation to match shapes if needed
            if ref_np.shape[1] != signal_np.shape[1]:
                temp_ref = np.zeros_like(signal_np)
                for ch in range(ref_np.shape[0]):
                    temp_ref[ch] = np.interp(
                        np.linspace(0, 1, signal_np.shape[1]),
                        np.linspace(0, 1, ref_np.shape[1]),
                        ref_np[ch]
                    )
                ref_np = temp_ref
        
        augmented = np.zeros_like(signal_np)
        
        # Apply DTW to each channel
        for ch in range(signal_np.shape[0]):
            try:
                # Calculate DTW path
                _, path = fastdtw(signal_np[ch], ref_np[ch], radius=dtw_radius)
                path = np.array(path)
                
                # Create mapping from original signal to warped signal
                source_idx = path[:, 0]
                target_idx = path[:, 1]
                
                # Remove duplicates if necessary
                if len(np.unique(source_idx)) < len(source_idx):
                    _, unique_idx = np.unique(source_idx, return_index=True)
                    source_idx = source_idx[unique_idx]
                    target_idx = target_idx[unique_idx]
                
                # Apply warping with proper bounds checking
                if len(source_idx) > 0 and len(target_idx) > 0:
                    # Ensure target_idx is within bounds
                    valid_target_idx = target_idx < ref_np.shape[1]
                    if not np.all(valid_target_idx):
                        source_idx = source_idx[valid_target_idx]
                        target_idx = target_idx[valid_target_idx]
                    
                    # Apply interpolation for warping
                    if len(source_idx) > 1:  # Need at least 2 points for interpolation
                        warped_signal = np.interp(
                            np.arange(signal_np.shape[1]), 
                            source_idx, 
                            ref_np[ch, target_idx]
                        )
                        augmented[ch] = warped_signal
                    else:
                        augmented[ch] = signal_np[ch]
                else:
                    augmented[ch] = signal_np[ch]
            except Exception as e:
                print(f"Guided warping error: {e} - Using original signal for channel {ch}")
                augmented[ch] = signal_np[ch]
        
        return torch.from_numpy(augmented).float()
    
    @staticmethod
    def apply_smote(X, y, sampling_strategy=1.0):
        """
        Apply SMOTE to oversample minority class
        
        Args:
            X: Features array of shape [n_samples, n_features]
            y: Labels array of shape [n_samples]
            sampling_strategy: Ratio of minority to majority class
        
        Returns:
            Augmented X and y
        """
        try:
            # Validate inputs
            if X.shape[0] != len(y):
                print(f"Error in SMOTE: X shape {X.shape} does not match y length {len(y)}")
                return X, y
                
            # Check class distribution
            class_counts = np.bincount(y)
            if len(class_counts) < 2:
                print("Error in SMOTE: Need at least 2 classes")
                return X, y
                
            # Check if we have enough samples for SMOTE
            min_samples = np.min(class_counts)
            if min_samples < 2:
                print(f"Error in SMOTE: Need at least 2 samples per class, got {min_samples}")
                return X, y
                
            # Determine k_neighbors (must be less than min_samples)
            k_neighbors = min(5, min_samples - 1)
            if k_neighbors < 1:
                print(f"Error in SMOTE: Not enough samples for SMOTE (k_neighbors={k_neighbors})")
                return X, y
            
            # Apply SMOTE with appropriate parameters
            smote = SMOTE(
                sampling_strategy=sampling_strategy, 
                random_state=42, 
                k_neighbors=k_neighbors
            )
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            return X_resampled, y_resampled
            
        except Exception as e:
            print(f"SMOTE error: {e} - Using original data")
            return X, y


# Augmentation dataset wrapper
class AugmentedDataset(Dataset):
    """Dataset wrapper that applies augmentations on the fly"""
    
    def __init__(self, dataset, augmentation=None, augment_prob=0.5, reference_signals=None):
        """
        Args:
            dataset: Original dataset
            augmentation: Augmentation method to apply ('time_masking', 'spec_augment', 'guided_warping')
            augment_prob: Probability of applying augmentation
            reference_signals: Reference signals for guided warping (dict of tensors by class)
        """
        self.dataset = dataset
        self.augmentation = augmentation
        self.augment_prob = augment_prob
        self.reference_signals = reference_signals
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        ecg_signal, label = self.dataset[idx]
        
        # Apply augmentation with probability augment_prob
        if self.augmentation and random.random() < self.augment_prob:
            if self.augmentation == 'time_masking':
                ecg_signal = ECGAugmentations.time_masking(ecg_signal)
            elif self.augmentation == 'spec_augment':
                ecg_signal = ECGAugmentations.spec_augment(ecg_signal)
            elif self.augmentation == 'guided_warping' and self.reference_signals is not None:
                ecg_signal = ECGAugmentations.guided_warping(ecg_signal, self.reference_signals)
        
        return ecg_signal, label


# Function to run experiment with different augmentations
def run_augmentation_experiment(task, augmentation_method=None, num_epochs=5, batch_size=8):
    """
    Run experiment with specified augmentation method
    
    Args:
        task: Task to train on ('MI', 'HYP', 'STTC', 'CD')
        augmentation_method: Augmentation method to use (None, 'time_masking', 'spec_augment', 'guided_warping', 'smote')
        num_epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        Dictionary with test metrics
    """
    print(f"\n{'='*50}")
    print(f"Running experiment for task: {task}, augmentation: {augmentation_method}")
    print(f"{'='*50}")
    
    # Load dataset
    print("Loading dataset...")
    csv_file = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv'
    if '100' in 'records100':
        root_dir = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
    else:
        root_dir = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    # Create subset for faster testing
    # subset_size = min(2000, len(df))
    # indices = np.random.choice(len(df), subset_size, replace=False)
    # df_subset = df.iloc[indices]
    df_subset = df
    
    # Create temporary dataset to get labels
    temp_dataset = PTBXLDataSet(csv_file=csv_file, root_dir=root_dir, 
                             indices=df_subset.index.tolist(), task=task)
    
    # Create labels for stratification
    labels = []
    for i in range(len(temp_dataset)):
        _, label = temp_dataset[i]
        labels.append(label.item())
    
    labels = np.array(labels)
    
    # Split data into train and test
    train_indices, test_indices = train_test_split(
        np.arange(len(df_subset)), 
        test_size=0.2, 
        random_state=42, 
        stratify=labels
    )
    
    # Create datasets
    train_dataset = PTBXLDataSet(csv_file=csv_file, root_dir=root_dir, 
                              indices=df_subset.index[train_indices].tolist(), task=task)
    test_dataset = PTBXLDataSet(csv_file=csv_file, root_dir=root_dir, 
                             indices=df_subset.index[test_indices].tolist(), task=task)
    
    print(f"Train set size: {len(train_dataset)}, Test set size: {len(test_dataset)}")
    
    # Apply SMOTE if selected
    if augmentation_method == 'smote':
        print("Applying SMOTE augmentation...")
        
        # Extract all training data
        X_train = []
        y_train = []
        
        for i in range(len(train_dataset)):
            ecg, label = train_dataset[i]
            # Flatten ECG to 1D for SMOTE
            X_train.append(ecg.reshape(-1).numpy())
            y_train.append(label.item())
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Apply SMOTE
        X_train_resampled, y_train_resampled = ECGAugmentations.apply_smote(X_train, y_train)
        
        print(f"Original class distribution: {np.bincount(y_train)}")
        print(f"Resampled class distribution: {np.bincount(y_train_resampled)}")
        
        # Create new dataset from resampled data
        class SMOTEDataset(Dataset):
            def __init__(self, X, y, original_shape):
                self.X = X
                self.y = y
                self.original_shape = original_shape
                
            def __len__(self):
                return len(self.y)
                
            def __getitem__(self, idx):
                # Reshape back to original ECG shape
                ecg = torch.from_numpy(self.X[idx].reshape(self.original_shape)).float()
                label = torch.tensor(self.y[idx])
                return ecg, label
        
        # Get original ECG shape from the first item
        original_shape = train_dataset[0][0].shape
        
        # Replace train dataset with SMOTE augmented dataset
        train_dataset = SMOTEDataset(X_train_resampled, y_train_resampled, original_shape)
    
    # For guided warping, collect reference signals by class
    reference_signals = None
    if augmentation_method == 'guided_warping':
        print("Collecting reference signals for guided warping...")
        reference_signals = defaultdict(list)
        
        # Collect a few examples from each class
        max_refs_per_class = 10
        class_counts = defaultdict(int)
        
        for i in range(len(train_dataset)):
            ecg, label = train_dataset[i]
            class_label = label.item()
            
            if class_counts[class_label] < max_refs_per_class:
                reference_signals[class_label].append(ecg)
                class_counts[class_label] += 1
    
    # Apply on-the-fly augmentations if selected
    if augmentation_method in ['time_masking', 'spec_augment', 'guided_warping']:
        train_dataset = AugmentedDataset(
            train_dataset, 
            augmentation=augmentation_method,
            augment_prob=0.5,
            reference_signals=reference_signals
        )
    
    # Set DataLoader parameters
    num_workers = 4 if torch.cuda.is_available() else 0
    pin_memory = torch.cuda.is_available()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Initialize model
    model = ecg_resnet18(num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    print(f"Training model...")
    train(model, train_loader, criterion, optimizer, num_epochs=num_epochs)
    
    # Evaluate the model
    metrics = evaluate_binary(model, test_loader, task_name=f"{task}_{augmentation_method}")
    
    return metrics


# Function to run experiments for multiple tasks and augmentations
def run_all_experiments(tasks, augmentations, num_epochs=5):
    """
    Run experiments for multiple tasks and augmentations
    
    Args:
        tasks: List of tasks to evaluate
        augmentations: List of augmentation methods
        num_epochs: Number of training epochs
    
    Returns:
        Results dictionary
    """
    results = {}
    
    for task in tasks:
        results[task] = {}
        
        for aug in augmentations:
            print(f"\nRunning experiment: Task={task}, Augmentation={aug}")
            metrics = run_augmentation_experiment(task, aug, num_epochs=num_epochs)
            results[task][aug] = metrics
    
    return results


# Function to print and compare results
def print_results_table(results):
    """
    Print results in table format
    
    Args:
        results: Results dictionary
    """
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    metrics_to_show = ['auroc', 'auprc', 'accuracy', 'f1']
    
    for task in results:
        print(f"\nTask: {task}")
        print("-"*80)
        
        # Table header
        header = "Augmentation"
        for metric in metrics_to_show:
            header += f" | {metric}"
        print(header)
        print("-"*80)
        
        for aug, metrics in results[task].items():
            row = f"{aug or 'None'}"
            for metric in metrics_to_show:
                row += f" | {metrics[metric]:.4f}"
            print(row)


# Add a function to check and print GPU info if available
def print_gpu_info():
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"CUDA Version: {torch.version.cuda}")
        # Print GPU memory info
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    else:
        print("CUDA is not available. Using CPU.")

# Print GPU info at the beginning
print_gpu_info()

# Update training loop with CUDA support and timing
def train(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    
    # For timing
    total_start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Move inputs and labels to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Convert labels to float and reshape for BCEWithLogitsLoss
            labels = labels.float().view(-1, 1)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Time: {epoch_time:.2f} seconds')
    
    total_time = time.time() - total_start_time
    print(f'Total training time: {total_time:.2f} seconds')

# Main execution
if __name__ == "__main__":
    # Define tasks and augmentations to evaluate
    tasks_to_evaluate = ['MI', 'HYP', 'STTC', 'CD']
    augmentations_to_evaluate = [None, 'time_masking', 'spec_augment',  'smote']
    
    # Number of epochs for each experiment
    num_epochs = 20
    
    # Run all experiments
    all_results = run_all_experiments(tasks_to_evaluate, augmentations_to_evaluate, num_epochs)
    
    # Print results
    print_results_table(all_results)
    
    # Save results to file
    # np.save("augmentation_results.npy", all_results)
    with open("augmentation_results.json", "w") as f:
        json.dump(all_results, f)
    print("\nResults saved to augmentation_results.json")

