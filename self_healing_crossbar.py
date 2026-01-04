import torch
import numpy as np
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
import memtorch
from memtorch.bh.memristor.VTEAM import VTEAM
from memtorch.bh.crossbar import Crossbar
import random

class SelfHealingCrossbar:
    """
    Self-healing memristive crossbar that works with MemTorch's implementation.
    Uses a simpler, more compatible approach for redundancy management.
    """
    
    def __init__(self, memristor_model=VTEAM, shape=(128, 128), redundancy_factor=0.1, tile_shape=None):
        """
        Initialize a self-healing crossbar with built-in redundancy.
        
        Args:
            memristor_model: MemTorch memristor model to use
            shape: Crossbar dimensions (rows, cols)
            redundancy_factor: Percentage of redundant devices (0.1 = 10%)
        """
        # Store original dimensions
        self.original_shape = shape
        self.redundancy_factor = redundancy_factor
        
        self.is_remapped = torch.zeros(self.original_shape, dtype=torch.bool)
        self.device_mapping = torch.zeros(self.original_shape, dtype=torch.long)
        
        # Calculate effective shape with redundancy
        effective_rows = int(shape[0] * (1 + redundancy_factor))
        effective_cols = int(shape[1] * (1 + redundancy_factor))
        self.effective_shape = (effective_rows, effective_cols)
        
        # Initialize tracking of which devices are redundant vs primary
        self.is_redundant = torch.zeros(self.effective_shape, dtype=torch.bool)
        self.is_redundant[shape[0]:, :] = True  # Rows beyond original are redundant
        self.is_redundant[:, shape[1]:] = True  # Columns beyond original are redundant
        
        # Track remapped devices (logical device -> physical device mapping)
        self.remapping = {}  # (logical_row, logical_col) -> (physical_row, physical_col)
        
        # Keep track of available redundant devices
        self.available_redundant = []
        for i in range(self.effective_shape[0]):
            for j in range(self.effective_shape[1]):
                if self.is_redundant[i, j]:
                    self.available_redundant.append((i, j))
        
        # Initialize memristor parameters with safe defaults from the documentation
        memristor_params = {
            'r_on': 100,      # Low resistance state (Ohms)
            'r_off': 10000,   # High resistance state (Ohms)
            'time_series_resolution': 1e-10  # Required parameter
        }
        
        # Add model-specific parameters if VTEAM
        if memristor_model.__name__ == 'VTEAM':
            memristor_params.update({
                'd': 3e-9,        # Device length (m)
                'k_on': -10,      # k_on parameter
                'k_off': 5e-4,    # k_off parameter
                'alpha_on': 3,    # alpha_on parameter
                'alpha_off': 1,   # alpha_off parameter
                'v_on': -0.2,     # Positive write threshold voltage (V)
                'v_off': 0.02,    # Negative write threshold voltage (V)
                'x_on': 0,        # x_on parameter
                'x_off': 3e-9,    # x_off parameter
            })
        
        # Initialize the crossbar using MemTorch
        self.crossbar = Crossbar(
            memristor_model=memristor_model,
            memristor_model_params=memristor_params, 
            shape=self.effective_shape,
            tile_shape=tile_shape
        )
        
        # Track health metrics without assuming internal properties
        self.r_on = memristor_params['r_on']
        self.r_off = memristor_params['r_off']
        self.g_min = 1.0 / self.r_off
        self.g_max = 1.0 / self.r_on
        
        # Statistics tracking
        self.mitigation_count = 0
        self.remapping_count = 0
        self.operation_count = 0
        
        print(f"Created self-healing crossbar with shape {shape} and {len(self.available_redundant)} redundant devices")
    
    def _map_logical_to_physical(self, logical_row, logical_col):
        """Maps logical (original) coordinates to physical (effective) coordinates."""
        if logical_row >= self.original_shape[0] or logical_col >= self.original_shape[1]:
            raise ValueError(f"Logical coordinates ({logical_row},{logical_col}) out of bounds")
        
        logical_pos = (logical_row, logical_col)
        if logical_pos in self.remapping:
            # This device has been remapped
            return self.remapping[logical_pos]
        else:
            # Not remapped, use the same coordinates
            return logical_row, logical_col
    
    def write_conductance_matrix(self, conductance_matrix):
        """
        Write a conductance matrix to the crossbar, accounting for any remapped devices.
        
        Args:
            conductance_matrix: Conductance matrix to write (shape should match original_shape)
        """
        if conductance_matrix.shape != self.original_shape:
            # Handle the shape mismatch more gracefully
            if conductance_matrix.numel() == self.original_shape[0] * self.original_shape[1]:
                # Reshape if same number of elements
                conductance_matrix = conductance_matrix.reshape(self.original_shape)
            else:
                raise ValueError(f"Expected conductance matrix of shape {self.original_shape}, got {conductance_matrix.shape}")
        
        # Create a full conductance matrix for the crossbar, initialized with zeros
        effective_conductance = torch.zeros(self.effective_shape)
        
        # Copy the original values to their appropriate location with bounds checking
        for i in range(self.original_shape[0]):
            for j in range(self.original_shape[1]):
                try:
                    physical_row, physical_col = self._map_logical_to_physical(i, j)
                    effective_conductance[physical_row, physical_col] = conductance_matrix[i, j]
                except (IndexError, ValueError) as e:
                    print(f"Warning: Error mapping ({i},{j}): {e}")
        
        # Write to the crossbar with error handling
        try:
            self.crossbar.write_conductance_matrix(effective_conductance)
            self.operation_count += 1
        except Exception as e:
            print(f"Error writing to crossbar: {e}")
    
    def read_conductance_matrix(self):
        """
        Read the conductance matrix from the crossbar, accounting for remapped devices.
        
        Returns:
            Conductance matrix with original dimensions
        """
        # Get the full conductance matrix from the crossbar
        full_conductance = self.crossbar.conductance_matrix
        
        # Create a matrix to hold the logical values
        logical_conductance = torch.zeros(self.original_shape)
        
        # Copy values from their physical positions to logical positions
        for i in range(self.original_shape[0]):
            for j in range(self.original_shape[1]):
                logical_pos = (i, j)
                
                physical_row, physical_col = self._map_logical_to_physical(i, j)
                
                # Get the conductance value from the physical position
                logical_conductance[i, j] = full_conductance[physical_row, physical_col]
        
        return logical_conductance
    
    def simulate_matmul(self, input_tensor):
        """
        Perform matrix multiplication, taking remapping into account.
        
        Args:
            input_tensor: Input tensor for the multiplication
            
        Returns:
            Output tensor from the matrix multiplication
        """
        # Check dimensions
        if input_tensor.shape[1] != self.original_shape[0]:
            raise ValueError(f"Input shape mismatch: expected {self.original_shape[0]}, got {input_tensor.shape[1]}")
        
        # Create input tensor with extended dimensions to account for redundant rows
        extended_input = torch.zeros((input_tensor.shape[0], self.effective_shape[0]))
        
        # Copy the original inputs to their correct positions
        for i in range(self.original_shape[0]):
            # Find all logical positions that map to this input row
            extended_input[:, i] = input_tensor[:, i]
            
            # Check for any remappings from this row to redundant rows
            for j in range(self.original_shape[1]):
                logical_pos = (i, j)
                if logical_pos in self.remapping:
                    physical_row, _ = self._map_logical_to_physical(i, j)
                    if physical_row >= self.original_shape[0]:  # If remapped to redundant row
                        extended_input[:, physical_row] = input_tensor[:, i]
        
        # Perform matrix multiplication
        full_output = self.crossbar.simulate_matmul(input=extended_input)
        
        # Extract only the relevant columns, accounting for remapping
        output = torch.zeros((input_tensor.shape[0], self.original_shape[1]))
        
        # Copy values from physical to logical positions
        for j in range(self.original_shape[1]):
            column_mapped = False
            
            # Check if any logical position has been remapped to a redundant column
            for i in range(self.original_shape[0]):
                logical_pos = (i, j)
                if logical_pos in self.remapping:
                    _, physical_col = self._map_logical_to_physical(i, j)
                    if physical_col >= self.original_shape[1]:  # If remapped to redundant column
                        output[:, j] = full_output[:, physical_col]
                        column_mapped = True
                        break
            
            # If not remapped, use the original column
            if not column_mapped:
                output[:, j] = full_output[:, j]
        
        self.operation_count += 1
        return output
    
    def remap_device(self, logical_row, logical_col):
        """
        Remap a device to an available redundant device.
        
        Args:
            logical_row: Row of the device to remap
            logical_col: Column of the device to remap
            
        Returns:
            True if remapping successful, False otherwise
        """
        logical_pos = (logical_row, logical_col)
        
        # Check if already remapped
        if logical_pos in self.remapping:
            return False
        
        # Check if we have redundant devices available
        if not self.available_redundant:
            return False
        
        # Get current conductance
        full_conductance = self.crossbar.conductance_matrix
        current_conductance = full_conductance[logical_row, logical_col]
        
        # Get an available redundant device
        physical_row, physical_col = self.available_redundant.pop(0)
        
        # Update mapping
        self.remapping[logical_pos] = (physical_row, physical_col)
        
        # Copy conductance to the redundant device
        self.crossbar.conductance_matrix[physical_row, physical_col] = current_conductance
        
        self.remapping_count += 1
        return True
    
    def inject_faults(self, lrs_proportion=0.01, hrs_proportion=0.01):
        """
        Inject stuck-at faults into the crossbar for testing.
        
        Args:
            lrs_proportion: Proportion of devices to make stuck at LRS
            hrs_proportion: Proportion of devices to make stuck at HRS
            
        Returns:
            Number of faults injected
        """
        # Get original crossbar size
        rows, cols = self.original_shape
        total_devices = rows * cols
        
        # Calculate number of devices to make stuck
        num_lrs_faults = int(lrs_proportion * total_devices)
        num_hrs_faults = int(hrs_proportion * total_devices)
        
        # Create random indices for stuck devices
        device_indices = torch.randperm(total_devices)
        lrs_indices = device_indices[:num_lrs_faults]
        hrs_indices = device_indices[num_lrs_faults:num_lrs_faults+num_hrs_faults]
        
        # Apply faults
        faults_injected = 0
        
        for idx in lrs_indices:
            row = idx // cols
            col = idx % cols
            
            # Skip already remapped devices
            if (row, col) in self.remapping:
                continue
                
            # Make device stuck at LRS
            self.crossbar.conductance_matrix[row, col] = self.g_max
            faults_injected += 1
        
        for idx in hrs_indices:
            row = idx // cols
            col = idx % cols
            
            # Skip already remapped devices
            if (row, col) in self.remapping:
                continue
                
            # Make device stuck at HRS
            self.crossbar.conductance_matrix[row, col] = self.g_min
            faults_injected += 1
        
        print(f"Injected {faults_injected} faults ({num_lrs_faults} LRS, {num_hrs_faults} HRS)")
        return faults_injected
    
    def detect_faults(self, threshold=0.1):
        """
        Detect potential faults in the crossbar based on extreme conductance values.
        
        Args:
            threshold: How close to g_min or g_max to consider a fault
            
        Returns:
            List of detected faults as (row, col) tuples
        """
        # Get current conductance matrix
        conductance = self.crossbar.conductance_matrix
        
        # Define thresholds for detecting stuck devices
        g_min_threshold = self.g_min + threshold * (self.g_max - self.g_min)
        g_max_threshold = self.g_max - threshold * (self.g_max - self.g_min)
        
        # Find potential faults
        faults = []
        
        for i in range(self.original_shape[0]):
            for j in range(self.original_shape[1]):
                # Skip already remapped devices
                if (i, j) in self.remapping:
                    continue
                
                g_value = conductance[i, j]
                
                # Check if device appears stuck
                if g_value <= g_min_threshold or g_value >= g_max_threshold:
                    faults.append((i, j))
        
        return faults
    
    def apply_self_healing(self):
        """
        Apply self-healing by detecting and remapping faulty devices.
        
        Returns:
            Number of devices healed
        """
        # Skip if no redundant devices available
        if not self.available_redundant:
            return 0
        
        # Detect potential faults
        faults = self.detect_faults()
        if not faults:
            return 0
        
        # Apply remapping to as many faults as possible
        healed_count = 0
        
        for row, col in faults:
            if healed_count >= len(self.available_redundant):
                break
                
            if self.remap_device(row, col):
                healed_count += 1
        
        self.mitigation_count += healed_count
        print(f"Healed {healed_count} devices. Total remapped: {self.remapping_count}")
        
        return healed_count