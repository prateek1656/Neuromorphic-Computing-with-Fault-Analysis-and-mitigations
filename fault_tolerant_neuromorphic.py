import torch
import numpy as np
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
from health_monitor import HealthMonitor
from memtorch.bh.memristor.VTEAM import VTEAM
from memtorch.bh.crossbar import Scheme
from memtorch.mn.Module import patch_model
from memtorch.map.Parameter import naive_map
from memtorch.map.Input import naive_scale
import random

from models import SimpleCNN
from self_healing_crossbar import SelfHealingCrossbar

class FaultTolerantNeuromorphic:
    """
    Enhanced fault-tolerant neuromorphic computing system with multiple mitigation strategies:
    1. Soft mitigation (adjusting conductance)
    2. Redundancy-based remapping (using spare devices)
    3. Layer reset (for catastrophic failures)
    """
    
    def __init__(self, base_model=None, memristor_model=VTEAM, redundancy_factor=0.15):
        """
        Initialize the enhanced fault-tolerant neuromorphic system.
        
        Args:
            base_model: Base PyTorch model (if None, creates a SimpleCNN)
            memristor_model: Memristor model to use
            redundancy_factor: Redundancy factor for self-healing crossbars
        """
        
        self.memristive_layers = {}
        self.crossbars = {}
        self.monitors = {}
        
        # Create base model if not provided
        if base_model is None:
            self.base_model = SimpleCNN()
        else:
            self.base_model = base_model
        
        # Define memristor parameters
        memristor_params = {
            'r_on': 100,      # Low resistance state (Ohms)
            'r_off': 10000,   # High resistance state (Ohms)
            'time_series_resolution': 1e-10  # Required parameter
        }
        
        # Add model-specific parameters if using VTEAM
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
        
        # Store for future use
        self.memristor_model = memristor_model
        self.memristor_params = memristor_params
        self.redundancy_factor = redundancy_factor
        
        # Convert to memristive model
        try:
            self.memristive_model = patch_model(
                model=self.base_model,
                memristor_model=memristor_model,
                memristor_model_params=memristor_params,
                module_parameters_to_patch=[torch.nn.Linear],  # Start with just linear layers
                mapping_routine=naive_map,
                transistor=True,
                scheme=Scheme.DoubleColumn,
                max_input_voltage=0.3,
                scaling_routine=naive_scale,
                ADC_resolution=8,
                ADC_overflow_rate=0.0,
                quant_method='linear'
            )
            print("Successfully patched model with memristors")
        except Exception as e:
            print(f"Error patching model: {e}")
            # Fall back to original model
            self.memristive_model = self.base_model
            print("Using original model as fallback")
        
        # Track memristive layers
        self.memristive_layers = {}  # name -> module mapping
        self.health_monitors = {}    # name -> health monitor mapping
        self.self_healing_crossbars = {}  # name -> self-healing crossbar mapping
        
        # Scan for memristive layers and create self-healing replacements
        for name, module in self.memristive_model.named_modules():
            if hasattr(module, 'crossbars'):
                print(f"Found memristive layer: {name}")
                self.memristive_layers[name] = module
                
                # Create health monitors and self-healing replacements for each crossbar
                for i, crossbar in enumerate(module.crossbars):
                    crossbar_name = f"{name}_crossbar_{i}"
                    
                    # Create an enhanced monitor for the original crossbar
                    self.health_monitors[crossbar_name] = HealthMonitor(crossbar)
                    
                    # Create a self-healing crossbar with the same shape
                    crossbar_shape = crossbar.conductance_matrix.shape
    
                    # FIX: Better calculation of adjusted shape with proper rounding and min size
                    original_rows = max(1, int(crossbar_shape[0] / (1 + redundancy_factor) + 0.5))  # Round properly
                    original_cols = max(1, int(crossbar_shape[1] / (1 + redundancy_factor) + 0.5))
                    adjusted_shape = (original_rows, original_cols)
                    
                    # FIX: Store the original crossbar dimensions for safety checks
                    healing_crossbar = SelfHealingCrossbar(
                        memristor_model=memristor_model,
                        shape=adjusted_shape,
                        redundancy_factor=redundancy_factor
                    )
                    self.self_healing_crossbars[crossbar_name] = healing_crossbar
                    healing_crossbar.original_crossbar_shape = crossbar_shape  # Store for reference
        
        # Create backup of original weights for critical situations
        self.original_weights = {}
        for name, module in self.base_model.named_modules():
            if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
                if hasattr(module, 'weight'):
                    self.original_weights[name] = module.weight.data.clone()
        
        # Statistics for all mitigation strategies
        self.total_operations = 0
        self.total_faults_detected = 0
        self.total_soft_mitigations = 0
        self.total_remappings = 0
        self.total_layer_resets = 0
        self.redundant_available = sum(len(sb.available_redundant) for sb in self.self_healing_crossbars.values())
        
        
        if hasattr(self.memristive_model, 'forward_legacy'):
            self.memristive_model.forward_legacy(True)
        
        print(f"Initialized enhanced fault-tolerant system with {len(self.memristive_layers)} memristive layers")
        print(f"Total redundant devices available: {self.redundant_available}")
    
    
    # high-level methods:
    def forward(self, x):
        """Forward pass with fault monitoring."""
        try:
            # Forward pass using the memristive model
            # Use a context manager to properly handle gradient requirements based on mode
            with torch.set_grad_enabled(self.memristive_model.training):
                output = self.memristive_model(x)
            
            # Update stats outside of gradient tracking
            with torch.no_grad():
                self.total_operations += 1
                
                # Periodically check health
                if self.total_operations % 10 == 0:
                    self.check_health()
            
            return output
        
        except RuntimeError as e:
            print("not using health check due to deep copy")
            with torch.set_grad_enabled(self.memristive_model.training):
                output = self.memristive_model(x)
            
            return output
    
    def check_health(
        self,
        health_threshold=40,
        force_remapping=False,
        enable_soft_mitigation=True,
        enable_remapping=True,
        enable_layer_reset=True,
        enable_predictive=True):
        """
        Enhanced health check system with both on-the-fly runtime monitoring and
        predictive degradation management for memristive layers.
        
        The function implements a tiered approach:
        1. Online/On-the-Fly Runtime Fault detection using direct weight verification
        2. Predictive Degradation Management
        3. Apply appropriate mitigation strategies based on fault severity
        
        Args:
            health_threshold: Threshold for determining mitigation strategy
            force_remapping: Whether to force remapping regardless of health
            enable_soft_mitigation: Whether to enable conductance adjustment
            enable_remapping: Whether to enable redundancy-based remapping
            enable_layer_reset: Whether to enable layer reset for catastrophic failures
            enable_predictive: Whether to enable predictive degradation management
            
        Returns:
            Number of issues found
        """
        total_issues = 0
        predictive_issues = 0
        runtime_issues = 0
        
        # Initialize device history tracking if not present
        if not hasattr(self, 'device_history'):
            self.device_history = {}
            self.mitigation_history = {}
        
        # Initialize reference weights if not present
        if not hasattr(self, 'reference_conductances'):
            self.initialize_reference_conductances()
            
        # Check if weights were updated and need refreshing
        if hasattr(self, 'weights_updated') and self.weights_updated:
            self._update_reference_conductances()
            self.weights_updated = False
        
        
        # Check if all mitigation strategies are disabled
        if not (enable_soft_mitigation or enable_remapping or enable_layer_reset):
            print("All mitigation strategies are disabled. Performing health monitoring only.")
            
            # Still update health metrics but don't apply any mitigations
            for name, monitor in self.health_monitors.items():
                try:
                    monitor.update_health_metrics()
                    critical_devices = monitor.get_critical_devices()
                    total_issues += len(critical_devices)
                    
                    if critical_devices:
                        layer_health = monitor.get_health_summary()
                        print(f"Found {len(critical_devices)} critical devices in {name} (health: {layer_health['avg_health']:.2f}%)")
                        print(f"No mitigation applied (all strategies disabled)")
                        
                except Exception as e:
                    print(f"Error checking health for {name}: {e}")
                    
            self.total_faults_detected += total_issues
            return total_issues
        
        # Process all layers
        for name, monitor in self.health_monitors.items():
            try:
                # Extract layer info
                name_parts = name.split('_crossbar_')
                layer_name = name_parts[0]
                crossbar_idx = int(name_parts[1])
                
                # Update basic health metrics
                monitor.update_health_metrics()
                
               
                
                # -------------------------------------------------------------------
                # PHASE 1: ON-THE-FLY RUNTIME FAULT DETECTION (using direct weight verification)
                # -------------------------------------------------------------------
                runtime_critical_devices = self._detect_runtime_faults_by_conductance(name, monitor)
                runtime_issues += len(runtime_critical_devices)
                
                # -------------------------------------------------------------------
                # PHASE 2: PREDICTIVE DEGRADATION MANAGEMENT
                # -------------------------------------------------------------------
                predictive_critical_devices = []
                if enable_predictive:
                    predictive_critical_devices = self._detect_predictive_degradation(name, monitor)
                    predictive_issues += len(predictive_critical_devices)
                
                # Combine both types of critical devices
                critical_devices = runtime_critical_devices.copy()
                for device in predictive_critical_devices:
                    if device not in critical_devices:
                        critical_devices.append(device)
                
                # -------------------------------------------------------------------
                # PHASE 3: APPLY MITIGATION STRATEGIES
                # -------------------------------------------------------------------
                if critical_devices or (force_remapping and enable_remapping):
                    # Log findings
                    if critical_devices:
                        print(f"Found {len(runtime_critical_devices)} runtime and {len(predictive_critical_devices)} predictive critical devices in {name}")
                    elif force_remapping and enable_remapping:
                        print(f"Forcing remapping check on {name} (even without critical devices)")
                    
                    # Get current health metrics
                    layer_health = monitor.get_health_summary()
                    
                    # Process runtime faults (higher priority)
                    if runtime_critical_devices:
                        self._apply_runtime_fault_mitigation(
                            name, 
                            runtime_critical_devices,
                            layer_health,
                            health_threshold,
                            enable_soft_mitigation,
                            enable_remapping,
                            enable_layer_reset,
                            force_remapping
                        )
                    
                    # Process predictive faults (lower priority, but still important)
                    if predictive_critical_devices:
                        self._apply_predictive_mitigation(
                            name, 
                            predictive_critical_devices,
                            layer_health,
                            health_threshold,
                            enable_soft_mitigation,
                            enable_remapping,
                            enable_layer_reset
                        )
            
            except Exception as e:
                print(f"Error during health check for {name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Update totals
        total_issues = runtime_issues + predictive_issues
        self.total_faults_detected += total_issues
        
        print(f"Health check completed: {runtime_issues} runtime issues, {predictive_issues} predictive issues")
        return total_issues
    
    # predicitive and runtime detections and mitigation
    
    def initialize_reference_conductances(self):
        """
        Initialize reference conductances for each crossbar to track expected behavior
        With additional error handling and more robust initialization
        """
        self.reference_conductances = {}
        
        # For each crossbar in the self-healing crossbars
        for name, healing_crossbar in self.self_healing_crossbars.items():
            try:
                # Get the original crossbar
                name_parts = name.split('_crossbar_')
                layer_name = name_parts[0]
                crossbar_idx = int(name_parts[1])
                
                if layer_name in self.memristive_layers:
                    module = self.memristive_layers[layer_name]
                    if hasattr(module, 'crossbars') and crossbar_idx < len(module.crossbars):
                        # Use the original crossbar as reference source
                        original_crossbar = module.crossbars[crossbar_idx]
                        
                        # Store a clone of the original conductance matrix
                        self.reference_conductances[name] = original_crossbar.conductance_matrix.clone().detach()
                        
                        # Print success message
                        print(f"Successfully stored reference conductance for {name}")
                    else:
                        # Use healing crossbar as fallback if original not accessible
                        print(f"Warning: Could not find original crossbar for {name}, using healing crossbar as reference")
                        self.reference_conductances[name] = healing_crossbar.crossbar.conductance_matrix.clone().detach()
                else:
                    # Use healing crossbar as fallback
                    print(f"Warning: Could not find layer {layer_name}, using healing crossbar as reference")
                    self.reference_conductances[name] = healing_crossbar.crossbar.conductance_matrix.clone().detach()
                    
            except Exception as e:
                print(f"Error initializing reference conductances for {name}: {e}")
                # Create an empty reference matrix as fallback
                try:
                    # Try to create a placeholder with the same shape
                    if hasattr(healing_crossbar, 'original_shape'):
                        self.reference_conductances[name] = torch.zeros(healing_crossbar.original_shape)
                    elif hasattr(healing_crossbar.crossbar, 'conductance_matrix'):
                        self.reference_conductances[name] = torch.zeros_like(healing_crossbar.crossbar.conductance_matrix)
                    else:
                        print(f"Cannot determine shape for reference conductance of {name}")
                except Exception as nested_e:
                    print(f"Critical error handling reference conductances for {name}: {nested_e}")
        
        # Also store reference from original weights
        for name, layer in self.memristive_model.named_modules():
            if hasattr(layer, 'weight') and hasattr(layer, 'crossbars'):
                # Store original weights as another reference
                weight_reference = layer.weight.data.clone().detach()
                self.reference_weights = getattr(self, 'reference_weights', {})
                self.reference_weights[name] = weight_reference
        
        # Register hooks to track weight updates during training
        self._register_weight_update_hooks()
        
        print(f"Initialized reference conductances for {len(self.reference_conductances)} crossbars")
    
    def _register_weight_update_hooks(self):
        """
        Register hooks to track weight updates during training
        """
        self.weights_updated = False
        
        def weight_update_hook(module, grad_input, grad_output):
            self.weights_updated = True
        
        # Register the hook with each relevant layer
        for name, layer in self.memristive_model.named_modules():
            if hasattr(layer, 'weight'):
                layer.register_backward_hook(weight_update_hook)
                
    def _update_reference_conductances(self):
        """
        Update reference conductances after training updates
        """
        # For each crossbar in the self-healing crossbars
        print("updating refrence conductance")
        for name, healing_crossbar in self.self_healing_crossbars.items():
            # Update the reference conductance matrix
            self.reference_conductances[name] = healing_crossbar.crossbar.conductance_matrix.clone().detach()
    
    def _detect_runtime_faults_by_conductance(self, layer_name, monitor):
        """
        Detect runtime faults based on direct conductance verification and electrical characteristics
        With enhanced performance via adaptive random sampling
        
        Returns:
            List of critical devices with runtime faults
        """
        import time
        import random
        
        critical_devices = []
        
        print(f"Starting runtime fault detection for {layer_name}")
        start_time = time.time()  # Track execution time
        
        # Get basic critical devices from monitor (safe fallback)
        basic_critical = monitor.get_critical_devices()
        print(f"Found {len(basic_critical)} critical devices from basic monitor")
        
        # Special case - if we already have many basic critical devices, just return them
        if len(basic_critical) > 500:
            print(f"Already found {len(basic_critical)} critical devices from basic monitor, skipping detailed check")
            return basic_critical
        
        # For each device, verify conductance implementation
        healing_crossbar = self.self_healing_crossbars.get(layer_name)
        if not healing_crossbar:
            print(f"Warning: No self-healing crossbar found for {layer_name}")
            return basic_critical
        
        # Get reference conductance matrix
        if layer_name not in self.reference_conductances:
            print(f"Warning: No reference conductance found for {layer_name}")
            return basic_critical
        
        reference_conductance_matrix = self.reference_conductances[layer_name]
        
        try:
            # Get the inner MemTorch crossbar
            inner_crossbar = healing_crossbar.crossbar
            
            # Check if the inner crossbar has a conductance matrix
            if not hasattr(inner_crossbar, 'conductance_matrix'):
                print(f"Warning: Inner crossbar for {layer_name} has no conductance_matrix")
                return basic_critical
                
            # Get original shape
            original_shape = healing_crossbar.original_shape
            total_cells = original_shape[0] * original_shape[1]
            
            print(f"Processing crossbar with {total_cells} cells ({original_shape[0]}x{original_shape[1]})")
            
            # ENHANCED SAMPLING APPROACH
            # -------------------------------
            # For large crossbars, use adaptive sampling that varies between runs
            max_sample_size = 2000  # Cap the maximum number of cells to check
            
            # Vary the base sampling rate randomly to avoid checking the same cells every time
            if total_cells > 10000:  # Only use sampling for larger crossbars
                # Vary the sampling rate based on crossbar size and add randomness
                if total_cells > 500000:
                    # Very large crossbar - use very small sample rate with variability
                    base_rate = 0.005  # 0.5% base rate
                    variation = random.uniform(0.001, 0.01)  # Variable between 0.1% and 1%
                elif total_cells > 100000:
                    # Large crossbar
                    base_rate = 0.01   # 1% base rate
                    variation = random.uniform(0.005, 0.02)  # Variable between 0.5% and 2%
                else:
                    # Medium crossbar
                    base_rate = 0.03   # 3% base rate
                    variation = random.uniform(0.01, 0.05)   # Variable between 1% and 5%
                    
                # Apply the randomized sampling rate
                sampling_rate = base_rate + variation
                
                # Calculate sample size with a minimum and maximum
                sample_size = min(max_sample_size, max(100, int(total_cells * sampling_rate)))
                
                print(f"Using adaptive random sampling: checking {sample_size} cells ({sampling_rate*100:.2f}% of total)")
                
                # Generate random indices for sampling - this creates variety between runs
                flat_indices = torch.randperm(total_cells)[:sample_size]
                sampled_rows = (flat_indices // original_shape[1]).tolist()
                sampled_cols = (flat_indices % original_shape[1]).tolist()
                
                # Create sampling coordinate pairs
                cell_coordinates = list(zip(sampled_rows, sampled_cols))
                
                # Add stratified sampling to ensure coverage across the crossbar
                # This adds cells from different regions of the crossbar
                strata_rows = min(10, original_shape[0])  # Number of row strata
                strata_cols = min(10, original_shape[1])  # Number of column strata
                
                # Add 3-5 samples from each stratum (region of the crossbar)
                samples_per_stratum = random.randint(3, 5)
                
                for i in range(strata_rows):
                    for j in range(strata_cols):
                        # Calculate the region boundaries
                        row_start = (i * original_shape[0]) // strata_rows
                        row_end = ((i + 1) * original_shape[0]) // strata_rows
                        col_start = (j * original_shape[1]) // strata_cols
                        col_end = ((j + 1) * original_shape[1]) // strata_cols
                        
                        # Add random samples from this region
                        for _ in range(samples_per_stratum):
                            row = random.randint(row_start, row_end - 1)
                            col = random.randint(col_start, col_end - 1)
                            if (row, col) not in cell_coordinates:
                                cell_coordinates.append((row, col))
            else:
                # For smaller crossbars, use all cells or a fixed sample
                if total_cells <= 2000:
                    # Very small crossbar - process all cells
                    cell_coordinates = [(i, j) for i in range(original_shape[0]) for j in range(original_shape[1])]
                    print(f"Small crossbar: checking all {len(cell_coordinates)} cells")
                else:
                    # Small-medium crossbar - use a fixed sample size with randomness
                    sample_size = min(max_sample_size, max(200, total_cells // 10))
                    flat_indices = torch.randperm(total_cells)[:sample_size]
                    sampled_rows = (flat_indices // original_shape[1]).tolist()
                    sampled_cols = (flat_indices % original_shape[1]).tolist()
                    cell_coordinates = list(zip(sampled_rows, sampled_cols))
                    print(f"Medium crossbar: checking {sample_size} random cells")
                
            # Always include all basic critical devices
            for device in basic_critical:
                if isinstance(device, tuple) and len(device) == 2:
                    row, col = device
                    if (row, col) not in cell_coordinates and row < original_shape[0] and col < original_shape[1]:
                        cell_coordinates.append((row, col))
            
            # FAST PROCESSING
            # -------------------------------
            # Use batch processing where possible to speed up computation
            progress_step = max(1, len(cell_coordinates) // 5)  # Show progress 5 times
            
            # Initialize counters
            processed = 0
            critical_count = 0
            skipped = 0
            
            # Process the selected cells
            for cell_idx, (i, j) in enumerate(cell_coordinates):
                # Show progress periodically
                if cell_idx % progress_step == 0:
                    elapsed = time.time() - start_time
                    print(f"Processed {cell_idx}/{len(cell_coordinates)} cells in {elapsed:.2f}s, found {critical_count} critical")
                    
                logical_pos = (i, j)
                cell_id = logical_pos
                
                # Quickly add basic critical devices and continue
                if cell_id in basic_critical:
                    critical_devices.append(cell_id)
                    critical_count += 1
                    continue
                
                try:
                    # Fast-path processing with minimal error handling for speed
                    
                    # Map to physical position and check bounds
                    physical_row, physical_col = healing_crossbar._map_logical_to_physical(i, j)
                    
                    if (physical_row >= inner_crossbar.conductance_matrix.shape[0] or
                        physical_col >= inner_crossbar.conductance_matrix.shape[1]):
                        skipped += 1
                        continue
                    
                    # Get conductances - try matrix first, then device if needed
                    try:
                        actual_conductance = inner_crossbar.conductance_matrix[physical_row, physical_col].item()
                        
                        # Get expected conductance - try at physical location first, then logical
                        if (physical_row < reference_conductance_matrix.shape[0] and 
                            physical_col < reference_conductance_matrix.shape[1]):
                            expected_conductance = reference_conductance_matrix[physical_row, physical_col].item()
                        elif (i < reference_conductance_matrix.shape[0] and 
                            j < reference_conductance_matrix.shape[1]):
                            expected_conductance = reference_conductance_matrix[i, j].item()
                        else:
                            skipped += 1
                            continue
                        
                        # Calculate error
                        if abs(expected_conductance) > 1e-10:
                            conductance_error = abs(actual_conductance - expected_conductance) / abs(expected_conductance)
                        else:
                            conductance_error = 1.0 if abs(actual_conductance) > 1e-10 else 0.0
                        
                        # Ultra-fast path: If error is very small, skip detailed health calculation
                        if conductance_error < 0.1:
                            processed += 1
                            continue
                        
                        # Use simplified metrics for most cells to improve speed
                        read_var = 0.1
                        write_var = 0.1
                        
                        # Calculate detailed metrics only occasionally for performance
                        if cell_idx % 10 == 0 and hasattr(inner_crossbar, 'devices'):
                            try:
                                actual_device = inner_crossbar.devices[physical_row][physical_col]
                                if hasattr(actual_device, 'x') and hasattr(actual_device, 'd'):
                                    x_var = actual_device.x / actual_device.d
                                    read_var = 4 * x_var * (1 - x_var)  # Simple parabolic approximation
                                    write_var = read_var  # Simplified approximation
                            except:
                                pass
                        
                        # Calculate health score
                        health_score = 1.0 - (0.7 * conductance_error + 0.15 * read_var + 0.15 * write_var)
                        health_score = max(min(health_score, 1.0), 0.0) * 100
                        
                        # Store history only for potentially problematic devices
                        if health_score < 60:
                            metrics = {
                                'conductance_error': float(conductance_error),
                                'read_var': float(read_var),
                                'write_var': float(write_var)
                            }
                            self._update_device_history(layer_name, cell_id, float(health_score), metrics)
                        
                        # Check if critical
                        if health_score < 40 or conductance_error > 0.3:
                            critical_devices.append(cell_id)
                            critical_count += 1
                        
                        processed += 1
                        
                    except Exception:
                        skipped += 1
                        continue
                    
                except Exception:
                    skipped += 1
                    continue
            
            # Report final statistics
            elapsed = time.time() - start_time
            print(f"Fault detection complete: {processed} processed, {skipped} skipped, {len(critical_devices)} critical in {elapsed:.2f}s")
            
            # Limit the number of critical devices to avoid overwhelming mitigation
            if len(critical_devices) > 1000:
                print(f"Limiting from {len(critical_devices)} to 1000 critical devices for mitigation")
                random.shuffle(critical_devices)  # Shuffle to get a random subset
                critical_devices = critical_devices[:1000]
                    
        except Exception as e:
            print(f"Error analyzing conductance for {layer_name}: {e}")
            import traceback
            traceback.print_exc()
            return basic_critical
            
        return critical_devices
    
    def _estimate_read_variability(self, device):
        """
        Estimate read variability for a memristor device
        """
        try:
            # For VTEAM models, this could be estimated from state variable position
            # Closer to boundaries (0 or d) typically means more stability
            normalized_x = device.x / device.d
            
            # Higher variability in the middle of the range
            read_var = 4 * normalized_x * (1 - normalized_x)  # Parabolic function with max at x=0.5
            
            # Adjust based on device properties
            if hasattr(device, 'alpha_on') and hasattr(device, 'alpha_off'):
                # Higher alpha values indicate sharper switching and potentially more variability
                alpha_factor = (device.alpha_on + device.alpha_off) / 4  # Normalize
                read_var *= alpha_factor
            
            return min(read_var, 1.0)
        except:
            # Default value if estimation fails
            return 0.1
    
    def _estimate_write_variability(self, device):
        """
        Estimate write variability for a memristor device
        """
        try:
            # For VTEAM models, estimate based on thresholds and state
            normalized_x = device.x / device.d
            
            # Higher variability near switching thresholds
            v_on_norm = abs(device.v_on)
            v_off_norm = abs(device.v_off)
            
            # Average threshold - lower thresholds lead to higher write variability
            thresh_factor = 1.0 - min(v_on_norm, v_off_norm) / max(0.5, max(v_on_norm, v_off_norm))
            
            # Position-dependent factor - more variability at extremes for write operations
            pos_factor = 0.5 + abs(normalized_x - 0.5)
            
            write_var = thresh_factor * pos_factor
            
            return min(write_var, 1.0)
        except:
            # Default value if estimation fails
            return 0.15
    
    def _apply_hard_reset(self, layer_name, device_ids):
        """
        Apply hard reset to specific devices in a layer.
        Adapted for SelfHealingCrossbar with nested crossbar structure.
        """
        success_count = 0
        
        try:
            healing_crossbar = self.self_healing_crossbars.get(layer_name)
            if not healing_crossbar or layer_name not in self.reference_conductances:
                return 0
            
            inner_crossbar = healing_crossbar.crossbar
            reference_conductance_matrix = self.reference_conductances[layer_name]
            
            # Check if the inner crossbar has tiles
            has_tiles = hasattr(inner_crossbar, 'tile_shape') and inner_crossbar.tile_shape is not None
            
            # Process based on whether tiling is used
            if has_tiles:
                # Tiled crossbar
                for device_id in device_ids:
                    i, j, k = device_id
                    # Get reference conductance
                    ref_conductance = reference_conductance_matrix[i][j][k].item()
                    # Reset device to reference conductance
                    inner_crossbar.devices[i][j][k].set_conductance(ref_conductance)
                    # Update conductance matrix
                    inner_crossbar.conductance_matrix[i][j][k] = ref_conductance
                    success_count += 1
            else:
                # Non-tiled crossbar with SelfHealingCrossbar mapping
                for device_id in device_ids:
                    i, j = device_id  # These are logical positions
                    
                    # Map to physical position
                    try:
                        physical_row, physical_col = healing_crossbar._map_logical_to_physical(i, j)
                        
                        # Get reference conductance (try physical first)
                        try:
                            ref_conductance = reference_conductance_matrix[physical_row][physical_col].item()
                        except:
                            # Fallback: use logical position
                            if i < reference_conductance_matrix.shape[0] and j < reference_conductance_matrix.shape[1]:
                                ref_conductance = reference_conductance_matrix[i][j].item()
                            else:
                                # Skip if out of bounds
                                continue
                        
                        # Reset device to reference conductance
                        inner_crossbar.devices[physical_row][physical_col].set_conductance(ref_conductance)
                        # Update conductance matrix
                        inner_crossbar.conductance_matrix[physical_row][physical_col] = ref_conductance
                        success_count += 1
                    except Exception as e:
                        print(f"Error during reset of device {device_id}: {e}")
                        continue
            
            # Update the crossbar from conductance matrix
            inner_crossbar.update(from_devices=False)
            
            return success_count
        except Exception as e:
            print(f"Error during hard reset: {e}")
            return 0
    
    def _detect_predictive_degradation(self, layer_name, monitor):
        """
        Detect predictive degradation based on:
        - Trend analysis over time 
        - Increasing variability
        - High operation count
        - Drifting health
        - History of mitigations
        
        Returns:
            List of devices showing signs of degradation
        """
        predictive_critical = []
        
        # Get device history for this layer
        layer_history = self.device_history.get(layer_name, {})
        mitigation_history = self.mitigation_history.get(layer_name, {})
        
        # Check each device's history
        for cell_id, history in layer_history.items():
            if len(history) < 3:  # Need at least 3 data points for trend analysis
                continue
                
            # Calculate health trend (negative slope indicates degradation)
            health_values = [entry['health'] for entry in history[-10:]]  # Last 10 readings
            health_trend = self._calculate_trend(health_values)
            
            # Calculate drift trend (positive slope indicates worsening)
            drift_values = [entry['metrics']['drift'] for entry in history[-10:]]
            drift_trend = self._calculate_trend(drift_values)
            
            # Get mitigation count
            mitigation_count = len(mitigation_history.get(cell_id, []))
            recent_mitigations = sum(1 for m in mitigation_history.get(cell_id, []) 
                                    if m['time'] > len(history) - 5)  # Mitigations in last 5 checks
            
            # Check for predictive degradation indicators
            is_critical = False
            
            # Negative health trend (declining health)
            if health_trend < -2.0:
                is_critical = True
                
            # Positive drift trend (increasing drift)
            if drift_trend > 0.05:
                is_critical = True
                
            # Recent health volatility (standard deviation)
            health_volatility = self._calculate_volatility(health_values)
            if health_volatility > 10:  # High volatility in health readings
                is_critical = True
                
            # Multiple recent mitigations indicate instability
            if recent_mitigations >= 2:
                is_critical = True
                
            # Current health score is borderline
            if history[-1]['health'] < 60 and history[-1]['health'] > 40:
                # Only mark as critical if showing trend of degradation
                if health_trend < -1.0 or drift_trend > 0.02:
                    is_critical = True
            
            if is_critical:
                predictive_critical.append(cell_id)
                
        return predictive_critical
    
    def _apply_runtime_fault_mitigation(self, name, critical_devices, layer_health, 
                                       health_threshold, enable_soft_mitigation, 
                                       enable_remapping, enable_layer_reset, force_remapping):
        """
        Apply mitigation strategies for runtime faults.
        Runtime faults require more aggressive mitigation.
        """
        # Extract layer name
        name_parts = name.split('_crossbar_')
        layer_name = name_parts[0]
        
        # CASE 1: Forced remapping (highest priority if enabled)
        if force_remapping and enable_remapping:
            print(f"[Runtime] Forcing remapping strategy on {name}")
            mitigated = self._apply_redundancy_remapping(name, critical_devices)
            print(f"[Runtime] Remapped {mitigated}/{len(critical_devices)} devices")
            
            # Record mitigation actions
            for cell_id in critical_devices[:mitigated]:
                self._record_mitigation(name, cell_id, 'hard_remap', 'forced')
            
            # Apply hard reset to remaining devices if remapping failed
            if mitigated < len(critical_devices) and enable_layer_reset:
                remaining_devices = critical_devices[mitigated:]
                print(f"[Runtime] Applying hard reset to remaining {len(remaining_devices)} devices")
                self._apply_hard_reset(name, remaining_devices)
                
                # Record mitigation actions
                for cell_id in remaining_devices:
                    self._record_mitigation(name, cell_id, 'hard_reset', 'forced')
            
        # CASE 2: Normal mitigation based on health
        else:
            # For runtime faults, use more aggressive mitigation strategy
            if layer_health['avg_health'] > health_threshold and enable_remapping:
                # Medium-high health: Try remapping first
                print(f"[Runtime] Using remapping for {name} (health: {layer_health['avg_health']:.2f}%)")
                mitigated = self._apply_redundancy_remapping(name, critical_devices)
                print(f"[Runtime] Remapped {mitigated}/{len(critical_devices)} devices")
                
                # Record mitigation actions
                for cell_id in critical_devices[:mitigated]:
                    self._record_mitigation(name, cell_id, 'remap', 'medium_health')
                
                # For remaining devices, try hard reset if available
                if mitigated < len(critical_devices) and enable_layer_reset:
                    remaining_devices = critical_devices[mitigated:]
                    print(f"[Runtime] Applying hard reset to remaining {len(remaining_devices)} devices")
                    self._apply_hard_reset(name, remaining_devices)
                    
                    # Record mitigation actions
                    for cell_id in remaining_devices:
                        self._record_mitigation(name, cell_id, 'hard_reset', 'fallback')
                        
            elif layer_health['avg_health'] > health_threshold/2 and enable_layer_reset:
                # Medium-low health: Try hard reset
                print(f"[Runtime] Using hard reset for {name} (health: {layer_health['avg_health']:.2f}%)")
                self._apply_hard_reset(name, critical_devices)
                
                # Record mitigation actions
                for cell_id in critical_devices:
                    self._record_mitigation(name, cell_id, 'hard_reset', 'low_health')
                
            else:
                # Very low health: Try layer reset as last resort
                print(f"[Runtime] Critical health ({layer_health['avg_health']:.2f}%), attempting layer reset")
                if self.reset_critical_layer(layer_name):
                    self.total_layer_resets += 1
                    
                    # Record mitigation action
                    for cell_id in critical_devices:
                        self._record_mitigation(name, cell_id, 'layer_reset', 'critical_health')
    
    def _apply_predictive_mitigation(self, name, critical_devices, layer_health, 
                                    health_threshold, enable_soft_mitigation, 
                                    enable_remapping, enable_layer_reset):
        """
        Apply mitigation strategies for predictively identified devices.
        Predictive mitigation starts with gentler approaches.
        """
        # Extract layer name
        name_parts = name.split('_crossbar_')
        layer_name = name_parts[0]
        
        # For each device, check mitigation history to decide approach
        for cell_id in critical_devices:
            mitigation_history = self.mitigation_history.get(name, {}).get(cell_id, [])
            soft_reset_count = sum(1 for m in mitigation_history if m['action'] == 'soft_reset')
            hard_reset_count = sum(1 for m in mitigation_history if m['action'] == 'hard_reset')
            
            # First try: Soft reset if enabled
            if enable_soft_mitigation and soft_reset_count < 3:
                print(f"[Predictive] Applying soft reset to {name} device {cell_id}")
                self._apply_soft_mitigation(name, [cell_id])
                self._record_mitigation(name, cell_id, 'soft_reset', 'predictive_first')
                
            # Second try: Hard reset if soft reset used too many times
            elif enable_layer_reset and soft_reset_count >= 3 and hard_reset_count < 2:
                print(f"[Predictive] Applying hard reset to {name} device {cell_id} (after {soft_reset_count} soft resets)")
                self._apply_hard_reset(name, [cell_id])
                self._record_mitigation(name, cell_id, 'hard_reset', 'predictive_escalation')
                
            # Last resort: Remapping if everything else fails
            elif enable_remapping and (soft_reset_count + hard_reset_count) >= 5:
                print(f"[Predictive] Applying remapping to {name} device {cell_id} (after multiple resets)")
                mitigated = self._apply_redundancy_remapping(name, [cell_id])
                if mitigated > 0:
                    self._record_mitigation(name, cell_id, 'remap', 'predictive_last_resort')
                         
    def _update_device_history(self, layer_name, cell_id, health_score, metrics):
        """
        Update the history tracking for a device
        """
        if layer_name not in self.device_history:
            self.device_history[layer_name] = {}
            
        if cell_id not in self.device_history[layer_name]:
            self.device_history[layer_name][cell_id] = []
            
        # Add current metrics to history
        self.device_history[layer_name][cell_id].append({
            'health': health_score,
            'metrics': metrics,
            'time': len(self.device_history[layer_name][cell_id])
        })
        
        # Limit history length
        if len(self.device_history[layer_name][cell_id]) > 100:
            self.device_history[layer_name][cell_id] = self.device_history[layer_name][cell_id][-100:]
            
    def _record_mitigation(self, layer_name, cell_id, action, reason):
        """
        Record mitigation action for a device
        """
        if layer_name not in self.mitigation_history:
            self.mitigation_history[layer_name] = {}
            
        if cell_id not in self.mitigation_history[layer_name]:
            self.mitigation_history[layer_name][cell_id] = []
            
        # Get current time (using history length as a proxy)
        current_time = 0
        if layer_name in self.device_history and cell_id in self.device_history[layer_name]:
            current_time = len(self.device_history[layer_name][cell_id])
            
        # Record mitigation
        self.mitigation_history[layer_name][cell_id].append({
            'action': action,
            'reason': reason,
            'time': current_time
        })
        
    def _calculate_trend(self, values):
        """
        Calculate the trend (slope) of a series of values
        Positive slope means increasing, negative means decreasing
        """
        if not values or len(values) < 2:
            return 0
            
        x = list(range(len(values)))
        x_mean = sum(x) / len(x)
        y_mean = sum(values) / len(values)
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(len(values)))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(len(values)))
        
        if denominator == 0:
            return 0
            
        return numerator / denominator
        
    def _calculate_volatility(self, values):
        """
        Calculate volatility (standard deviation) of values
        """
        if not values or len(values) < 2:
            return 0
            
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return (variance ** 0.5)  # Standard deviation
     
    #  mitigation techniques
    def _apply_soft_mitigation(self, crossbar_name, critical_devices):
        """Apply soft mitigation by adjusting conductance toward middle range."""
        # Extract the layer_name and crossbar_idx
        name_parts = crossbar_name.split('_crossbar_')
        layer_name = name_parts[0]
        crossbar_idx = int(name_parts[1])
        
        if layer_name not in self.memristive_layers:
            return 0
        
        module = self.memristive_layers[layer_name]
        if not hasattr(module, 'crossbars') or crossbar_idx >= len(module.crossbars):
            return 0
        
        crossbar = module.crossbars[crossbar_idx]
        monitor = self.health_monitors[crossbar_name]
        
        # Get mid-range conductance
        g_mid = (monitor.g_min + monitor.g_max) / 2
        
        # Apply soft mitigation by moving conductance toward middle range
        mitigated_count = 0
        for row, col in critical_devices:
            current_g = crossbar.conductance_matrix[row, col]
            # Move 30% of the way toward mid-range
            new_g = current_g * 0.7 + g_mid * 0.3
            crossbar.conductance_matrix[row, col] = new_g
            mitigated_count += 1
        
        self.total_soft_mitigations += mitigated_count
        print(f"Applied soft mitigation to {mitigated_count} devices in {crossbar_name}")
        return mitigated_count
    
    def _apply_redundancy_remapping(self, crossbar_name, critical_devices):
        """
        Apply redundancy-based remapping that works with forward pass,
        using a non-recursive approach.
        """
        # Get the self-healing crossbar for this name
        if crossbar_name not in self.self_healing_crossbars:
            return 0
            
        # Extract the layer_name and crossbar_idx
        name_parts = crossbar_name.split('_crossbar_')
        layer_name = name_parts[0]
        crossbar_idx = int(name_parts[1])
            
        if layer_name not in self.memristive_layers:
            return 0
            
        module = self.memristive_layers[layer_name]
        if not hasattr(module, 'crossbars') or crossbar_idx >= len(module.crossbars):
            return 0
            
        # Get the original crossbar
        original_crossbar = module.crossbars[crossbar_idx]
            
        # Get the self-healing crossbar
        healing_crossbar = self.self_healing_crossbars[crossbar_name]
        
        # Key addition: Check if this crossbar has been integrated with the module
        if not hasattr(healing_crossbar, '_integrated_with_original'):
            print(f"First-time integration of self-healing crossbar for {crossbar_name}")
            try:
                # Store the original weights/conductances for remapping
                original_conductance = original_crossbar.conductance_matrix.detach().clone()
                
                # Important: Update healing_crossbar with original values without replacing crossbar
                if hasattr(healing_crossbar, 'write_conductance_matrix'):
                    # Create a properly sized matrix for the healing crossbar
                    temp_matrix = torch.zeros(healing_crossbar.original_shape)
                    
                    # Copy what fits
                    copy_rows = min(original_conductance.shape[0], healing_crossbar.original_shape[0])
                    copy_cols = min(original_conductance.shape[1], healing_crossbar.original_shape[1])
                    
                    if copy_rows > 0 and copy_cols > 0:
                        temp_matrix[:copy_rows, :copy_cols] = original_conductance[:copy_rows, :copy_cols]
                        
                        # Write to self-healing crossbar
                        healing_crossbar.write_conductance_matrix(temp_matrix)
                        print(f"Updated self-healing crossbar with values from original crossbar")
                
                # Flag that this crossbar has been integrated
                healing_crossbar._integrated_with_original = True
                
                # IMPORTANT: Instead of replacing the original crossbar or wrapping methods,
                # we'll just use them both separately and apply remapping within our health check
                print("Successfully initialized self-healing crossbar for remapping")
                
            except Exception as e:
                print(f"Error during crossbar initialization: {e}")
                import traceback
                traceback.print_exc()
                return 0
        
        # Now apply remapping to critical devices
        remapped_count = 0
        for row, col in critical_devices:
            # Check if within bounds of the healing crossbar's original shape
            if (row >= healing_crossbar.original_shape[0] or 
                col >= healing_crossbar.original_shape[1]):
                print(f"Warning: Device ({row},{col}) out of bounds, skipping remapping")
                continue
                    
            # Try to remap this device using the self-healing crossbar's remap_device method
            try:
                if healing_crossbar.remap_device(row, col):
                    # Get the physical location after remapping
                    physical_row, physical_col = healing_crossbar._map_logical_to_physical(row, col)
                    
                    # Get the conductance value from the remapped device
                    remapped_conductance = healing_crossbar.crossbar.conductance_matrix[physical_row, physical_col]
                    
                    # If the physical location is outside the original crossbar dimensions,
                    # we can't directly update the original but still track it for the self-healing crossbar
                    if (row < original_crossbar.conductance_matrix.shape[0] and 
                        col < original_crossbar.conductance_matrix.shape[1]):
                        # Update the conductance in the original crossbar for consistency
                        # This keeps both crossbars somewhat synchronized
                        try:
                            original_crossbar.conductance_matrix[row, col] = remapped_conductance
                        except Exception as e:
                            print(f"Warning: Could not update original crossbar: {e}")
                    
                    remapped_count += 1
                    self.total_remappings += 1
                    print(f"Remapped device ({row},{col}) to ({physical_row},{physical_col})")
                else:
                    print(f"Could not remap device ({row},{col}) - no redundant devices available")
            except Exception as e:
                print(f"Error during remapping of device ({row},{col}): {e}")
                    
            # Stop if we've run out of redundant devices
            if not healing_crossbar.available_redundant:
                print("No more redundant devices available")
                break
                
        print(f"Remapped {remapped_count} devices in {crossbar_name} using redundancy")
            
        # Update the count of available redundant devices
        self.redundant_available = sum(len(sb.available_redundant) for sb in self.self_healing_crossbars.values())
            
        return remapped_count
    
    def reset_critical_layer(self, layer_name):
        """Reset a critical layer back to its original weights as a last resort."""
        if layer_name not in self.memristive_layers or layer_name not in self.original_weights:
            return False
        
        try:
            # Get the memristive module
            module = self.memristive_layers[layer_name]
            
            # Check if module has weight parameter
            if not hasattr(module, 'weight'):
                return False
            
            # Reset weight to original value
            module.weight.data.copy_(self.original_weights[layer_name])
            
            # If module has crossbars, update them
            if hasattr(module, 'crossbars'):
                # Re-map weights to crossbars (simplistic approach)
                for crossbar in module.crossbars:
                    # This is a simplified approach - in a real implementation, 
                    # you would need to follow the same mapping routine used during initialization
                    crossbar.conductance_matrix.copy_(module.weight.data)
            
            print(f"Reset layer {layer_name} to original weights")
            return True
        
        except Exception as e:
            print(f"Error resetting layer {layer_name}: {e}")
            return False
    
    # utility methods:
    def inject_faults(self, fault_density=0.05, distribution='random'):
        """
        Inject faults into memristive layers for testing.
        
        Args:
            fault_density: Proportion of devices to make faulty
            distribution: How to distribute faults ('random', 'clustered', or 'gradient')
            
        Returns:
            Number of faults injected
        """
        total_faults = 0
        
        # Inject faults into each crossbar
        for name, monitor in self.health_monitors.items():
            try:
                # Get the crossbar
                name_parts = name.split('_crossbar_')
                layer_name = name_parts[0]
                crossbar_idx = int(name_parts[1])
                
                if layer_name not in self.memristive_layers:
                    continue
                
                module = self.memristive_layers[layer_name]
                if not hasattr(module, 'crossbars') or crossbar_idx >= len(module.crossbars):
                    continue
                
                crossbar = module.crossbars[crossbar_idx]
                
                # Get total number of devices
                total_devices = crossbar.conductance_matrix.numel()
                num_faults = int(total_devices * fault_density)
                
                # Distribute faults based on chosen pattern
                if distribution == 'clustered':
                    # Create clustered faults by selecting central points and adding neighbors
                    rows, cols = crossbar.conductance_matrix.shape
                    centers = torch.randperm(total_devices)[:num_faults // 5]  # Select 1/5 as many centers
                    
                    # Convert to coordinates
                    center_rows = centers // cols
                    center_cols = centers % cols
                    
                    fault_locations = set()
                    for i in range(len(centers)):
                        r, c = center_rows[i], center_cols[i]
                        # Add center and neighbors in a small cluster
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < rows and 0 <= nc < cols:
                                    fault_locations.add((nr.item(), nc.item()))
                                    if len(fault_locations) >= num_faults:
                                        break
                            if len(fault_locations) >= num_faults:
                                break
                        if len(fault_locations) >= num_faults:
                            break
                    
                    # Convert set to lists of rows and columns
                    fault_rows = [r for r, _ in fault_locations]
                    fault_cols = [c for _, c in fault_locations]
                    
                elif distribution == 'gradient':
                    # Create faults with higher density at one side (aging gradient)
                    rows, cols = crossbar.conductance_matrix.shape
                    all_rows = torch.arange(rows).repeat_interleave(cols)
                    all_cols = torch.arange(cols).repeat(rows)
                    
                    # Calculate distance from top-left corner
                    distances = all_rows.float() / rows + all_cols.float() / cols
                    
                    # Normalize to [0,1] and apply probability gradient
                    probabilities = 1 - distances / 2  # Higher probability near origin
                    
                    # Sample according to probability distribution
                    selected = torch.rand(total_devices) < probabilities * fault_density * 2
                    selected_indices = torch.nonzero(selected).squeeze()
                    
                    # Take only the number we need
                    if selected_indices.numel() > num_faults:
                        selected_indices = selected_indices[:num_faults]
                    elif selected_indices.numel() < num_faults:
                        # If we didn't get enough, sample randomly for the rest
                        remaining = num_faults - selected_indices.numel()
                        excluded = set(selected_indices.tolist())
                        candidates = [i for i in range(total_devices) if i not in excluded]
                        if candidates:
                            additional = torch.tensor(random.sample(candidates, min(remaining, len(candidates))))
                            selected_indices = torch.cat([selected_indices, additional])
                    
                    # Convert to coordinates
                    fault_rows = selected_indices // cols
                    fault_cols = selected_indices % cols
                    
                else:  # 'random' distribution (default)
                    # Randomly select devices to fault
                    rows, cols = crossbar.conductance_matrix.shape
                    indices = torch.randperm(total_devices)[:num_faults]
                    fault_rows = indices // cols
                    fault_cols = indices % cols
                    
                    fault_rows = fault_rows.tolist()
                    fault_cols = fault_cols.tolist()

                    # Add error handling
                    if len(fault_rows) < num_faults:
                        print(f"Warning: Could only generate {len(fault_rows)} clustered faults instead of {num_faults}")
                
                # Inject faults - half stuck at HRS, half at LRS
                valid_fault_rows = []
                valid_fault_cols = []
                for i in range(len(fault_rows)):
                    if (0 <= fault_rows[i] < rows and 0 <= fault_cols[i] < cols):
                        valid_fault_rows.append(fault_rows[i])
                        valid_fault_cols.append(fault_cols[i])
                    else:
                        print(f"Warning: Ignoring out-of-bounds fault at ({fault_rows[i]}, {fault_cols[i]})")
                
                fault_rows = valid_fault_rows
                fault_cols = valid_fault_cols
                
                # Inject faults - half stuck at HRS, half at LRS
                half = len(fault_rows) // 2
                
                # Stuck at HRS (low conductance)
                for i in range(min(half, len(fault_rows))):
                    crossbar.conductance_matrix[fault_rows[i], fault_cols[i]] = monitor.g_min
                    # Force health score to be low
                    monitor.health_scores[fault_rows[i], fault_cols[i]] = 0
                    # Also make stability and stress worse
                    monitor.stability_index[fault_rows[i], fault_cols[i]] = 0.1
                    monitor.stress[fault_rows[i], fault_cols[i]] = 0.9
                
                # Stuck at LRS (high conductance)
                for i in range(half, min(num_faults, len(fault_rows))):
                    crossbar.conductance_matrix[fault_rows[i], fault_cols[i]] = monitor.g_max
                    # Force health score to be low
                    monitor.health_scores[fault_rows[i], fault_cols[i]] = 0
                    # Also make stability and stress worse
                    monitor.stability_index[fault_rows[i], fault_cols[i]] = 0.1
                    monitor.stress[fault_rows[i], fault_cols[i]] = 0.9
                
                total_faults += min(num_faults, len(fault_rows))
                print(f"Injected {min(num_faults, len(fault_rows))} faults into {name}")
            
            except Exception as e:
                print(f"Error injecting faults into {name}: {e}")
        
        self.total_faults_detected += total_faults
        return total_faults
    
    def get_health_report(self, detailed=False):
        """Generate a comprehensive health report for the system."""
        report = {
            'total_operations': self.total_operations,
            'total_faults_detected': self.total_faults_detected,
            'total_soft_mitigations': self.total_soft_mitigations,
            'total_remappings': self.total_remappings,
            'total_layer_resets': self.total_layer_resets,
            'redundant_available': self.redundant_available,
            'layer_health': {}
        }
        
        # Get health for each layer
        total_health_sum = 0
        total_devices = 0
        
        for name, monitor in self.health_monitors.items():
            try:
                health_summary = monitor.get_health_summary()
                report['layer_health'][name] = health_summary
                
                # Aggregate stats
                total_health_sum += health_summary['avg_health'] * monitor.shape[0] * monitor.shape[1]
                total_devices += monitor.shape[0] * monitor.shape[1]
            except Exception as e:
                print(f"Error getting health summary for {name}: {e}")
        
        # Calculate overall health
        if total_devices > 0:
            report['overall_health'] = total_health_sum / total_devices
        else:
            report['overall_health'] = 100.0
        
        # Add a compatible field name for avg_health
        report['avg_health'] = report['overall_health']
        
        # Calculate mitigation efficiency
        if report['total_faults_detected'] > 0:
            report['mitigation_efficiency'] = (report['total_soft_mitigations'] + 
                                               report['total_remappings'] * 2) / report['total_faults_detected']
        else:
            report['mitigation_efficiency'] = 1.0
        
        # Calculate estimated lifetime based on current degradation rate
        if self.total_operations > 0:
            degradation_rate = (100 - report['avg_health']) / self.total_operations
            if degradation_rate > 0:
                remaining_health = report['avg_health'] - 20  # Assume 20% is minimum acceptable health
                estimated_remaining_ops = remaining_health / degradation_rate
                report['estimated_remaining_percentage'] = 100 * estimated_remaining_ops / self.total_operations
            else:
                report['estimated_remaining_percentage'] = float('inf')
        
        # Add more detailed health metrics if requested
        if detailed:
            # Calculate stress and stability across all monitors
            stress_values = torch.cat([monitor.stress.flatten() for monitor in self.health_monitors.values()])
            stability_values = torch.cat([monitor.stability_index.flatten() for monitor in self.health_monitors.values()])
            
            report['avg_stress'] = stress_values.mean().item()
            report['avg_stability'] = stability_values.mean().item()
            report['high_stress_percent'] = 100 * (stress_values > 0.7).sum().item() / stress_values.numel()
            report['low_stability_percent'] = 100 * (stability_values < 0.3).sum().item() / stability_values.numel()
            
            # Add redundancy metrics
            total_redundant = sum(len(sb.available_redundant) + len(sb.remapping) 
                                 for sb in self.self_healing_crossbars.values())
            report['total_redundant_devices'] = total_redundant
            report['redundant_used'] = total_redundant - self.redundant_available
            report['redundant_used_percent'] = 100 * report['redundant_used'] / total_redundant if total_redundant > 0 else 0
        
        return report
    
    def check_health_and_mitigate(self, force_check=False):
        """Alias for check_health to maintain compatibility with existing code."""
        return self.check_health(force_check=force_check)
    
    def visualize_health(self):
        """Visualize health of all memristive layers."""
        num_monitors = len(self.health_monitors)
        if num_monitors == 0:
            print("No health monitors to visualize")
            return
        
        # Determine grid size
        cols = min(3, num_monitors)
        rows = (num_monitors + cols - 1) // cols
        
        plt.figure(figsize=(5*cols, 4*rows))
        
        for i, (name, monitor) in enumerate(self.health_monitors.items()):
            plt.subplot(rows, cols, i+1)
            plt.imshow(monitor.health_scores.numpy(), cmap='RdYlGn', vmin=0, vmax=100)
            plt.colorbar(label='Health Score')
            plt.title(f'Health: {name}')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_system_health(self):
        """Comprehensive visualization of system health including all mitigation strategies."""
        # Create a figure with subplot grid
        plt.figure(figsize=(16, 12))
        
        # 1. Overall health metrics
        plt.subplot(2, 3, 1)
        report = self.get_health_report(detailed=True)
        
        # Create bar chart of key metrics
        metrics = ['avg_health', 'total_faults_detected', 'total_soft_mitigations', 
                   'total_remappings', 'total_layer_resets']
        values = [report.get(m, 0) for m in metrics]
        plt.bar(range(len(metrics)), values, color=['green', 'red', 'blue', 'purple', 'orange'])
        plt.xticks(range(len(metrics)), [m.replace('total_', '').replace('_', ' ') for m in metrics], rotation=45)
        plt.title('System Health Overview')
        
        # 2. Example health monitor visualization
        if self.health_monitors:
            plt.subplot(2, 3, 2)
            # Get first monitor for example
            monitor_name = list(self.health_monitors.keys())[0]
            monitor = self.health_monitors[monitor_name]
            plt.imshow(monitor.health_scores.numpy(), cmap='RdYlGn', vmin=0, vmax=100)
            plt.colorbar(label='Health Score')
            plt.title(f'Example Health: {monitor_name}')
        
        # 3. Example self-healing crossbar visualization
        if self.self_healing_crossbars:
            plt.subplot(2, 3, 3)
            # Get first self-healing crossbar for example
            crossbar_name = list(self.self_healing_crossbars.keys())[0]
            healing_crossbar = self.self_healing_crossbars[crossbar_name]
            
            # Show remapping status
            remapping_mask = torch.zeros(healing_crossbar.effective_shape)
            
            # Mark original area
            for i in range(healing_crossbar.original_shape[0]):
                for j in range(healing_crossbar.original_shape[1]):
                    if (i, j) in healing_crossbar.remapping:
                        physical_row, physical_col = healing_crossbar.remapping[(i, j)]
                        remapping_mask[physical_row, physical_col] = 2  # Remapped destination
                        remapping_mask[i, j] = 1  # Original position (remapped source)
                    else:
                        remapping_mask[i, j] = 0.5  # Normal original device
            
            # Mark redundant area
            for i in range(healing_crossbar.effective_shape[0]):
                for j in range(healing_crossbar.effective_shape[1]):
                    if healing_crossbar.is_redundant[i, j] and remapping_mask[i, j] == 0:
                        remapping_mask[i, j] = 0.25  # Unused redundant
            
            plt.imshow(remapping_mask.numpy(), cmap='coolwarm')
            plt.colorbar(label='Device Status')
            plt.title(f'Remapping: {crossbar_name}')
        
        # 4. Mitigation statistics over time (placeholder - would need to track history)
        plt.subplot(2, 3, 4)
        # Create placeholder values showing increasing mitigations over time
        ops = range(0, self.total_operations, max(1, self.total_operations // 20))
        if not ops:
            ops = [0, 1]  # Default if no operations yet
        
        soft_vals = [self.total_soft_mitigations * o / self.total_operations for o in ops]
        remap_vals = [self.total_remappings * o / self.total_operations for o in ops]
        reset_vals = [self.total_layer_resets * o / self.total_operations for o in ops]
        
        plt.plot(ops, soft_vals, 'b-', label='Soft Mitigations')
        plt.plot(ops, remap_vals, 'g-', label='Remappings')
        plt.plot(ops, reset_vals, 'r-', label='Layer Resets')
        plt.xlabel('Operations')
        plt.ylabel('Count')
        plt.title('Mitigation Strategies Over Time')
        plt.legend()
        
        # 5. Redundancy usage
        plt.subplot(2, 3, 5)
        labels = ['Available', 'Used']
        sizes = [self.redundant_available, 
                 sum(len(sb.remapping) for sb in self.self_healing_crossbars.values())]
        
        # Add a small value if both are zero to avoid empty pie chart
        if sum(sizes) == 0:
            sizes = [1, 0]
            
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        plt.title('Redundant Device Usage')
        
        # 6. Estimated remaining lifetime
        plt.subplot(2, 3, 6)
        if 'estimated_remaining_percentage' in report:
            remaining = min(report['estimated_remaining_percentage'], 100)  # Cap at 100%
        else:
            remaining = 100
            
        # Create a gauge-like visualization
        plt.pie([remaining, 100-remaining], labels=['Remaining', 'Used'],
                colors=['green', 'lightgray'], startangle=90, counterclock=False)
        plt.title(f'Estimated Remaining Lifetime: {remaining:.1f}%')
        
        plt.tight_layout()
        plt.show()