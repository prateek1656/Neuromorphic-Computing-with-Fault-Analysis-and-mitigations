import torch
import matplotlib.pyplot as plt

class HealthMonitor:
    """
    Comprehensive health monitoring and prediction system for memristive crossbars.
    Combines features from MemristorHealthMonitor and MemristorHealthPredictor.
    """
    
    def __init__(self, crossbar):
        """Initialize the enhanced health monitor."""
        self.crossbar = crossbar
        self.shape = crossbar.conductance_matrix.shape
        
        # Extract device parameters
        try:
            # Try direct access first
            self.r_on = getattr(crossbar, 'r_on', None)
            self.r_off = getattr(crossbar, 'r_off', None)
            
            # If not available, try through memristor_model_params
            if self.r_on is None and hasattr(crossbar, 'memristor_model_params'):
                self.r_on = crossbar.memristor_model_params.get('r_on', 100)
                self.r_off = crossbar.memristor_model_params.get('r_off', 16000.0)
            
            # Fall back to defaults if necessary
            self.r_on = self.r_on if self.r_on is not None else 100
            self.r_off = self.r_off if self.r_off is not None else 16000.0
        except Exception:
            # Fallback to defaults
            self.r_on = 100
            self.r_off = 16000.0
        
        # Derived properties
        self.g_min = 1.0 / self.r_off
        self.g_max = 1.0 / self.r_on
        self.g_mid = (self.g_max + self.g_min) / 2
        self.g_range = self.g_max - self.g_min
        
        # Health tracking metrics
        self.stress = torch.zeros(self.shape)
        self.health_scores = torch.ones(self.shape) * 100
        self.write_count = torch.zeros(self.shape)
        self.read_count = torch.zeros(self.shape)
        self.stability_index = torch.ones(self.shape)
        self.operation_counts = torch.zeros(self.shape)
        
        # History tracking for predictive analytics
        self.g_history = [crossbar.conductance_matrix.clone()]
        self.max_history = 20
        
        # Fault prediction parameters
        self.failure_probability = torch.zeros(self.shape)
        self.failure_threshold = 0.8
        
        print(f"Initialized enhanced health monitor for {self.shape[0]}x{self.shape[1]} crossbar")
    
    def update_health_metrics(self, voltage_applied=None, write_op=False):
        """Update health metrics based on recent operations and current state."""
        current_g = self.crossbar.conductance_matrix
        
        # Update operation counters
        if write_op:
            self.write_count += torch.ones(self.shape)
        else:
            self.read_count += torch.ones(self.shape)
            
        self.operation_counts += 1
        
        # Calculate normalized conductance and deviation from ideal
        g_normalized = (current_g - self.g_min) / self.g_range
        g_normalized = torch.clamp(g_normalized, 0, 1)
        deviation_from_mid = 2 * torch.abs(g_normalized - 0.5)
        
        # Calculate stability metrics if we have history
        if len(self.g_history) >= 3:
            prev_g1 = self.g_history[-1]
            prev_g2 = self.g_history[-2]
            
            # Compute changes
            recent_change1 = torch.abs(current_g - prev_g1) / self.g_range
            recent_change2 = torch.abs(prev_g1 - prev_g2) / self.g_range
            
            # Compute accelerations
            change_acceleration = torch.abs(recent_change1 - recent_change2)
            
            # Update stability index - lower means less stable
            self.stability_index = self.stability_index * 0.9 + (1.0 - change_acceleration * 10) * 0.1
            self.stability_index = torch.clamp(self.stability_index, 0.0, 1.0)
            
            # Large changes indicate instability
            unstable = recent_change1 > 0.1
            self.health_scores[unstable] -= 0.5  # Additional penalty
        
        # Calculate stress from operation
        op_stress_factor = 0.02 if write_op else 0.001
        voltage_factor = 1.0
        
        if voltage_applied is not None:
            max_voltage = self.crossbar.max_voltage if hasattr(self.crossbar, 'max_voltage') else 1.0
            voltage_normalized = torch.abs(voltage_applied) / max_voltage
            voltage_factor = 1.0 + voltage_normalized
        
        # New stress from this operation
        new_stress = op_stress_factor * voltage_factor * (0.5 + deviation_from_mid)
        
        # Stress increases for unstable devices
        new_stress = new_stress * (2.0 - self.stability_index)
        
        # Update accumulated stress (with partial decay over time)
        self.stress = self.stress * 0.99 + new_stress
        self.stress = torch.clamp(self.stress, 0.0, 1.0)
        
        # Update health scores
        self.health_scores = 100 * (1.0 - self.stress)
        self.health_scores = torch.clamp(self.health_scores, 0, 100)
        
        # Store current state in history
        self.g_history.append(current_g.clone())
        if len(self.g_history) > self.max_history:
            self.g_history.pop(0)
    
    def predict_faults(self):
        """Predict which devices are likely to fail soon."""
        # 1. Extreme conductance values indicate potential issues
        g_normalized = (self.crossbar.conductance_matrix - self.g_min) / self.g_range
        g_normalized = torch.clamp(g_normalized, 0, 1)
        proximity_to_extreme = torch.max(
            4 * torch.pow(g_normalized, 2),
            4 * torch.pow(1 - g_normalized, 2)
        )
        proximity_to_extreme = torch.clamp(proximity_to_extreme, 0.0, 1.0)
        
        # 2. Instability is a strong predictor of failure
        instability_factor = 1.0 - self.stability_index
        
        # 3. Recent conductance drift trend
        drift_factor = torch.zeros(self.shape)
        if len(self.g_history) >= 5:
            g_diffs = []
            for i in range(1, min(5, len(self.g_history))):
                g_diff = (self.g_history[-i] - self.g_history[-i-1]) / self.g_range
                g_diffs.append(g_diff)
            
            g_diffs_tensor = torch.stack(g_diffs)
            drift_magnitude = torch.mean(torch.abs(g_diffs_tensor), dim=0)
            drift_consistency = torch.std(torch.sign(g_diffs_tensor), dim=0)
            
            drift_factor = drift_magnitude * (2.0 - drift_consistency)
            drift_factor = torch.clamp(drift_factor * 5.0, 0.0, 1.0)
        
        # 4. Stress history (accumulated wear)
        stress_factor = self.stress
        
        # Combine factors with weights to get failure probability
        self.failure_probability = (
            0.4 * stress_factor +
            0.3 * instability_factor +
            0.2 * proximity_to_extreme +
            0.1 * drift_factor
        )
        
        return self.failure_probability
    
    def get_critical_devices(self, health_threshold=10, probability_threshold=0.8, extreme_threshold=0.02):
        """Get devices in critical condition that should be prioritized for healing."""
        # Run prediction to update failure probabilities
        self.predict_faults()
        
        # Get current conductance
        current_g = self.crossbar.conductance_matrix
        g_normalized = (current_g - self.g_min) / self.g_range
        g_normalized = torch.clamp(g_normalized, 0, 1)
        
        # Find critical devices
        critical_devices = []
        
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                # Check all criteria
                if (self.health_scores[i, j] < health_threshold or
                    self.failure_probability[i, j] > probability_threshold or
                    g_normalized[i, j] < extreme_threshold or 
                    g_normalized[i, j] > (1 - extreme_threshold)):
                    critical_devices.append((i, j))
        
        return critical_devices
    
    def get_health_summary(self):
        """Get a comprehensive health report."""
        # Run prediction to ensure up-to-date values
        self.predict_faults()
        
        # Calculate failure categories
        imminent_failures = (self.failure_probability > 0.8)
        upcoming_failures = (self.failure_probability > 0.5) & (self.failure_probability <= 0.8)
        at_risk_devices = (self.failure_probability > 0.3) & (self.failure_probability <= 0.5)
        
        # Count devices in each category
        imminent_count = imminent_failures.sum().item()
        upcoming_count = upcoming_failures.sum().item()
        at_risk_count = at_risk_devices.sum().item()
        total_devices = self.shape[0] * self.shape[1]
        
        # Calculate average values
        avg_health = self.health_scores.mean().item()
        min_health = self.health_scores.min().item()
        avg_stability = self.stability_index.mean().item()
        avg_failure_prob = self.failure_probability.mean().item()
        
        # Check for stuck devices
        g_values = self.crossbar.conductance_matrix
        g_normalized = (g_values - self.g_min) / self.g_range
        
        stuck_at_min = (g_normalized < 0.05).sum().item()
        stuck_at_max = (g_normalized > 0.95).sum().item()
        
        return {
            'avg_health': avg_health,
            'min_health': min_health,
            'avg_stability': avg_stability,
            'avg_failure_prob': avg_failure_prob,
            'imminent_failures': imminent_count,
            'imminent_percent': 100 * imminent_count / total_devices,
            'upcoming_failures': upcoming_count,
            'upcoming_percent': 100 * upcoming_count / total_devices,
            'at_risk_count': at_risk_count,
            'at_risk_percent': 100 * at_risk_count / total_devices,
            'total_at_risk': imminent_count + upcoming_count + at_risk_count,
            'total_at_risk_percent': 100 * (imminent_count + upcoming_count + at_risk_count) / total_devices,
            'stuck_at_min': stuck_at_min,
            'stuck_at_max': stuck_at_max,
            'total_stuck': stuck_at_min + stuck_at_max,
            'stuck_percent': 100 * (stuck_at_min + stuck_at_max) / total_devices,
            'critical_count': (self.health_scores < 20).sum().item(),
            'total_operations': self.operation_counts.mean().item()
        }
    
    def visualize_health(self):
        """Visualize health metrics in a comprehensive dashboard."""
        # Update failure probabilities
        self.predict_faults()
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Health scores
        im1 = axes[0, 0].imshow(self.health_scores.numpy(), cmap='RdYlGn', vmin=0, vmax=100)
        axes[0, 0].set_title('Health Scores')
        fig.colorbar(im1, ax=axes[0, 0])
        
        # 2. Stability index
        im2 = axes[0, 1].imshow(self.stability_index.numpy(), cmap='coolwarm', vmin=0, vmax=1)
        axes[0, 1].set_title('Stability Index')
        fig.colorbar(im2, ax=axes[0, 1])
        
        # 3. Failure probability
        im3 = axes[1, 0].imshow(self.failure_probability.numpy(), cmap='plasma', vmin=0, vmax=1)
        axes[1, 0].set_title('Failure Probability')
        fig.colorbar(im3, ax=axes[1, 0])
        
        # 4. Current conductance
        g_normalized = ((self.crossbar.conductance_matrix - self.g_min) / self.g_range).numpy()
        im4 = axes[1, 1].imshow(g_normalized, cmap='viridis', vmin=0, vmax=1)
        axes[1, 1].set_title('Normalized Conductance')
        fig.colorbar(im4, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.show()
    
    def visualize_health_and_mapping(self):
        """
        Visualize the health and mapping of devices in the crossbar.
        """
        plt.figure(figsize=(12, 4))
        
        # 1. Show conductance matrix
        plt.subplot(1, 3, 1)
        g_normalized = (self.crossbar.conductance_matrix - self.g_min) / (self.g_max - self.g_min)
        g_normalized = torch.clamp(g_normalized, 0, 1)
        plt.imshow(g_normalized.numpy(), cmap='viridis')
        plt.colorbar(label='Normalized Conductance')
        plt.title('Conductance Values')
        
        # 2. Show remapped devices
        plt.subplot(1, 3, 2)
        remapping_mask = torch.zeros(self.effective_shape)
        
        # Mark original area
        for i in range(self.original_shape[0]):
            for j in range(self.original_shape[1]):
                if (i, j) in self.remapping:
                    physical_row, physical_col = self.remapping[(i, j)]
                    remapping_mask[physical_row, physical_col] = 2  # Remapped destination
                    remapping_mask[i, j] = 1  # Original position (remapped source)
                else:
                    remapping_mask[i, j] = 0.5  # Normal original device
        
        # Mark redundant area
        for i in range(self.effective_shape[0]):
            for j in range(self.effective_shape[1]):
                if self.is_redundant[i, j] and remapping_mask[i, j] == 0:
                    remapping_mask[i, j] = 0.25  # Unused redundant
        
        plt.imshow(remapping_mask.numpy(), cmap='coolwarm')
        plt.colorbar(label='Device Status')
        plt.title('Device Remapping')
        
        # 3. Show health estimate
        plt.subplot(1, 3, 3)
        # Estimate health based on how close conductances are to extremes
        g_normalized = (self.crossbar.conductance_matrix - self.g_min) / (self.g_max - self.g_min)
        g_normalized = torch.clamp(g_normalized, 0, 1)
        deviation = torch.abs(g_normalized - 0.5) * 2
        health_scores = 100 * (1.0 - deviation)
        
        plt.imshow(health_scores.numpy(), cmap='RdYlGn', vmin=0, vmax=100)
        plt.colorbar(label='Health Score')
        plt.title('Estimated Device Health')
        
        plt.tight_layout()
        plt.show()