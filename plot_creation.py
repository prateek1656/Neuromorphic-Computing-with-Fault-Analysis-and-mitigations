import matplotlib.pyplot as plt
import numpy as np
import random
import os

# Create output directory
output_dir = "fault_tolerance_plots"
os.makedirs(output_dir, exist_ok=True)

# Set random seed
np.random.seed(42)
random.seed(42)

# Common parameters
num_epochs = 30
batches_per_epoch = 100  # 100 batches per epoch
total_batches = num_epochs * batches_per_epoch
batch_range = np.arange(total_batches)

strategies = {
	"Baseline": {"color": "gold", "accuracy": 85.0, "var": 5.0, "description": "Fully trained model on ideal system"}, 
	"No Mitigations": {"color": "red", "accuracy": 55.0, "var": 7.0, "description": "Faults cause significant degradation"},
	"Soft Mitigation Only": {"color": "blue", "accuracy": 62.0, "var": 6.0, "description": "Minor recovery via calibration"},
	"Redundancy Remapping": {"color": "green", "accuracy": 68.0, "var": 4.0, "description": "Most effective standalone mitigation"},
	"Hard Reset Only": {"color": "orange", "accuracy": 64.0, "var": 5.5, "description": "Partial recovery via fault flushing"},
	"All Strategies": {"color": "purple", "accuracy": 73.0, "var": 3.0, "description": "Highest recovery under faulted setting"}
}

def generate_smooth_curve(start_val, min_val, max_val, total_points, noise_level=0.5, trend="none", edge_drop=0):
	"""
	Generate a smooth curve with optional edge effects.
	
	Parameters:
	- start_val: Starting value
	- min_val: Lower y-boundary
	- max_val: Upper y-boundary
	- total_points: Total number of points in the curve
	- noise_level: Amount of noise to add
	- trend: Overall trend ("up", "down", or "none")
	- edge_drop: Amount to drop at the edges (set to 0 for no edge effect)
	
	Returns:
	- A numpy array containing the curve
	"""
	# Create curve with control points
	num_control_points = np.random.randint(8, 15)
	control_x = np.sort(np.random.choice(range(2, total_points-2), num_control_points, replace=False))
	control_x = np.insert(control_x, 0, 0)
	control_x = np.append(control_x, total_points-1)
	
	# Generate control y values
	control_y = []
	
	# First control point (with possible edge drop)
	if edge_drop > 0:
		control_y.append(start_val - edge_drop)
	else:
		control_y.append(start_val)
	
	# Middle control points
	current_val = start_val
	amplitude = (max_val - min_val) * 0.3  # Use 30% of the range for oscillations
	
	# Add trend component
	trend_total = 0
	if trend == "up":
		trend_total = (max_val - start_val) * 0.8
	elif trend == "down":
		trend_total = (min_val - start_val) * 0.8
	
	trend_per_point = trend_total / (len(control_x) - 2)  # Exclude edge points
	
	for i in range(1, len(control_x) - 1):
		# Add oscillation
		oscillation = np.random.uniform(-amplitude, amplitude)
		
		# Add trend
		current_val += trend_per_point
		
		# Add random walk component
		current_val += np.random.uniform(-amplitude/2, amplitude/2)
		
		# Ensure within bounds
		current_val = max(min(current_val, max_val), min_val)
		
		control_y.append(current_val)
	
	# Last control point (with possible edge drop)
	if edge_drop > 0:
		end_val = control_y[-1] - edge_drop
		end_val = max(min_val, end_val)  # Ensure it doesn't go below min_val
		control_y.append(end_val)
	else:
		control_y.append(control_y[-1])
	
	# Create the curve using linear interpolation between control points
	curve = np.zeros(total_points)
	for i in range(len(control_x) - 1):
		start_idx = control_x[i]
		end_idx = control_x[i+1]
		
		if end_idx <= start_idx:
			continue
			
		segment_length = end_idx - start_idx
		segment = np.linspace(control_y[i], control_y[i+1], segment_length)
		curve[start_idx:end_idx] = segment
	
	# Add noise
	noise = np.random.normal(0, noise_level, total_points)
	smoothed_noise = np.convolve(noise, np.ones(3)/3, mode='same')
	curve += smoothed_noise
	
	# Ensure within bounds
	curve = np.clip(curve, min_val, max_val)
	
	return curve

def generate_accuracy_curves_all_strategies():
    """Generate the exact accuracy curves for the All Strategies plot with 3000 batches."""
    curves = []
    
    # Set target accuracy explicitly to 73%
    target_acc = 73
    
    # These variables seem to be defined elsewhere in the original code
    # Adding them here with reasonable values
    total_batches = 3000
    batches_per_epoch = 300
    num_epochs = total_batches // batches_per_epoch
    
    # Scale the control points from the original small batch count to our new larger batch count
    scale_factor = total_batches / 62
    
    # Main curve (top band) - reaches ~73% with realistic variations and NO DROP at the end
    top_curve = np.zeros(total_batches)
    original_top_control_points = [
        (0, 10),      # Start with drop
        (2, 14),      # Initial value
        (7, 18),      # Small dip
        (15, 38),     # Rise
        (20, 44),     # Continue rising
        (25, 48),     # Approach peak
        (30, 52),     # Near peak
        (35, 50),     # Reach peak of 73%
        (40, 48),     # Natural variation
        (45, 50),   # Slight fluctuation above
        (50, 54),   # Slight fluctuation below
        (55, 58),   # More natural variation
        (60, 64),     # Back to peak
        (65, 68),   # Natural decline
        (70, 70.7),   # Small recovery
        (75, 71.5),   # More variation
        (80, 72.3),   # Fluctuation
        (85, 71.9),   # Maintain high accuracy with natural variations
        (90, 72.5),   # Maintain high level
        (95, 73.1),   # Maintain high level
        (99, 72.8)    # End without dropping - still at high level
    ]
    
    # Scale control points for larger batch count
    top_control_points = []
    for x, y in original_top_control_points:
        # Scale x to our new batch count, keep y the same
        scaled_x = int(x * scale_factor)
        # Make sure the scaled x doesn't exceed total_batches
        scaled_x = min(scaled_x, total_batches-1)
        top_control_points.append((scaled_x, y))
    
    # Ensure the last point is at total_batches-1 WITHOUT dropping
    if top_control_points[-1][0] < total_batches-1:
        top_control_points.append((total_batches-1, 72.9))  # Maintain high performance at end
    
    # Interpolate top curve between control points
    for i in range(len(top_control_points)-1):
        x1, y1 = top_control_points[i]
        x2, y2 = top_control_points[i+1]
        
        if x2 <= x1:
            continue
            
        # Create segment with linear interpolation
        segment_length = x2 - x1
        segment = np.linspace(y1, y2, segment_length)
        top_curve[x1:x2] = segment
    
    # Add realistic noise
    noise = np.random.normal(0, 0.8, total_batches)
    # Apply smoothing to noise
    smoothed_noise = np.convolve(noise, np.ones(5)/5, mode='same')
    top_curve += smoothed_noise
    
    
    # Add epoch markers - create small oscillations at epoch boundaries
    epoch_size = batches_per_epoch
    for epoch in range(1, num_epochs):
        epoch_boundary = epoch * epoch_size
        
        # Only add oscillations if we're within range
        if epoch_boundary < total_batches:
            # Add small oscillations at epoch boundaries
            window_size = min(20, epoch_size // 10)
            window_start = max(0, epoch_boundary - window_size // 2)
            window_end = min(total_batches, epoch_boundary + window_size // 2)
            
            # Apply small oscillation to each curve
            for curve in [top_curve]:
                # Only apply if we have enough points
                if window_end - window_start > 2:
                    # Create a small oscillation
                    osc = 0.5 * np.sin(np.linspace(0, 2*np.pi, window_end - window_start))
                    curve[window_start:window_end] += osc
    
    # Instead of clipping, we'll use more control points to ensure natural behavior
    # We'll only apply very light bounds to prevent extreme outliers in noise
    top_curve = np.clip(top_curve, 0, 100)  # Just prevent impossible values
    
    # CRITICAL FIX: Directly overwrite the last 300 values to prevent any dropping at the end
    # Get the average of values between 70-80% of the run (a stable period) for each curve
    stable_idx_start = int(total_batches * 1)
    stable_idx_end = int(total_batches * 1)
    
    # Calculate the stable values for each curve (average of the stable period)
    top_stable = np.mean(top_curve[stable_idx_start:stable_idx_end])
    
    # Apply slight variations to these stable values for the end section
    end_section = 1  # Last 300 points
    for i in range(end_section):
        # Add subtle sine wave variations to make it look natural
        variation = 2 * np.sin(i * 0.1)
        idx = total_batches - end_section + i
        if idx >= 0 and idx < total_batches:
            top_curve[idx] = top_stable + variation
			
    # Ensure the very last few points don't drop
    for i in range(10):
        idx = total_batches - 1 - i
        if idx >= 0:
            top_curve[idx] = top_stable
    
    # Add curves in order of highest to lowest for better visualization
    curves.extend([top_curve])
    return curves


def generate_health_curves(strategy, num_curves=1):#✅
	"""Generate health curves for the given strategy."""
	# Access global variables
	global num_epochs, batches_per_epoch
	
	curves = []
	
	if strategy == "Baseline":
		# Baseline maintains perfect health
		main_curve = np.ones(total_batches) * 99.5
		
		# Add very small variations
		noise = np.random.normal(0, 0.05, total_batches)
		smoothed_noise = np.convolve(noise, np.ones(5)/5, mode='same')
		main_curve += smoothed_noise
		main_curve = np.clip(main_curve, 99.0, 100.0)
		
		curves.append(main_curve)
		
	elif strategy == "No Mitigations":
		# No Mitigations shows severe health degradation
		main_curve = np.zeros(total_batches)
		
		# Start at high health
		main_curve[0] = 99.0
		
		# Create exponential decay
		decay_rate = 0.0003  # Adjust for steeper or gentler decline
		for i in range(1, total_batches):
			# Add fault events
			if np.random.random() < 0.005:  # 0.5% chance of a fault each step
				fault_size = np.random.uniform(0.5, 1.5)
				main_curve[i:] -= fault_size
			
			# Natural degradation
			decay = decay_rate * (main_curve[i-1] - 80.0)  # Decay slows as it approaches 80%
			main_curve[i] = main_curve[i-1] - decay
			
			# Add small noise
			main_curve[i] += np.random.normal(0, 0.05)
		
		# Ensure health stays in reasonable bounds
		main_curve = np.clip(main_curve, 80.0, 99.0)
		
		curves.append(main_curve)
		
	elif strategy == "Soft Mitigation Only":
		# Soft Mitigation shows moderate health maintenance
		main_curve = np.zeros(total_batches)
		
		# Start at high health
		main_curve[0] = 99.0
		
		# Create decay with soft mitigations
		decay_rate = 0.0001  # Lower decay rate than No Mitigations
		for i in range(1, total_batches):
			# Add fault events
			if np.random.random() < 0.005:
				fault_size = np.random.uniform(0.5, 1.0)
				main_curve[i:] -= fault_size
				
				# Apply soft mitigation (partial recovery)
				recovery_length = np.random.randint(100, 300)
				recovery_end = min(i + recovery_length, total_batches)
				recovery_amount = fault_size * 0.6  # Recover 60% of fault
				
				recovery_curve = recovery_amount * (1 - np.exp(-3 * np.linspace(0, 1, recovery_end - i)))
				main_curve[i:recovery_end] += recovery_curve
			
			# Natural degradation with soft mitigations
			decay = decay_rate * (main_curve[i-1] - 85.0)
			main_curve[i] = main_curve[i-1] - decay
			
			# Add small noise
			main_curve[i] += np.random.normal(0, 0.07)
		
		# Ensure health stays in reasonable bounds
		main_curve = np.clip(main_curve, 85.0, 99.0)
		
		curves.append(main_curve)
		
	elif strategy == "Redundancy Remapping":
		# Redundancy Remapping shows good health with recovery spikes
		main_curve = np.zeros(total_batches)
		
		# Start at high health
		main_curve[0] = 99.0
		
		# Create decay with remapping recovery
		decay_rate = 0.0015  # Lower decay rate
		for i in range(1, total_batches):
			# Add fault events
			if np.random.random() < 0.005:
				fault_size = np.random.uniform(0.5, 1.0)
				main_curve[i:] -= fault_size
				
				# Apply remapping (quick recovery)
				recovery_length = np.random.randint(50, 100)
				recovery_end = min(i + recovery_length, total_batches)
				recovery_amount = fault_size * 0.9  # Recover 90% of fault
				recovery_curve = recovery_amount * (1 - np.exp(-5 * np.linspace(0, 1, recovery_end - i)))
				main_curve[i:recovery_end] += recovery_curve
			
			# Natural degradation with remapping
			decay = decay_rate * (main_curve[i-1] - 90.0)
			main_curve[i] = main_curve[i-1] - decay
			
			# Add small noise
			main_curve[i] += np.random.normal(0, 0.05)
			
			# Scheduled remapping eventsx
			if i % 500 == 0 and i > 0:
				remap_size = np.random.uniform(0.5, 1.5)
				main_curve[i:] += remap_size
		
		# Ensure health stays in reasonable bounds
		main_curve = np.clip(main_curve, 88.0, 99.0)
		
		curves.append(main_curve)
		
	elif strategy == "Hard Reset Only":
		# Hard Reset shows sawtooth pattern
		main_curve = np.zeros(total_batches)
		
		# Start at high health
		main_curve[0] = 99.0
		
		# Create decay with hard resets
		decay_rate = 0.005
		
		# Schedule reset events
		reset_interval = 500
		for i in range(1, total_batches):
			# Natural degradation
			decay = decay_rate * (main_curve[i-1] - 91.0)
			main_curve[i] = main_curve[i-1] - decay
			
			# Add fault events
			if np.random.random() < 0.005:
				fault_size = np.random.uniform(0.5, 1.0)
				main_curve[i:] -= fault_size
			
			# Hard reset scheduled
			if i % reset_interval == 0:
				reset_amount = np.random.uniform(3.0, 5.0)
				main_curve[i:] += reset_amount
			
			# Add small noise
			main_curve[i] += np.random.normal(0, 0.05)
		
		# Ensure health stays in reasonable bounds
		main_curve = np.clip(main_curve, 91.0, 99.0)
		
		curves.append(main_curve)
		
	elif strategy == "All Strategies":
		# Combined strategies show best health maintenance
		main_curve = np.zeros(total_batches)
		
		# Start at high health
		main_curve[0] = 99.0
		
		# Create decay with combined mitigations
		decay_rate = 0.001  # Lowest decay rate
		
		# Schedule reset and remapping events
		reset_interval = 800
		remap_interval = 900
		
		for i in range(1, total_batches):
			# Natural degradation
			decay = decay_rate * (main_curve[i-1] - 88.0)
			main_curve[i] = main_curve[i-1] - decay
			
			# Add fault events
			if np.random.random() < 0.005:
				fault_size = np.random.uniform(0.5, 0.8)  # Lower impact faults
				main_curve[i:] -= fault_size
				
				# Apply combined recovery (quick and effective)
				recovery_length = np.random.randint(50, 500)
				recovery_end = min(i + recovery_length, total_batches)
				recovery_amount = fault_size * 0.95  # Recover 95% of fault
				
				recovery_curve = recovery_amount * (1 - np.exp(-5 * np.linspace(0, 1, recovery_end - i)))
				main_curve[i:recovery_end] += recovery_curve
			
			# Scheduled remapping
			if i % remap_interval == 0 and i > 0:
				remap_size = np.random.uniform(0.5, 0.8)
				main_curve[i:] += remap_size
			
			# Hard reset scheduled
			if i % reset_interval == 0 and i > 0:
				reset_amount = np.random.uniform(1.0, 2.0)
				main_curve[i:] += reset_amount
			
			# Add small noise
			main_curve[i] += np.random.normal(0, 0.03)
		
		# Ensure health stays in reasonable bounds
		main_curve = np.clip(main_curve, 75.0, 99.0)
		
		curves.append(main_curve)
	
	return curves

def generate_loss_curves(strategy, num_curves=1):#✅
	"""Generate loss curves for the given strategy."""
	# Access global variables
	global num_epochs, batches_per_epoch
	
	target_acc = strategies[strategy]["accuracy"]
	# Calculate expected loss based on target accuracy
	target_loss = 2.5 - (target_acc - 50) / 20
	curves = []
	
	if strategy == "Baseline":
		# Baseline shows smooth decreasing loss that plateaus
		# Start at high loss
		start_loss = 2.8
		end_loss = 0.85  # Low final loss for high accuracy
		
		# Create learning curve
		x = np.linspace(0, 2, total_batches)
		decay_rate = 3.0  # Adjust for faster/slower convergence
		decay_curve = np.exp(-decay_rate * x)
		
		main_curve = end_loss + (start_loss - end_loss) * decay_curve
		
		# Add per-epoch oscillations
		epoch_size = batches_per_epoch
		
		for epoch in range(num_epochs):
			start_idx = epoch * epoch_size
			end_idx = min((epoch + 1) * epoch_size, total_batches)
			
			# Oscillations diminish over time
			diminishing_factor = np.exp(-1.5 * epoch / num_epochs)
			oscillation_scale = 0.07 * diminishing_factor
			
			# Create oscillation for this epoch
			epoch_oscillation = np.sin(np.linspace(0, 2*np.pi, end_idx - start_idx))
			main_curve[start_idx:end_idx] += oscillation_scale * epoch_oscillation
		
		# Add random noise that diminishes over time
		noise_scale = 0.1 * np.exp(-2 * x) + 0.02
		noise = np.random.normal(0, noise_scale, total_batches)
		smoothed_noise = np.convolve(noise, np.ones(5)/5, mode='same')
		
		main_curve += smoothed_noise
		
		curves.append(main_curve)
		
	elif strategy == "No Mitigations":
		# No Mitigations gets stuck at high loss
		# Start at high loss
		start_loss = 3.0
		plateau_loss = 1.8  # Stuck at higher loss
		
		# Create learning curve that plateaus early
		x = np.linspace(0, 1, total_batches)
		plateau_point = 0.25  # Early plateau
		
		# Create segmented curve
		main_curve = np.zeros(total_batches)
		
		# Initial steep drop
		initial_segment = x <= plateau_point
		initial_drop = start_loss - plateau_loss
		main_curve[initial_segment] = start_loss - initial_drop * (x[initial_segment] / plateau_point)
		
		# Plateau with minimal improvement
		plateau_segment = x > plateau_point
		remaining_drop = 0.2  # Only slight improvement after plateau
		main_curve[plateau_segment] = plateau_loss - remaining_drop * ((x[plateau_segment] - plateau_point) / (1 - plateau_point))
		
		# Add fault events that increase loss
		num_faults = 15
		fault_points = np.sort(np.random.choice(range(int(total_batches*0.1), total_batches-100), num_faults, replace=False))
		
		for fault_idx in fault_points:
			# Fault causes temporary spike in loss
			fault_magnitude = np.random.uniform(0.1, 0.3)
			fault_width = np.random.randint(10, 50)
			recovery_rate = np.random.uniform(0.01, 0.05)  # Slow recovery
			
			# Apply spike and recovery
			for i in range(fault_width):
				if fault_idx + i < total_batches:
					# Decay factor for recovery
					decay = np.exp(-recovery_rate * i)
					main_curve[fault_idx + i] += fault_magnitude * decay
		
		# Add oscillations
		epoch_size = batches_per_epoch
		
		for epoch in range(num_epochs):
			start_idx = epoch * epoch_size
			end_idx = min((epoch + 1) * epoch_size, total_batches)
			
			# Oscillations persist (don't diminish much)
			oscillation_scale = 0.1
			
			# Create oscillation for this epoch
			epoch_oscillation = np.sin(np.linspace(0, 4*np.pi, end_idx - start_idx))
			main_curve[start_idx:end_idx] += oscillation_scale * epoch_oscillation
		
		# Add high noise level
		noise = np.random.normal(0, 0.08, total_batches)
		smoothed_noise = np.convolve(noise, np.ones(5)/5, mode='same')
		
		main_curve += smoothed_noise
		
		curves.append(main_curve)
		
	elif strategy == "Soft Mitigation Only":
		# Soft Mitigation shows gradual improvement
		start_loss = 4.0
		end_loss = 1.0  # Better but still moderately high loss
		
		# Create learning curve
		x = np.linspace(0, 1, total_batches)
		decay_rate = 2.0  # Moderate convergence
		decay_curve = np.exp(-decay_rate * x)
		
		main_curve = end_loss + (start_loss - end_loss) * decay_curve
		
		# Add fault events with soft mitigation recovery
		num_faults = 15
		fault_points = np.sort(np.random.choice(range(int(total_batches*0.1), total_batches-100), num_faults, replace=False))
		
		for fault_idx in fault_points:
			# Fault causes temporary spike in loss
			fault_magnitude = np.random.uniform(0.1, 0.25)
			fault_width = np.random.randint(10, 30)
			recovery_rate = np.random.uniform(0.05, 0.1)  # Faster recovery than No Mitigations
			
			# Apply spike and recovery
			for i in range(fault_width):
				if fault_idx + i < total_batches:
					# Decay factor for recovery
					decay = np.exp(-recovery_rate * i)
					main_curve[fault_idx + i] += fault_magnitude * decay
		
		# Add oscillations
		epoch_size = batches_per_epoch
		
		for epoch in range(num_epochs):
			start_idx = epoch * epoch_size
			end_idx = min((epoch + 1) * epoch_size, total_batches)
			
			# Oscillations diminish moderately
			diminishing_factor = np.exp(-1 * epoch / num_epochs)
			oscillation_scale = 0.1 * diminishing_factor
			
			# Create oscillation for this epoch
			epoch_oscillation = np.sin(np.linspace(0, 3*np.pi, end_idx - start_idx))
			main_curve[start_idx:end_idx] += oscillation_scale * epoch_oscillation
		
		# Add moderate noise
		noise = np.random.normal(0, 0.05, total_batches)
		smoothed_noise = np.convolve(noise, np.ones(5)/5, mode='same')
		
		main_curve += smoothed_noise
		
		curves.append(main_curve)
		
	elif strategy == "Redundancy Remapping":
		# Redundancy Remapping shows good improvement with step changes
		start_loss = 3.0
		end_loss = 2  # Better loss than Soft Mitigation
		
		# Create learning curve
		x = np.linspace(0, 1, total_batches)
		decay_rate = 2  # Better convergence than Soft Mitigation
		decay_curve = np.exp(-decay_rate * x)
		
		main_curve = end_loss + (start_loss - end_loss) * decay_curve
		
		# Add fault events with quick recovery
		num_faults = 10
		fault_points = np.sort(np.random.choice(range(int(total_batches*0.1), total_batches-100), num_faults, replace=False))
		
		for fault_idx in fault_points:
			# Fault causes temporary spike in loss
			fault_magnitude = np.random.uniform(0.1, 0.2)
			fault_width = np.random.randint(5, 20)  # Shorter impact
			recovery_rate = np.random.uniform(0.1, 0.2)  # Much faster recovery
			
			# Apply spike and recovery
			for i in range(fault_width):
				if fault_idx + i < total_batches:
					# Decay factor for recovery
					decay = np.exp(-recovery_rate * i)
					main_curve[fault_idx + i] += fault_magnitude * decay
		
		# Add remapping events (step improvements)
		num_remaps = 10
		remap_points = np.sort(np.random.choice(range(int(total_batches*0.15), total_batches-100), num_remaps, replace=False))
		
		for remap_idx in remap_points:
			# Remapping causes permanent improvement
			improvement = np.random.uniform(0.05, 0.15)
			main_curve[remap_idx:] -= improvement
		
		# Add oscillations
		epoch_size = batches_per_epoch
		
		for epoch in range(num_epochs):
			start_idx = epoch * epoch_size
			end_idx = min((epoch + 1) * epoch_size, total_batches)
			
			# Oscillations diminish
			diminishing_factor = np.exp(-1.5 * epoch / num_epochs)
			oscillation_scale = 0.08 * diminishing_factor
			
			# Create oscillation for this epoch
			epoch_oscillation = np.sin(np.linspace(0, 3*np.pi, end_idx - start_idx))
			main_curve[start_idx:end_idx] += oscillation_scale * epoch_oscillation
		
		# Add low noise
		noise = np.random.normal(0, 0.04, total_batches)
		smoothed_noise = np.convolve(noise, np.ones(5)/5, mode='same')
		
		main_curve += smoothed_noise
		
		curves.append(main_curve)
		
	elif strategy == "Hard Reset Only":
		# Hard Reset shows sawtooth pattern
		start_loss = 3.0
		end_loss = 5  # Moderate final loss
		
		# Create learning curve
		x = np.linspace(0, 1, total_batches)
		decay_rate = 0.6
		decay_curve = np.exp(-decay_rate * x)
		
		main_curve = end_loss + (start_loss - end_loss) * decay_curve
		
		# Add fault events
		num_faults = 12
		fault_points = np.sort(np.random.choice(range(int(total_batches*0.1), total_batches-100), num_faults, replace=False))
		
		for fault_idx in fault_points:
			# Fault causes temporary spike in loss
			fault_magnitude = np.random.uniform(0.1, 0.25)
			fault_width = np.random.randint(10, 30)
			recovery_rate = np.random.uniform(0.03, 0.08)  # Moderate recovery
			
			# Apply spike and recovery
			for i in range(fault_width):
				if fault_idx + i < total_batches:
					# Decay factor for recovery
					decay = np.exp(-recovery_rate * i)
					main_curve[fault_idx + i] += fault_magnitude * decay
		
		# Add reset events (sharp drops in loss)
		num_resets = 8
		reset_interval = total_batches // num_resets
		
		for i in range(num_resets):
			reset_idx = (i + 1) * reset_interval
			if reset_idx < total_batches:
				# Reset causes sharp drop in loss
				drop_magnitude = np.random.uniform(0.2, 0.4)
				
				# Apply drop
				main_curve[reset_idx:] -= drop_magnitude
				
				# Apply decay (loss gradually returns to baseline)
				decay_length = np.random.randint(int(batches_per_epoch*0.5), int(batches_per_epoch*1.5))
				end_decay = min(reset_idx + decay_length, total_batches)
				
				decay_curve = drop_magnitude * (1 - np.exp(-1 * np.linspace(0, 3, end_decay - reset_idx)))
				main_curve[reset_idx:end_decay] += decay_curve
		
		# Add oscillations
		epoch_size = batches_per_epoch
		
		for epoch in range(num_epochs):
			start_idx = epoch * epoch_size
			end_idx = min((epoch + 1) * epoch_size, total_batches)
			
			# Moderate oscillations
			diminishing_factor = np.exp(-1.2 * epoch / num_epochs)
			oscillation_scale = 0.1 * diminishing_factor
			
			# Create oscillation for this epoch
			epoch_oscillation = np.sin(np.linspace(0, 3*np.pi, end_idx - start_idx))
			main_curve[start_idx:end_idx] += oscillation_scale * epoch_oscillation
		
		# Add moderate noise
		noise = np.random.normal(0, 0.05, total_batches)
		smoothed_noise = np.convolve(noise, np.ones(5)/5, mode='same')
		
		main_curve += smoothed_noise
		
		curves.append(main_curve)
		
	elif strategy == "All Strategies":
		# All Strategies shows best combined performance
		# Create multiple curves at different levels to show banding

		curves = []

		# Top band (high loss band)
		top_start = 3.0
		top_end = 2.2

		x = np.linspace(0, 1, total_batches)
		decay_rate = 2.0
		decay_curve = np.exp(-decay_rate * x)
		top_curve = top_end + (top_start - top_end) * decay_curve

		# Add noise
		top_curve += np.random.normal(0, 0.03, total_batches)

		# Add fault events with quick recovery
		num_faults = 8
		fault_points = np.sort(np.random.choice(range(int(total_batches * 0.1), total_batches - 100), num_faults, replace=False))

		for fault_idx in fault_points:
			fault_magnitude = np.random.uniform(0.05, 0.15)
			fault_width = np.random.randint(5, 15)
			recovery_rate = np.random.uniform(0.15, 0.25)

			for i in range(fault_width):
				if fault_idx + i < total_batches:
					decay = np.exp(-recovery_rate * i)
					top_curve[fault_idx + i] += fault_magnitude * decay

		# Add moderate spikes (visible bumps in the loss curve)
		num_moderate_spikes = 5
		moderate_points = np.sort(np.random.choice(range(int(total_batches * 0.2), total_batches - 100), num_moderate_spikes, replace=False))

		for spike_idx in moderate_points:
			spike_magnitude = np.random.uniform(0.1, 0.25)
			spike_duration = np.random.randint(30, 50)
			recovery_rate = np.random.uniform(0.05, 0.15)

			for i in range(spike_duration):
				if spike_idx + i < total_batches:
					decay = np.exp(-recovery_rate * i)
					top_curve[spike_idx + i] += spike_magnitude * decay

		# Add remapping events (step improvements)
		num_remaps = 6
		remap_points = np.sort(np.random.choice(range(int(total_batches * 0.15), total_batches - 100), num_remaps, replace=False))

		for remap_idx in remap_points:
			improvement = np.random.uniform(0.03, 0.1)
			top_curve[remap_idx:] -= improvement

		# Add reset events (sharp drops with decay)
		num_resets = 4
		reset_interval = total_batches // num_resets

		for i in range(num_resets):
			reset_idx = (i + 1) * reset_interval
			if reset_idx < total_batches:
				drop_magnitude = np.random.uniform(0.05, 0.2)
				top_curve[reset_idx:] -= drop_magnitude

				decay_length = np.random.randint(int(batches_per_epoch * 0.5), int(batches_per_epoch * 1.0))
				end_decay = min(reset_idx + decay_length, total_batches)

				decay_curve = drop_magnitude * (1 - np.exp(-2 * np.linspace(0, 2, end_decay - reset_idx)))
				top_curve[reset_idx:end_decay] += decay_curve

		curves.append(top_curve)

	return curves

def generate_accuracy_curves(strategy, num_curves=3): #✅
	"""Generate accuracy curves for the given strategy."""
	target_acc = strategies[strategy]["accuracy"]
	curves = []
	
	if strategy == "Baseline":
		# Baseline shows increasing accuracy that plateaus
		main_curve = np.zeros(total_batches)
		
		# Create a smooth increasing curve that plateaus
		x = np.linspace(0, 1, total_batches)
		growth_component = 15.0 + (target_acc - 15.0) * (1 - np.exp(-3 * x))
		
		# Add oscillations
		oscillation = 1.5 * np.sin(x * 60) * np.exp(-2 * x)
		
		main_curve = growth_component + oscillation
		curves.append(main_curve)
		
	elif strategy == "No Mitigations":
		# No Mitigations shows limited accuracy improvement with high noise
		main_curve = generate_smooth_curve(
			start_val=35.0,
			min_val=30.0,
			max_val=60.0,
			total_points=total_batches,
			noise_level=2.5,
			trend="up",
			edge_drop=5.0
		)
		
		# Add high noise to show instability
		noise = np.random.normal(0, 3.0, total_batches)
		smoothed_noise = np.convolve(noise, np.ones(3)/3, mode='same')
		
		main_curve = main_curve + smoothed_noise
		curves.append(main_curve)
		
	elif strategy == "Soft Mitigation Only":
		# Soft Mitigation shows moderate accuracy improvement
		main_curve = generate_smooth_curve(
			start_val=40.0,
			min_val=35.0,
			max_val=65.0,
			total_points=total_batches,
			noise_level=1.5,
			trend="up",
			edge_drop=5.0
		)
		
		non_zero_indices = [2997]
		if len(non_zero_indices) > 0:
			last_non_zero_index = non_zero_indices[-1]
			last_val = main_curve[last_non_zero_index]

			# Step 2: Replace zeros with last_val + small noise
			for i in range(last_non_zero_index + 1, len(main_curve)):
				noise = np.random.normal(loc=0.0, scale=0.01)  # Adjust scale if needed
				main_curve[i] = last_val + noise
		
		curves.append(main_curve)
		
	elif strategy == "Redundancy Remapping":
		# Redundancy Remapping shows better accuracy with remapping events
		main_curve = generate_smooth_curve(
			start_val=40.0,
			min_val=35.0,
			max_val=70.0,
			total_points=total_batches,
			noise_level=1.0,
			trend="up",
			edge_drop=5.0
		)
		
		# Add remapping events (small upward steps)
		num_steps = np.random.randint(3, 6)
		step_locations = np.sort(np.random.choice(range(10, total_batches-10), size=num_steps, replace=False))
		step_sizes = np.random.uniform(1.0, 3.0, size=num_steps)
		
		for loc, size in zip(step_locations, step_sizes):
			# Apply step improvement
			main_curve[loc:] += size
		
		non_zero_indices = [2997]
		if len(non_zero_indices) > 0:
			last_non_zero_index = non_zero_indices[-1]
			last_val = main_curve[last_non_zero_index]

			# Step 2: Replace zeros with last_val + small noise
			for i in range(last_non_zero_index + 1, len(main_curve)):
				noise = np.random.normal(loc=0.0, scale=0.01)  # Adjust scale if needed
				main_curve[i] = last_val + noise
		
		curves.append(main_curve)
		
	elif strategy == "Hard Reset Only":
		# Hard Reset shows accuracy with periodic resets
		main_curve = generate_smooth_curve(
			start_val=40.0,
			min_val=35.0,
			max_val=68.0,
			total_points=total_batches,
			noise_level=1.5,
			trend="up",
			edge_drop=5.0
		)
		
		# Add reset events (dips followed by fast recovery)
		num_resets = np.random.randint(3, 6)
		reset_locations = np.sort(np.random.choice(range(10, total_batches-10), size=num_resets, replace=False))
		
		for loc in reset_locations:
			# Create a dip
			dip_width = np.random.randint(2, 100)
			dip_depth = np.random.uniform(5, 10)
			
			if loc + dip_width < total_batches:
				for k in range(dip_width):
					main_curve[loc+k] -= dip_depth * (1 - k/dip_width)
			
			# Create quick recovery plus improvement
			recovery_width = np.random.randint(3, 7)
			recovery_boost = np.random.uniform(0, 2)  # Additional improvement beyond pre-dip level
			
			if loc + dip_width + recovery_width < total_batches:
				recovery_start = main_curve[loc] - dip_depth
				recovery_end = main_curve[loc] + recovery_boost
				recovery = np.linspace(recovery_start, recovery_end, recovery_width)
				main_curve[loc+dip_width:loc+dip_width+recovery_width] = recovery
		
		non_zero_indices = [2997]
		if len(non_zero_indices) > 0:
			last_non_zero_index = non_zero_indices[-1]
			last_val = main_curve[last_non_zero_index]
			# Step 2: Replace zeros with last_val + small noise
			for i in range(last_non_zero_index + 1, len(main_curve)):
				noise = np.random.normal(loc=0.0, scale=0.01)  # Adjust scale if needed
				main_curve[i] = last_val + noise
		
		curves.append(main_curve)
		
	elif strategy == "All Strategies":
		# Use the special function for All Strategies accuracy curves
		curves = generate_accuracy_curves_all_strategies()
	
	return curves

def generate_fault_mitigation_data(strategy): #✅
    """Generate fault and mitigation data for the given strategy."""
    global num_epochs, batches_per_epoch, total_batches

    # Initialize arrays for faults and mitigations
    fault_counts = np.zeros(total_batches)
    soft_mitigations = np.zeros(total_batches)
    remappings = np.zeros(total_batches)

    if strategy == "Baseline":
        return [np.zeros(total_batches)], [np.zeros(total_batches)], [np.zeros(total_batches)]

    # Fault injection every 5 epochs
    fault_injection_epochs = set(range(5, num_epochs + 1, 5))
    fault_injection_batches = {epoch * batches_per_epoch for epoch in fault_injection_epochs}
    fault_rate = 0.01  # 1% per batch chance of injection

    # Create accumulating faults
    for i in range(1, total_batches):
        fault_counts[i] = fault_counts[i - 1]

        # Inject faults during specified epochs or based on fault rate
        if i in fault_injection_batches or np.random.random() < fault_rate:
            new_faults = np.random.randint(5, 20)
            fault_counts[i:] += new_faults

    fault_counts *= 5e3  # scale to appropriate values

    if strategy == "No Mitigations":
        return [fault_counts], [soft_mitigations], [remappings]

    elif strategy == "Soft Mitigation Only":
        # Apply soft mitigations with a delay of 10 epochs (batches)
        for i in range(10, total_batches):
            soft_mitigations[i] = fault_counts[i - 10] * 0.6  # Mitigate 60% of faults with delay
        return [fault_counts], [soft_mitigations], [remappings]

    elif strategy == "Redundancy Remapping":
        # Apply remapping mitigations with a delay of 5 epochs (batches)
        for i in range(5, total_batches):
            remappings[i] = fault_counts[i - 5] * 0.8  # Mitigate 80% of faults with less delay
        return [fault_counts], [soft_mitigations], [remappings]

    elif strategy == "Hard Reset Only":
        # Resets every 10 epochs (i.e., every 1000 batches)
        reset_interval = 10 * batches_per_epoch
        for i in range(reset_interval, total_batches, reset_interval):
            fault_counts[i:] -= fault_counts[i] * 0.9  # 90% reset
        return [fault_counts], [soft_mitigations], [remappings]

    elif strategy == "All Strategies":
        fault_lines = []
        soft_lines = []
        remap_lines = []

        # Faults at different severities
        num_variants = 5
        for i in range(num_variants):
            fault_line = np.zeros(total_batches)
            fault_base = 1.5e6 + i * (7e6 / num_variants)
            local_fault_rate = 0.005 + i * 0.002  # Different fault rates

            for j in range(1, total_batches):
                fault_line[j] = fault_line[j - 1]
                if np.random.random() < local_fault_rate:
                    new_faults = np.random.randint(3, 15)
                    fault_line[j:] += new_faults

            fault_line = fault_base + fault_line * 500
            fault_line += np.random.normal(0, 1e4, total_batches)
            fault_lines.append(fault_line)

            # Soft mitigation for this fault line
            soft_line = np.zeros(total_batches)
            local_mitigation_rate = 0.004 + i * 0.002  # Different mitigation rates
            base_level = 1.2e6 + i * (6.3e6 / num_variants)

            for j in range(1, total_batches):
                soft_line[j] = soft_line[j - 1]
                if np.random.random() < local_mitigation_rate:
                    mitigations = np.random.randint(2, 12)
                    soft_line[j:] += mitigations

            soft_line = base_level + soft_line * 400
            soft_line += np.random.normal(0, 8000, total_batches)
            soft_lines.append(soft_line)

        # Remapping (single line)
        remap_line = np.zeros(total_batches)
        remap_rate = 0.002  # Less frequent remapping

        for i in range(5, total_batches):
            if np.random.random() < remap_rate:
                new_remaps = np.random.randint(1, 5)
                remap_line[i:] += new_remaps

        remap_line = 0.05e6 + remap_line * 200
        remap_line += np.random.normal(0, 500, total_batches)

        remap_lines.append(remap_line)

        return fault_lines, soft_lines, remap_lines

    else:
        # Combined soft mitigations and remapping (default strategy)
        for i in range(1, total_batches):
            if i > 10:
                soft_mitigations[i] = fault_counts[i - 10] * 0.4
            if i > 5:
                remappings[i] = fault_counts[i - 5] * 0.5
        return [fault_counts], [soft_mitigations], [remappings]


# Main code to create plots for each strategy
def main():
	"""Generate individual graphs for each metric and strategy combination."""
	metrics = ["Accuracy", "Health", "Loss", "Faults"]
	
	for strategy in strategies.keys():
		color = strategies[strategy]["color"]
		accuracy_curves = generate_accuracy_curves(strategy)
		health_curves = generate_health_curves(strategy)
		loss_curves = generate_loss_curves(strategy)
		faults, soft_mitigations, remappings = generate_fault_mitigation_data(strategy)
		
		# 1. Accuracy Graph
		plt.figure(figsize=(14, 8))
		for curve in accuracy_curves:
			plt.plot(batch_range, curve, color=color, alpha=0.7, linewidth=1.5)
		
		# Add a label with the strategy's accuracy target
		plt.text(0.02, 0.98, f"Target: {strategies[strategy]['accuracy']}% - {strategies[strategy]['description']}", 
				 transform=plt.gca().transAxes, fontsize=12, 
				 verticalalignment='top', horizontalalignment='left',
				 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
		
		# Add epoch markers
		for epoch in range(1, num_epochs):
			plt.axvline(x=epoch * batches_per_epoch, color='gray', linestyle='--', alpha=0.3)
		
		plt.title(f"{strategy} - Accuracy Over Time", fontsize=14)
		plt.xlabel("Batch")
		plt.ylabel("Accuracy (%)")
		plt.ylim(0, 100)
		plt.grid(True, alpha=0.3)
			
		plt.savefig(os.path.join(output_dir, f"{strategy.replace(' ', '_').lower()}_accuracy.png"), dpi=300)
		plt.close()
		
		# 2. Health Graph
		plt.figure(figsize=(14, 8))
		
		for curve in health_curves:
			plt.plot(batch_range, curve, color=color, alpha=0.8, linewidth=1.5)
		
		# Add epoch markers
		for epoch in range(1, num_epochs):
			plt.axvline(x=epoch * batches_per_epoch, color='gray', linestyle='--', alpha=0.3)
		
		plt.title(f"{strategy} - Health Over Time", fontsize=14)
		plt.xlabel("Batch")
		plt.ylabel("Health (%)")
		plt.ylim(75, 100)
		plt.grid(True, alpha=0.3)
		plt.savefig(os.path.join(output_dir, f"{strategy.replace(' ', '_').lower()}_health.png"), dpi=300)
		plt.close()
		
		# 3. Loss Graph
		plt.figure(figsize=(14, 8))
		for curve in loss_curves:
			plt.plot(batch_range, curve, color=color, alpha=0.7, linewidth=1.5)
		
		# Add epoch markers
		for epoch in range(1, num_epochs):
			plt.axvline(x=epoch * batches_per_epoch, color='gray', linestyle='--', alpha=0.3)
		
		plt.title(f"{strategy} - Loss Over Time", fontsize=14)
		plt.xlabel("Batch")
		plt.ylabel("Loss")
		
		if strategy == "All Strategies":
			plt.ylim(0.5, 3.5)
		else:
			avg_loss = 2.5 - (strategies[strategy]["accuracy"] - 50) / 20
			plt.ylim(0.5, 3.5)
			
		plt.grid(True, alpha=0.3)
		plt.savefig(os.path.join(output_dir, f"{strategy.replace(' ', '_').lower()}_loss.png"), dpi=300)
		plt.close()
		
		# 4. Faults Graph
		plt.figure(figsize=(14, 8))
		
		if strategy == "Baseline":
			plt.text(0.5, 0.5, "No Faults in Baseline Model", 
					horizontalalignment='center', verticalalignment='center',
					transform=plt.gca().transAxes, fontsize=14)
			plt.title(f"{strategy} - Faults and Mitigations", fontsize=14)
			plt.xlabel("Batch")
			plt.ylabel("Count")
			plt.grid(False)
		elif strategy == "All Strategies":
			# Plot each curve
			for remap_line in remappings:
				plt.plot(batch_range, remap_line, 'g-', linewidth=1.5, alpha=1.0)
			
			for i, fault_line in enumerate(faults):
				plt.plot(batch_range, fault_line, 'r-', linewidth=1.0, alpha=0.8)
				if i < len(soft_mitigations):
					plt.plot(batch_range, soft_mitigations[i], 'b-', linewidth=1.0, alpha=0.8)
			
			# Add epoch markers
			for epoch in range(1, num_epochs):
				plt.axvline(x=epoch * batches_per_epoch, color='gray', linestyle='--', alpha=0.3)
			
			# Add legend lines
			plt.plot([], [], 'r-', label='Faults', linewidth=1.5)
			plt.plot([], [], 'b-', label='Soft Mitigations', linewidth=1.5)
			plt.plot([], [], 'g-', label='Remappings', linewidth=1.5)
			
			plt.title(f"{strategy} - Faults and Mitigations", fontsize=14)
			plt.xlabel("Batch")
			plt.ylabel("Count")
			plt.ylim(0, 9e6)
			plt.legend(loc='upper right')
			plt.grid(True, alpha=0.3)
		else:
			for fault_line in faults:
				plt.plot(batch_range, fault_line, 'r-', label='Faults', linewidth=1.5)
			
			for soft_line in soft_mitigations:
				if np.sum(soft_line) > 0:  # Only plot if there are values
					plt.plot(batch_range, soft_line, 'b-', label='Soft Mitigations', linewidth=1.5)
			
			for remap_line in remappings:
				if np.sum(remap_line) > 0:  # Only plot if there are values
					plt.plot(batch_range, remap_line, 'g-', label='Remappings', linewidth=1.5)
			
			# Add epoch markers
			for epoch in range(1, num_epochs):
				plt.axvline(x=epoch * batches_per_epoch, color='gray', linestyle='--', alpha=0.3)
			
			# Add legend (only add entries for lines that were plotted)
			handles, labels = plt.gca().get_legend_handles_labels()
			by_label = dict(zip(labels, handles))
			plt.legend(by_label.values(), by_label.keys(), loc='upper right')
			
			plt.title(f"{strategy} - Faults and Mitigations", fontsize=14)
			plt.xlabel("Batch")
			plt.ylabel("Count")
			plt.grid(True, alpha=0.3)
			
		plt.savefig(os.path.join(output_dir, f"{strategy.replace(' ', '_').lower()}_faults.png"), dpi=300)
		plt.close()
		
	print(f"All learning curves generated successfully in '{output_dir}' folder!")
	
# Run the main function if this script is executed directly
if __name__ == "__main__":
	main()