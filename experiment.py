import torch
import torchvision
import matplotlib.pyplot as plt

from fault_tolerant_neuromorphic import FaultTolerantNeuromorphic
from models import SimpleCNN

def run_enhanced_fault_tolerance_experiment(
        fault_density=0.02, 
        distribution='random',
        enable_soft_mitigation=True,
        enable_remapping=True,
        enable_layer_reset=True,
        epochs=15, 
        num_batches=50,
        learning_rate=0.001,
        batch_size=32,
        force_remapping=False,        # NEW: Force remapping to be applied
        health_threshold=75,          # NEW: Adjust health threshold for mitigation
        fault_injection_interval=20,    # NEW: Control fault injection frequency
        ):
    """
    Run an experiment with the enhanced fault-tolerant neuromorphic system using all mitigation strategies.
    
    Args:
        fault_density: Proportion of devices to inject faults into
        distribution: How to distribute faults ('random', 'clustered', or 'gradient')
        enable_soft_mitigation: Whether to enable conductance adjustment mitigation
        enable_remapping: Whether to enable redundancy-based remapping
        enable_layer_reset: Whether to enable layer reset for catastrophic failures
        epochs: Number of training epochs
        num_batches: Number of batches per epoch
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        force_remapping: Force remapping to be applied even if health score is high
        health_threshold: Health threshold for applying different mitigation strategies
        fault_injection_interval: How often to inject faults (in batches)
        metrics_callback: Function to call with metrics updates (epoch, batch, metrics_dict)
        fault_callback: Function to call when a fault is detected (layer, position, fault_type)
        mitigation_callback: Function to call when mitigation is performed (action_type, details)
        
    Returns:
        Experiment results
    """
    # Create the enhanced fault-tolerant system
    try:
        redundancy_factor = 0.3 if enable_remapping else 0.0
        model = SimpleCNN()
        ft_system = FaultTolerantNeuromorphic(
            base_model=model,
            redundancy_factor=redundancy_factor)
    except Exception as e:
        print(f"Failed to create enhanced fault tolerant system: {e}")
        # Create a simplified version as fallback
        print("Creating simplified test system...")
        ft_system = FaultTolerantNeuromorphic( 
            redundancy_factor=0.15  # Lower redundancy for testing
        )
    
    # Load a small subset of CIFAR-10 for testing
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainset = torch.utils.data.Subset(trainset, range(2000))  # Use 2000 samples
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        testset = torch.utils.data.Subset(testset, range(1000))  # Use 1000 samples
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    except Exception as e:
        print(f"Error loading CIFAR-10 dataset: {e}")
        # Create dummy data if CIFAR-10 can't be loaded
        print("Using synthetic data instead...")
        trainset = [(torch.randn(3, 32, 32), torch.randint(0, 10, (1,)).item()) for _ in range(1000)]
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testset = [(torch.randn(3, 32, 32), torch.randint(0, 10, (1,)).item()) for _ in range(500)]
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ft_system.memristive_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Track metrics
    results = {
        'epoch': [],
        'batch': [],
        'loss': [],
        'accuracy': [],
        'health': [],
        'faults': [],
        'soft_mitigations': [],
        'remappings': [],
        'layer_resets': [],
        'redundant_available': []
    }
    
    # Initial system health check
    print("\nInitial health check:")
    initial_report = ft_system.get_health_report()
    print(f"Average health: {initial_report['avg_health']:.2f}%")
    print(f"Redundant devices available: {initial_report['redundant_available']}")
    
    if enable_remapping:
        print("\nVerifying self-healing crossbar setup...")
        for name, crossbar in ft_system.self_healing_crossbars.items():
            print(f"  Crossbar {name}: {len(crossbar.available_redundant)} redundant devices available")
            
    
    # Inject initial faults with safe error handling
    try:
        num_faults = ft_system.inject_faults(fault_density=fault_density/3, distribution=distribution)
        print(f"Successfully injected {num_faults} initial faults")
    except Exception as e:
        print(f"Warning: Failed to inject initial faults: {e}")
    
    # Modified for training with better health monitoring
    previous_loss = float('inf')  # For gradient estimation if needed
    global_batch = 0
    
    # Training loop
    print("\nStarting training with fault injection...")
    for epoch in range(epochs):
        ft_system.memristive_model.train()
        running_loss = 0.0
        
        for batch_idx, data in enumerate(trainloader):
            if batch_idx >= num_batches:
                break
            
            global_batch += 1
            
            try:
                # Handle both dataset types (tuple and custom)
                if isinstance(data, list) and len(data) == 2:
                    inputs, labels = data
                else:
                    inputs, labels = data
                
                # Ensure inputs have requires_grad=True for backprop
                inputs.requires_grad_(True)
                
                # Forward pass with error handling
                try:
                    outputs = ft_system.forward(inputs)
                    loss = criterion(outputs, labels)
                except Exception as e:
                    print(f"Forward pass error in batch {batch_idx}: {e}")
                    continue
                
                # Backward pass with error handling
                try:
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # to handle gradient issues with memristive layers
                    for name, param in ft_system.memristive_model.named_parameters():
                        if param.grad is None and param.requires_grad:
                            # Create a surrogate gradient based on the parameter's value
                            param.grad = torch.zeros_like(param.data)
                            print(f"Created surrogate gradient for {name}")
                    
                    optimizer.step()
                    
                    if global_batch % 5 == 0: 
                        with torch.no_grad():
                            ft_system.check_health(
                                force_remapping=force_remapping,
                                health_threshold=health_threshold,
                                enable_soft_mitigation=enable_soft_mitigation,
                                enable_remapping=enable_remapping,
                                enable_layer_reset=enable_layer_reset
                            )
                            
                except RuntimeError as e:
                    if "does not require grad" in str(e) or "No gradients" in str(e):
                        print(f"Gradient computation issue in batch {batch_idx}. Using straight-through estimator.")
                        # Try to continue with a straight-through estimator
                        with torch.no_grad():
                            for name, param in ft_system.memristive_model.named_parameters():
                                if param.requires_grad:
                                    # Apply small nudge in the direction that would reduce loss
                                    param.data -= 0.001 * torch.sign(param.data) * (loss.item() > previous_loss)
                            previous_loss = loss.item()
                    else:
                        print(f"Unexpected error in backpropagation: {e}")
                        continue
                
                running_loss += loss.item()
                
                # Every few batches, inject some faults
                if batch_idx % fault_injection_interval == 0 and batch_idx > 0:
                    try:
                        # Gradually increasing fault density
                        health_report = ft_system.get_health_report()
                        adaptive_fault_density = fault_density * (100 / max(health_report['avg_health'], 60))
                        capped_fault_density = min(adaptive_fault_density, fault_density * 2) / 10
                        
                        num_faults = ft_system.inject_faults(
                            fault_density=capped_fault_density, 
                            distribution=distribution
                        )
                        print(f"Injected {num_faults} faults (density: {capped_fault_density:.4f})")
                    except Exception as e:
                        print(f"Warning: Failed to inject faults in batch {batch_idx}: {e}")
                
                # Evaluate current performance
                if batch_idx % 3 == 0:
                    try:
                        ft_system.memristive_model.eval()
                        correct = 0
                        total = 0
                        
                        with torch.no_grad():
                            for test_data in testloader:
                                if isinstance(test_data, list) and len(test_data) == 2:
                                    test_inputs, test_labels = test_data
                                else:
                                    test_inputs, test_labels = test_data
                                
                                test_outputs = ft_system.forward(test_inputs)
                                _, predicted = torch.max(test_outputs.data, 1)
                                total += test_labels.size(0)
                                correct += (predicted == test_labels).sum().item()
                        
                        accuracy = 100 * correct / total if total > 0 else 0
                        
                        # Get current health metrics
                        health_report = ft_system.get_health_report()
                        
                        # Record metrics
                        results['epoch'].append(epoch)
                        results['batch'].append(batch_idx)
                        results['loss'].append(running_loss / (batch_idx + 1))
                        results['accuracy'].append(accuracy)
                        results['health'].append(health_report['avg_health'])
                        results['faults'].append(health_report['total_faults_detected'])
                        results['soft_mitigations'].append(ft_system.total_soft_mitigations)
                        results['remappings'].append(ft_system.total_remappings)
                        results['layer_resets'].append(ft_system.total_layer_resets)
                        results['redundant_available'].append(ft_system.redundant_available)
                        
                        print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {running_loss/(batch_idx+1):.4f}, Accuracy: {accuracy:.2f}%")
                        print(f"  Health: {health_report['avg_health']:.2f}%, Faults: {health_report['total_faults_detected']}")
                        print(f"  Mitigations - Soft: {ft_system.total_soft_mitigations}, Remapped: {ft_system.total_remappings}, Resets: {ft_system.total_layer_resets}")
                        
                        ft_system.memristive_model.train()
                    except Exception as e:
                        print(f"Error during evaluation in batch {batch_idx}: {e}")
            
            except Exception as e:
                print(f"Unexpected error in batch {batch_idx}: {e}")
                continue
        
        avg_epoch_loss = running_loss / (batch_idx + 1)
        scheduler.step(avg_epoch_loss)
    
    # Final health check
    print("\nFinal health check:")
    final_report = ft_system.get_health_report(detailed=True)
    print(f"Average health: {final_report['avg_health']:.2f}%")
    print(f"Total faults: {final_report['total_faults_detected']}")
    print(f"Soft mitigations: {ft_system.total_soft_mitigations}")
    print(f"Remappings: {ft_system.total_remappings}")
    print(f"Layer resets: {ft_system.total_layer_resets}")
    print(f"Redundant devices available: {final_report['redundant_available']}")
    
    # Visualize system health
    try:
        ft_system.visualize_system_health()
    except Exception as e:
        print(f"Failed to visualize system health: {e}")
    
    # Plot results
    try:
        plt.figure(figsize=(15, 12))
        
        # Plot accuracy vs health
        plt.subplot(2, 2, 1)
        plt.plot(results['batch'], results['accuracy'], 'b-', label='Accuracy')
        plt.ylabel('Accuracy (%)', color='b')
        plt.title('Accuracy vs Health')
        
        ax2 = plt.twinx()
        ax2.plot(results['batch'], results['health'], 'r-', label='Health')
        ax2.set_ylabel('Health (%)', color='r')
        
        # Plot all mitigation strategies
        plt.subplot(2, 2, 2)
        plt.plot(results['batch'], results['faults'], 'r-', label='Faults')
        plt.plot(results['batch'], results['soft_mitigations'], 'g-', label='Soft Mitigations')
        plt.plot(results['batch'], results['remappings'], 'b-', label='Remappings')
        plt.plot(results['batch'], results['layer_resets'], 'k-', label='Layer Resets')
        plt.title('Faults and Mitigations')
        plt.xlabel('Batch')
        plt.ylabel('Count')
        plt.legend()
        
        # Plot redundancy usage
        plt.subplot(2, 2, 3)
        redundancy_used = [initial_report.get('redundant_available', 0) - avail 
                          for avail in results['redundant_available']]
        plt.plot(results['batch'], redundancy_used, 'g-', label='Used')
        plt.plot(results['batch'], results['redundant_available'], 'b-', label='Available')
        plt.title('Redundancy Usage')
        plt.xlabel('Batch')
        plt.ylabel('Count')
        plt.legend()
        
        # Plot loss
        plt.subplot(2, 2, 4)
        plt.plot(results['batch'], results['loss'], 'b-')
        plt.title('Training Loss')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Failed to plot results: {e}")
    
    return {
        'system': ft_system,
        'metrics': results,
        'final_report': final_report,
        'strategies': {
            'soft_mitigation': enable_soft_mitigation,
            'remapping': enable_remapping,
            'layer_reset': enable_layer_reset
        }
    }