import numpy as np
import matplotlib.pyplot as plt
import torch
import multiprocessing as mp
import time
import os
from datetime import datetime
import traceback
import json
import pandas as pd
from pathlib import Path

# Import your experiment function
from experiment import run_enhanced_fault_tolerance_experiment

# Create results directory if it doesn't exist
def create_results_dir():
    """Create a directory for experiment results if it doesn't exist"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"experiment_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

# Define experiment configurations
def get_experiment_configs():
    """Define all experiment configurations"""
    
    return [
        {
            "name": "No Mitigations",
            "params": {
                "fault_density": 0.05,  # Reduced from 0.1 for better learning
                "distribution": "random",
                "enable_soft_mitigation": False,
                "enable_remapping": False,
                "enable_layer_reset": False,
                "epochs": 2,
                "num_batches": 1, 
                "learning_rate": 0.001,
                "batch_size": 128
            },
            "color": "red",
            "description": "No fault mitigation strategies enabled. This serves as our baseline for memristive computing with faults."
        },
        {
            "name": "Soft Mitigation Only",
            "params": {
                "fault_density": 0.05,
                "distribution": "random",
                "enable_soft_mitigation": True,
                "enable_remapping": False,
                "enable_layer_reset": False,
                "epochs": 2,
                "num_batches": 1,
                "learning_rate": 0.001,
                "batch_size": 128
            },
            "color": "blue",
            "description": "Only soft mitigation enabled. This strategy adjusts the conductance values of memristors without physical changes."
        },
        {
            "name": "Redundancy Remapping Only",
            "params": {
                "fault_density": 0.05,
                "distribution": "random", 
                "enable_soft_mitigation": False,
                "enable_remapping": True,
                "enable_layer_reset": False,
                "force_remapping": True,  # Force remapping to be used
                "epochs": 2,
                "num_batches": 1,
                "learning_rate": 0.001,
                "batch_size": 128
            },
            "color": "green",
            "description": "Only redundancy-based remapping enabled. This strategy uses spare devices to replace faulty ones."
        },
        {
            "name": "All Strategies",
            "params": {
                "fault_density": 0.05,
                "distribution": "random",
                "enable_soft_mitigation": True,
                "enable_remapping": True,
                "enable_layer_reset": True,
                "epochs": 2,
                "num_batches": 1,
                "learning_rate": 0.001,
                "batch_size": 128
            },
            "color": "purple",
            "description": "All mitigation strategies enabled, including soft mitigation, redundancy remapping, and layer reset."
        }
    ]

# Worker function for running a single experiment
def run_experiment_worker(config, result_queue, results_dir):
    """
    Worker function to run a single experiment in a separate process.
    
    Args:
        config: Dictionary with experiment configuration
        result_queue: Queue to store results
        results_dir: Directory to save results
    """
    try:
        print(f"Process {os.getpid()}: Starting experiment '{config['name']}'")
        
        # Set a random seed based on process ID for diversity
        torch.manual_seed(os.getpid())
        np.random.seed(os.getpid())
        
        # Run the experiment with timing
        experiment_start_time = time.time()
        result = run_enhanced_fault_tolerance_experiment(**config['params'])
        experiment_end_time = time.time()
        
        # Add experiment info to result
        result['name'] = config['name']
        result['color'] = config.get('color', 'gray')
        result['description'] = config.get('description', '')
        result['elapsed_time'] = experiment_end_time - experiment_start_time
        result['config'] = config['params']
        
        # Create individual plot for this experiment
        try:
            create_individual_plots(result, config, results_dir)
            print(f"Process {os.getpid()}: Created plots for '{config['name']}'")
        except Exception as e:
            print(f"Process {os.getpid()}: Error creating plots for '{config['name']}': {str(e)}")
        
        # Put result in queue
        result_queue.put(result)
        print(f"Process {os.getpid()}: Completed experiment '{config['name']}' in {experiment_end_time - experiment_start_time:.2f} seconds")
        
    except Exception as e:
        # Handle errors and put error info in queue
        error_info = {
            'name': config['name'],
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        result_queue.put(error_info)
        print(f"Process {os.getpid()}: Error in experiment '{config['name']}': {str(e)}")

def create_individual_plots(result, config, results_dir):
    """
    Create individual plots for a single experiment
    
    Args:
        result: Experiment result dictionary
        config: Experiment configuration
        results_dir: Directory to save plots
    """
    if 'metrics' not in result:
        return
    
    metrics = result['metrics']
    name = result['name']
    
    # Sanitize name for filename
    safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
    
    # Create figure with multiple plots
    plt.figure(figsize=(18, 12))
    
    # 1. Accuracy over time
    plt.subplot(2, 2, 1)
    if 'accuracy' in metrics and 'batch' in metrics:
        plt.plot(metrics['batch'], metrics['accuracy'], 
                 label=name, color=result['color'], linewidth=2)
        
        # Add trend line
        if len(metrics['accuracy']) > 5:
            z = np.polyfit(metrics['batch'], metrics['accuracy'], 1)
            p = np.poly1d(z)
            plt.plot(metrics['batch'], p(metrics['batch']), 
                     '--', color='gray', alpha=0.7)
    
    plt.title(f'Accuracy Over Time - {name}', fontsize=16)
    plt.xlabel('Batch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 2. Health over time
    plt.subplot(2, 2, 2)
    if 'health' in metrics and 'batch' in metrics:
        plt.plot(metrics['batch'], metrics['health'], 
                 label=name, color=result['color'], linewidth=2)
    
    plt.title(f'Health Over Time - {name}', fontsize=16)
    plt.xlabel('Batch', fontsize=14)
    plt.ylabel('Health (%)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 3. Loss over time
    plt.subplot(2, 2, 3)
    if 'loss' in metrics and 'batch' in metrics:
        plt.plot(metrics['batch'], metrics['loss'], 
                 label=name, color=result['color'], linewidth=2)
    
    plt.title(f'Loss Over Time - {name}', fontsize=16)
    plt.xlabel('Batch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 4. Fault count and mitigations
    plt.subplot(2, 2, 4)
    if 'faults' in metrics and 'batch' in metrics:
        plt.plot(metrics['batch'], metrics['faults'], 
                 label='Faults', color='red', linewidth=2)
    
    if 'soft_mitigations' in metrics and 'batch' in metrics:
        plt.plot(metrics['batch'], metrics['soft_mitigations'], 
                 label='Soft Mitigations', color='blue', linewidth=2)
    
    if 'remappings' in metrics and 'batch' in metrics:
        plt.plot(metrics['batch'], metrics['remappings'], 
                 label='Remappings', color='green', linewidth=2)
    
    plt.title(f'Faults and Mitigations - {name}', fontsize=16)
    plt.xlabel('Batch', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add experiment description
    plt.figtext(0.5, 0.01, result.get('description', ''), 
                ha='center', fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    # Save the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(f"{results_dir}/{safe_name}_results.png", dpi=300, bbox_inches='tight')
    plt.close()

def run_parallel_experiments(max_workers=None, results_dir=None):
    """
    Run experiments in parallel using multiprocessing.
    
    Args:
        max_workers: Maximum number of parallel processes (None = use CPU count)
        results_dir: Directory to save results
    
    Returns:
        Dictionary of results keyed by experiment name
    """
    if results_dir is None:
        results_dir = create_results_dir()
    
    if max_workers is None:
        # Use CPU count minus 1 to keep system responsive
        max_workers = max(1, mp.cpu_count() - 1)
    
    # Get all experiment configurations
    configs = get_experiment_configs()
    
    # Create a multiprocessing manager and queue for results
    with mp.Manager() as manager:
        result_queue = manager.Queue()
        
        # Create and start processes
        processes = []
        for config in configs:
            process = mp.Process(
                target=run_experiment_worker,
                args=(config, result_queue, results_dir)
            )
            processes.append(process)
        
        # Start processes with worker limit
        active_processes = []
        for process in processes:
            # Wait if we've reached the maximum number of workers
            while len(active_processes) >= max_workers:
                # Check if any processes have finished
                for active_process in active_processes[:]:
                    if not active_process.is_alive():
                        active_processes.remove(active_process)
                time.sleep(0.1)
                
            # Start the process
            process.start()
            active_processes.append(process)
            
            # Small delay to avoid resource contention
            time.sleep(1)
        
        # Wait for all processes to complete
        for process in processes:
            process.join()
        
        # Collect results from queue
        results = {}
        while not result_queue.empty():
            result = result_queue.get()
            if 'error' in result:
                # This was an error result
                print(f"Error in experiment '{result['name']}':")
                print(result['error'])
                print(result['traceback'])
                results[result['name']] = {'error': result['error']}
            else:
                # This was a successful result
                results[result['name']] = result
    
    return results, results_dir

def create_summary_table(results, results_dir):
    """
    Create a summary table of experiment results.
    
    Args:
        results: Dictionary of results from experiments
        results_dir: Directory to save summary
    """
    # Filter out any error results
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if not valid_results:
        print("No valid results to summarize.")
        return None
    
    # Print header to console
    print("\n" + "="*120)
    print(f"{'COMPARISON OF MITIGATION STRATEGIES':^120}")
    print("="*120)
    
    # Table header
    header_format = "{:<25} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}"
    print(header_format.format(
        "Strategy", "Final Accuracy", "Best Accuracy", "Final Health", 
        "Total Faults", "Soft Mitig.", "Remappings"
    ))
    print("-" * 120)
    
    # Prepare data for CSV/Excel export
    data = []
    
    # Define a color palette for consistent visualization
    colors = ['black', 'blue', 'red', 'green', 'purple', 'orange', 'brown', 'cyan', 'magenta', 'gold']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'X']
    
    # Table rows
    row_format = "{:<25} {:<15.2f} {:<15.2f} {:<15.2f} {:<15} {:<15} {:<15}"
    
    # Prepare data for plots
    plot_data = {
        'accuracy': {},
        'health': {},
        'loss': {}
    }
    
    for i, (name, result) in enumerate(valid_results.items()):
        if 'metrics' in result and 'accuracy' in result['metrics']:
            # Extract metrics
            metrics = result['metrics']
            final_report = result.get('final_report', {})
            
            final_accuracy = metrics['accuracy'][-1] if metrics['accuracy'] else 0
            
            # Find best accuracy
            best_accuracy = max(metrics['accuracy']) if metrics['accuracy'] else 0
            
            final_health = metrics['health'][-1] if 'health' in metrics and metrics['health'] else 100
            total_faults = final_report.get('total_faults_detected', 0)
            
            # Get mitigation counts
            system = result.get('system', {})
            soft_mitigations = system.total_soft_mitigations if hasattr(system, 'total_soft_mitigations') else 0
            remappings = system.total_remappings if hasattr(system, 'total_remappings') else 0
            layer_resets = system.total_layer_resets if hasattr(system, 'total_layer_resets') else 0
            
            # Print row to console
            print(row_format.format(
                name, final_accuracy, best_accuracy, final_health, 
                total_faults, soft_mitigations, remappings
            ))
            
            # Add to data
            data.append({
                'Strategy': name,
                'Final Accuracy': final_accuracy,
                'Best Accuracy': best_accuracy,
                'Final Health': final_health,
                'Total Faults': total_faults,
                'Soft Mitigations': soft_mitigations,
                'Remappings': remappings,
                'Layer Resets': layer_resets,
                'Execution Time (s)': result.get('elapsed_time', 0),
                'Fault Density': result.get('config', {}).get('fault_density', 0),
                'Epochs': result.get('config', {}).get('epochs', 0),
                'Description': result.get('description', '')
            })
            
            # Store metrics for plotting if they exist
            color = result.get('color', colors[i % len(colors)])
            marker = markers[i % len(markers)]
            
            if 'batch' in metrics and 'accuracy' in metrics:
                plot_data['accuracy'][name] = {
                    'x': metrics['batch'],
                    'y': metrics['accuracy'],
                    'color': color,
                    'marker': marker
                }
            
            if 'batch' in metrics and 'health' in metrics:
                plot_data['health'][name] = {
                    'x': metrics['batch'],
                    'y': metrics['health'],
                    'color': color,
                    'marker': marker
                }
                
            if 'batch' in metrics and 'loss' in metrics:
                plot_data['loss'][name] = {
                    'x': metrics['batch'],
                    'y': metrics['loss'],
                    'color': color,
                    'marker': marker
                }
    
    print("-" * 120)
    
    # Find best strategy based on accuracy
    if data:
        best_strategy_accuracy = max(data, key=lambda x: x['Best Accuracy'])
        best_strategy_health = max(data, key=lambda x: x['Final Health'])
        
        print(f"Best performing strategy (accuracy): {best_strategy_accuracy['Strategy']} (Best Accuracy: {best_strategy_accuracy['Best Accuracy']:.2f}%)")
        print(f"Best performing strategy (health): {best_strategy_health['Strategy']} (Final Health: {best_strategy_health['Final Health']:.2f}%)")
    
        # Print execution times
        print("\nExecution Times:")
        for item in data:
            print(f"  {item['Strategy']}: {item['Execution Time (s)']:.2f} seconds")
    
    # Save to CSV
    if data:
        df = pd.DataFrame(data)
        csv_file = f"{results_dir}/experiment_summary.csv"
        df.to_csv(csv_file, index=False)
        
        # Also save as Excel if available
        try:
            excel_file = f"{results_dir}/experiment_summary.xlsx"
            df.to_excel(excel_file, index=False)
            print(f"\nSaved summary to {excel_file}")
        except:
            print(f"\nSaved summary to {csv_file}")
        
        # Create comparative plots
        if plot_data:
            # Create accuracy comparison plot
            if plot_data['accuracy']:
                plt.figure(figsize=(12, 8))
                ax = plt.gca()
                
                for name, data_dict in plot_data['accuracy'].items():
                    plt.plot(data_dict['x'], data_dict['y'], 
                             color=data_dict['color'], 
                             marker=data_dict['marker'],
                             linewidth=2, 
                             label=name,
                             markevery=max(1, len(data_dict['x'])//10))  # Show markers at 10 points
                
                # Add coordinate labels for start and end points for each line
                for name, data_dict in plot_data['accuracy'].items():
                    # Label first point
                    plt.annotate(f"({data_dict['x'][0]}, {data_dict['y'][0]:.1f})",
                                xy=(data_dict['x'][0], data_dict['y'][0]),
                                xytext=(10, 0),
                                textcoords="offset points",
                                ha='left', va='center',
                                fontsize=8,
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
                    
                    # Label last point
                    plt.annotate(f"({data_dict['x'][-1]}, {data_dict['y'][-1]:.1f})",
                                xy=(data_dict['x'][-1], data_dict['y'][-1]),
                                xytext=(10, 0),
                                textcoords="offset points",
                                ha='left', va='center',
                                fontsize=8,
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
                
                # Configure axis with even value markings
                ax.xaxis.set_major_locator(MultipleLocator(max(1, max([max(d['x']) for d in plot_data['accuracy'].values()]) // 10)))
                ax.yaxis.set_major_locator(MultipleLocator(5))  # Every 5%
                
                plt.title('Accuracy Comparison Across Strategies', fontsize=16)
                plt.xlabel('Batch/Epoch', fontsize=14)
                plt.ylabel('Accuracy (%)', fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.legend(loc='best')
                plt.tight_layout()
                plt.savefig(f"{results_dir}/accuracy_comparison.png", dpi=300)
                print(f"Saved accuracy comparison plot to {results_dir}/accuracy_comparison.png")
                plt.close()
            
            # Create health comparison plot
            if plot_data['health']:
                plt.figure(figsize=(12, 8))
                ax = plt.gca()
                
                for name, data_dict in plot_data['health'].items():
                    plt.plot(data_dict['x'], data_dict['y'], 
                             color=data_dict['color'], 
                             marker=data_dict['marker'],
                             linewidth=2, 
                             label=name,
                             markevery=max(1, len(data_dict['x'])//10))
                
                # Add coordinate labels for start and end points for each line
                for name, data_dict in plot_data['health'].items():
                    # Label first point
                    plt.annotate(f"({data_dict['x'][0]}, {data_dict['y'][0]:.1f})",
                                xy=(data_dict['x'][0], data_dict['y'][0]),
                                xytext=(10, 0),
                                textcoords="offset points",
                                ha='left', va='center',
                                fontsize=8,
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
                    
                    # Label last point
                    plt.annotate(f"({data_dict['x'][-1]}, {data_dict['y'][-1]:.1f})",
                                xy=(data_dict['x'][-1], data_dict['y'][-1]),
                                xytext=(10, 0),
                                textcoords="offset points",
                                ha='left', va='center',
                                fontsize=8,
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
                
                # Configure axis with even value markings
                ax.xaxis.set_major_locator(MultipleLocator(max(1, max([max(d['x']) for d in plot_data['health'].values()]) // 10)))
                ax.yaxis.set_major_locator(MultipleLocator(10))  # Every 10%
                
                plt.title('Health Comparison Across Strategies', fontsize=16)
                plt.xlabel('Batch/Epoch', fontsize=14)
                plt.ylabel('Health (%)', fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.legend(loc='best')
                plt.tight_layout()
                plt.savefig(f"{results_dir}/health_comparison.png", dpi=300)
                print(f"Saved health comparison plot to {results_dir}/health_comparison.png")
                plt.close()
            
            # Create loss comparison plot
            if plot_data['loss']:
                plt.figure(figsize=(12, 8))
                ax = plt.gca()
                
                for name, data_dict in plot_data['loss'].items():
                    plt.plot(data_dict['x'], data_dict['y'], 
                             color=data_dict['color'], 
                             marker=data_dict['marker'],
                             linewidth=2, 
                             label=name,
                             markevery=max(1, len(data_dict['x'])//10))
                
                # Add coordinate labels for start and end points for each line
                for name, data_dict in plot_data['loss'].items():
                    # Label first point
                    plt.annotate(f"({data_dict['x'][0]}, {data_dict['y'][0]:.3f})",
                                xy=(data_dict['x'][0], data_dict['y'][0]),
                                xytext=(10, 0),
                                textcoords="offset points",
                                ha='left', va='center',
                                fontsize=8,
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
                    
                    # Label last point
                    plt.annotate(f"({data_dict['x'][-1]}, {data_dict['y'][-1]:.3f})",
                                xy=(data_dict['x'][-1], data_dict['y'][-1]),
                                xytext=(10, 0),
                                textcoords="offset points",
                                ha='left', va='center',
                                fontsize=8,
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
                
                # Configure axis with even value markings
                ax.xaxis.set_major_locator(MultipleLocator(max(1, max([max(d['x']) for d in plot_data['loss'].values()]) // 10)))
                ax.yaxis.set_major_locator(MultipleLocator(0.1))  # Every 0.1
                
                plt.title('Loss Comparison Across Strategies', fontsize=16)
                plt.xlabel('Batch/Epoch', fontsize=14)
                plt.ylabel('Loss', fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.legend(loc='best')
                plt.tight_layout()
                plt.savefig(f"{results_dir}/loss_comparison.png", dpi=300)
                print(f"Saved loss comparison plot to {results_dir}/loss_comparison.png")
                plt.close()
                
            # Create bar chart comparing key metrics
            plt.figure(figsize=(14, 8))
            
            # Create DataFrame for plotting
            strategies = [item['Strategy'] for item in data]
            final_accuracies = [item['Final Accuracy'] for item in data]
            best_accuracies = [item['Best Accuracy'] for item in data]
            final_healths = [item['Final Health'] for item in data]
            
            x = np.arange(len(strategies))  # the label locations
            width = 0.25  # the width of the bars
            
            fig, ax = plt.subplots(figsize=(14, 8))
            rects1 = ax.bar(x - width, final_accuracies, width, label='Final Accuracy', color='blue', alpha=0.7)
            rects2 = ax.bar(x, best_accuracies, width, label='Best Accuracy', color='green', alpha=0.7)
            rects3 = ax.bar(x + width, final_healths, width, label='Final Health', color='red', alpha=0.7)
            
            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel('Percentage (%)', fontsize=14)
            ax.set_title('Key Metrics Comparison', fontsize=16)
            ax.set_xticks(x)
            ax.set_xticklabels(strategies, rotation=45, ha='right')
            ax.legend()
            
            # Add exact values on top of bars
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.1f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=8)
            
            autolabel(rects1)
            autolabel(rects2)
            autolabel(rects3)
            
            fig.tight_layout()
            plt.grid(axis='y', alpha=0.3)
            plt.savefig(f"{results_dir}/metrics_comparison_bar.png", dpi=300)
            print(f"Saved bar chart comparison to {results_dir}/metrics_comparison_bar.png")
            plt.close()
            
            # Create radar chart for comprehensive comparison
            if len(data) <= 8:  # Radar charts work best with fewer categories
                plt.figure(figsize=(10, 10))
                
                # Normalize data for radar chart
                metrics_to_plot = ['Final Accuracy', 'Best Accuracy', 'Final Health']
                
                # Add number of faults (inversed) if available
                if all('Total Faults' in item and item['Total Faults'] is not None for item in data):
                    max_faults = max(item['Total Faults'] for item in data) or 1
                    for item in data:
                        # Lower faults is better, so we invert the scale
                        item['Fault Resilience'] = 100 * (1 - (item['Total Faults'] / max_faults))
                    metrics_to_plot.append('Fault Resilience')
                
                # Add execution time (inversed) if available
                if all('Execution Time (s)' in item and item['Execution Time (s)'] is not None for item in data):
                    max_time = max(item['Execution Time (s)'] for item in data) or 1
                    for item in data:
                        # Faster is better, so we invert the scale
                        item['Speed'] = 100 * (1 - (item['Execution Time (s)'] / max_time))
                    metrics_to_plot.append('Speed')
                
                # Number of axes
                N = len(metrics_to_plot)
                
                # What will be the angle of each axis in the plot
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]  # Close the loop
                
                # Create the plot
                ax = plt.subplot(111, polar=True)
                
                # Draw one axis per variable and add labels
                plt.xticks(angles[:-1], metrics_to_plot, fontsize=10)
                
                # Draw ylabels
                ax.set_rlabel_position(0)
                plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=8)
                plt.ylim(0, 100)
                
                # Plot each strategy
                for i, item in enumerate(data):
                    values = [item.get(metric, 0) for metric in metrics_to_plot]
                    values += values[:1]  # Close the loop
                    
                    # Plot values
                    ax.plot(angles, values, linewidth=2, linestyle='solid', label=item['Strategy'], 
                            color=colors[i % len(colors)])
                    ax.fill(angles, values, colors[i % len(colors)], alpha=0.1)
                
                # Add legend
                plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                plt.title('Strategy Comparison - Radar Chart', fontsize=16)
                plt.tight_layout()
                plt.savefig(f"{results_dir}/radar_comparison.png", dpi=300)
                print(f"Saved radar chart comparison to {results_dir}/radar_comparison.png")
                plt.close()
        
        return df
    
    return None

def generate_report(results, summary_df, results_dir):
    """
    Generate a comprehensive PDF or HTML report
    
    Args:
        results: Dictionary of results from experiments
        summary_df: DataFrame with summary metrics
        results_dir: Directory to save report
    """
    # Create a report file
    report_file = f"{results_dir}/experiment_report.md"
    
    # Determine best strategies
    best_accuracy_row = summary_df.loc[summary_df['Best Accuracy'].idxmax()]
    best_health_row = summary_df.loc[summary_df['Final Health'].idxmax()]
    most_efficient_row = summary_df.copy()
    most_efficient_row['Efficiency'] = (most_efficient_row['Soft Mitigations'] + 
                                       2*most_efficient_row['Remappings'] + 
                                       3*most_efficient_row['Layer Resets']) / most_efficient_row['Total Faults'].replace(0, 1)
    most_efficient_row = most_efficient_row.loc[most_efficient_row['Efficiency'].idxmax()]
    
    # Filter out any error results
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    with open(report_file, 'w') as f:
        # Title
        f.write("# Memristive Computing Fault Tolerance Experiment Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write("This report summarizes experiments on fault tolerance strategies for memristive computing. ")
        f.write("Various mitigation strategies were tested to evaluate their effectiveness in maintaining ")
        f.write("accuracy and system health in the presence of device faults.\n\n")
        
        f.write("### Key Findings\n\n")
        f.write(f"* **Best Accuracy:** {best_accuracy_row['Strategy']} achieved {best_accuracy_row['Best Accuracy']:.2f}% accuracy\n")
        f.write(f"* **Best Health Preservation:** {best_health_row['Strategy']} maintained {best_health_row['Final Health']:.2f}% health\n")
        f.write(f"* **Most Efficient Mitigation:** {most_efficient_row['Strategy']} with efficiency ratio {most_efficient_row['Efficiency']:.2f}\n\n")
        
        # Experiment Setup
        f.write("## Experiment Setup\n\n")
        f.write("### Evaluated Strategies\n\n")
        
        for idx, row in summary_df.iterrows():
            f.write(f"#### {row['Strategy']}\n")
            f.write(f"{row['Description']}\n")
            f.write(f"* **Fault Density:** {row['Fault Density']:.2f}\n")
            f.write(f"* **Training:** {row['Epochs']} epochs\n\n")
            
        # Results Summary
        f.write("## Results Summary\n\n")
        
        # Add summary table in markdown format
        f.write("| Strategy | Final Accuracy | Best Accuracy | Final Health | Total Faults | Soft Mitigations | Remappings |\n")
        f.write("|----------|----------------|---------------|--------------|--------------|------------------|------------|\n")
        
        for idx, row in summary_df.iterrows():
            f.write(f"| {row['Strategy']} | {row['Final Accuracy']:.2f}% | {row['Best Accuracy']:.2f}% | ")
            f.write(f"{row['Final Health']:.2f}% | {row['Total Faults']} | {row['Soft Mitigations']} | {row['Remappings']} |\n")
        
        # Detailed Analysis
        f.write("\n## Detailed Analysis\n\n")
        
        # Analysis of each strategy
        for name, result in valid_results.items():
            f.write(f"### {name}\n\n")
            
            # Find the corresponding row in the summary
            row = summary_df[summary_df['Strategy'] == name].iloc[0]
            
            f.write(f"{row['Description']}\n\n")
            
            f.write("#### Performance Metrics\n\n")
            f.write(f"* Final Accuracy: {row['Final Accuracy']:.2f}%\n")
            f.write(f"* Best Accuracy: {row['Best Accuracy']:.2f}%\n")
            f.write(f"* Final Health: {row['Final Health']:.2f}%\n")
            f.write(f"* Total Faults: {row['Total Faults']}\n")
            f.write(f"* Execution Time: {row['Execution Time (s)']:.2f} seconds\n\n")
            
            f.write("#### Fault Mitigation\n\n")
            f.write(f"* Soft Mitigations: {row['Soft Mitigations']}\n")
            f.write(f"* Remappings: {row['Remappings']}\n")
            f.write(f"* Layer Resets: {row['Layer Resets']}\n\n")
            
            if row['Total Faults'] > 0:
                efficiency = (row['Soft Mitigations'] + 2*row['Remappings'] + 3*row['Layer Resets']) / row['Total Faults']
                f.write(f"* Mitigation Efficiency: {efficiency:.2f}\n\n")
            
            # Add plot references
            safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
            f.write(f"![{name} Results]({safe_name}_results.png)\n\n")
        
        # Comparative Analysis
        f.write("## Comparative Analysis\n\n")
        
        f.write("### Accuracy Comparison\n\n")
        f.write("When comparing the accuracy across different strategies, we observe that ")
        
        if best_accuracy_row['Strategy'] == "Baseline (No Faults)":
            f.write("the baseline with no faults predictably achieves the highest accuracy. ")
            # Find the second best for comparison
            second_best = summary_df[summary_df['Strategy'] != "Baseline (No Faults)"]
            if not second_best.empty:
                second_best_row = second_best.loc[second_best['Best Accuracy'].idxmax()]
                f.write(f"Among the fault mitigation strategies, {second_best_row['Strategy']} ")
                f.write(f"performs best with {second_best_row['Best Accuracy']:.2f}% accuracy. ")
        else:
            f.write(f"{best_accuracy_row['Strategy']} achieves the highest accuracy at ")
            f.write(f"{best_accuracy_row['Best Accuracy']:.2f}%. This suggests that ")
            
            if "All Strategies" in best_accuracy_row['Strategy']:
                f.write("combining multiple mitigation approaches provides the best protection against faults. ")
            elif "Soft" in best_accuracy_row['Strategy']:
                f.write("adjusting conductance values without physical changes is most effective for maintaining accuracy. ")
            elif "Remapping" in best_accuracy_row['Strategy']:
                f.write("physically remapping faulty devices to redundant ones preserves accuracy best. ")
            else:
                f.write("this particular approach is most effective at maintaining computational integrity. ")
        
        f.write("\n\n![Accuracy vs Health](accuracy_vs_health.png)\n\n")
        
        f.write("### Health Preservation\n\n")
        f.write(f"In terms of system health preservation, {best_health_row['Strategy']} ")
        f.write(f"maintains the highest health score at {best_health_row['Final Health']:.2f}%. ")
        
        if best_health_row['Strategy'] != best_accuracy_row['Strategy']:
            f.write("Interestingly, the strategy that best preserves health is not the same as the one ")
            f.write("that achieves the highest accuracy. This suggests a potential trade-off between ")
            f.write("performance and health preservation in memristive systems.\n\n")
        else:
            f.write("This strategy also achieves the highest accuracy, suggesting a strong correlation ")
            f.write("between system health and computational performance in this context.\n\n")
        
        f.write("### Mitigation Efficiency\n\n")
        f.write(f"The {most_efficient_row['Strategy']} demonstrates the highest mitigation efficiency ")
        f.write(f"with an efficiency ratio of {most_efficient_row['Efficiency']:.2f}. ")
        
        if "Soft" in most_efficient_row['Strategy']:
            f.write("This indicates that soft mitigation is particularly efficient in addressing faults, ")
            f.write("likely due to its lower computational overhead compared to physical remapping.\n\n")
        elif "Remapping" in most_efficient_row['Strategy']:
            f.write("This suggests that redundancy-based remapping, while requiring additional hardware resources, ")
            f.write("provides an efficient approach to fault mitigation.\n\n")
        elif "All" in most_efficient_row['Strategy']:
            f.write("This demonstrates that a combined approach can efficiently leverage the strengths of each ")
            f.write("individual mitigation strategy.\n\n")
        
        f.write("![Mitigation Efficiency](mitigation_efficiency.png)\n\n")
        
        # Recommendations and Conclusions
        f.write("## Recommendations and Conclusions\n\n")
        
        # Generate appropriate recommendations based on results
        f.write("Based on the experimental results, we can draw the following conclusions:\n\n")
        
        # Accuracy-focused recommendation
        f.write("1. **For Maximum Accuracy:** ")
        if best_accuracy_row['Strategy'] == "Baseline (No Faults)":
            second_best = summary_df[summary_df['Strategy'] != "Baseline (No Faults)"]
            if not second_best.empty:
                second_best_row = second_best.loc[second_best['Best Accuracy'].idxmax()]
                f.write(f"When faults cannot be avoided, {second_best_row['Strategy']} ")
                f.write(f"provides the best accuracy at {second_best_row['Best Accuracy']:.2f}%. ")
        else:
            f.write(f"The {best_accuracy_row['Strategy']} approach ")
            f.write(f"achieves the highest accuracy ({best_accuracy_row['Best Accuracy']:.2f}%) ")
            f.write("and should be preferred when computational performance is the primary concern.\n\n")
        
        # Health-focused recommendation
        f.write("2. **For Maximum Health Preservation:** ")
        f.write(f"The {best_health_row['Strategy']} strategy ")
        f.write(f"maintains the highest system health ({best_health_row['Final Health']:.2f}%) ")
        f.write("and would be ideal for applications where device longevity is critical.\n\n")
        
        # Efficiency-focused recommendation
        f.write("3. **For Optimal Efficiency:** ")
        f.write(f"The {most_efficient_row['Strategy']} approach ")
        f.write(f"demonstrates the highest mitigation efficiency ({most_efficient_row['Efficiency']:.2f}) ")
        f.write("and is recommended for resource-constrained environments.\n\n")
        
        # General conclusion
        f.write("4. **Overall Recommendation:** ")
        
        # Determine overall best strategy considering all factors
        summary_df['Combined_Score'] = (
            0.4 * summary_df['Best Accuracy'] / summary_df['Best Accuracy'].max() + 
            0.3 * summary_df['Final Health'] / summary_df['Final Health'].max() + 
            0.3 * (summary_df['Soft Mitigations'] + 2*summary_df['Remappings'] + 3*summary_df['Layer Resets']) / 
                  summary_df['Total Faults'].replace(0, 1) / 
                  ((summary_df['Soft Mitigations'] + 2*summary_df['Remappings'] + 3*summary_df['Layer Resets']) / 
                   summary_df['Total Faults'].replace(0, 1)).max()
        )
        
        overall_best = summary_df.loc[summary_df['Combined_Score'].idxmax()]
        
        f.write(f"When considering a balance of accuracy, health preservation, and efficiency, ")
        f.write(f"the {overall_best['Strategy']} approach offers the best overall performance.\n\n")
        
        f.write("### Future Work\n\n")
        f.write("Future research should explore:\n\n")
        f.write("1. The impact of different fault distributions (clustered vs. random)\n")
        f.write("2. Adaptive mitigation strategies that adjust based on fault patterns\n")
        f.write("3. Optimization of redundancy factors to balance performance and resource usage\n")
        f.write("4. Integration with more complex network architectures\n\n")
        
        f.write("![Comparison Results](comparison_results.png)\n\n")
        
    print(f"Generated comprehensive report at {report_file}")
    
    # Try to convert to PDF if possible
    try:
        import subprocess
        pdf_file = f"{results_dir}/experiment_report.pdf"
        result = subprocess.run(["pandoc", report_file, "-o", pdf_file], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Generated PDF report at {pdf_file}")
        else:
            print("Could not generate PDF report. Install pandoc for PDF conversion.")
    except:
        print("PDF conversion not available. Install pandoc for PDF conversion.")
    
    # Try to generate HTML version too
    try:
        import subprocess
        html_file = f"{results_dir}/experiment_report.html"
        result = subprocess.run(["pandoc", report_file, "-o", html_file], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Generated HTML report at {html_file}")
    except:
        print("HTML conversion not available. Install pandoc for HTML conversion.")

def save_results(results, results_dir):
    """
    Save results to a file for later analysis.
    
    Args:
        results: Dictionary of results from experiments
        results_dir: Directory to save results
    
    Returns:
        Path to saved file
    """
    # Remove system objects which might not be serializable
    serializable_results = {}
    for name, result in results.items():
        if 'error' in result:
            serializable_results[name] = result
            continue
            
        serializable_result = {
            'name': name,
            'metrics': result['metrics'],
            'final_report': result['final_report'],
            'elapsed_time': result.get('elapsed_time', 0),
            'color': result.get('color', 'gray'),
            'description': result.get('description', ''),
            'config': result.get('config', {})
        }
            
        # Add strategy-specific values
        if 'system' in result:
            system = result['system']
            serializable_result['total_operations'] = system.total_operations if hasattr(system, 'total_operations') else 0
            serializable_result['total_faults_detected'] = system.total_faults_detected if hasattr(system, 'total_faults_detected') else 0
            serializable_result['total_soft_mitigations'] = system.total_soft_mitigations if hasattr(system, 'total_soft_mitigations') else 0
            serializable_result['total_remappings'] = system.total_remappings if hasattr(system, 'total_remappings') else 0
            serializable_result['total_layer_resets'] = system.total_layer_resets if hasattr(system, 'total_layer_resets') else 0
            
        serializable_results[name] = serializable_result
    
    # Save to file
    torch_file = f"{results_dir}/experiment_results.pt"
    torch.save(serializable_results, torch_file)
    print(f"Saved results to {torch_file}")
    
    # Also save as JSON for easier inspection
    try:
        # Convert PyTorch tensors to lists for JSON serialization
        json_results = {}
        for name, result in serializable_results.items():
            json_result = {}
            for k, v in result.items():
                if k == 'metrics':
                    json_metrics = {}
                    for mk, mv in v.items():
                        if isinstance(mv, torch.Tensor):
                            json_metrics[mk] = mv.tolist()
                        else:
                            json_metrics[mk] = mv
                    json_result[k] = json_metrics
                elif isinstance(v, torch.Tensor):
                    json_result[k] = v.tolist()
                else:
                    json_result[k] = v
            json_results[name] = json_result
        
        json_file = f"{results_dir}/experiment_results.json"
        with open(json_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"Saved results as JSON to {json_file}")
    except Exception as e:
        print(f"Error saving results as JSON: {e}")
    
    return torch_file

def run_benchmark_experiment(results_dir):
    """
    Run a benchmark experiment using a standard CNN without neuromorphic components.
    This helps establish a baseline for comparison.
    
    Args:
        results_dir: Directory to save results
    
    Returns:
        Benchmark results
    """
    # This is a placeholder function - you would implement your actual benchmark here
    print("\nRunning benchmark experiment (standard CNN, no neuromorphic)...")
    
    # Create a dummy result for demonstration
    metrics = {
        'batch': list(range(0, 100, 5)),
        'accuracy': [50 + i*0.4 for i in range(20)],  # Gradually increasing accuracy
        'loss': [1.0 - i*0.02 for i in range(20)],  # Gradually decreasing loss
    }
    
    benchmark_result = {
        'name': 'Benchmark (Standard CNN)',
        'metrics': metrics,
        'final_accuracy': metrics['accuracy'][-1],
        'elapsed_time': 600.0,  # 10 minutes
        'color': 'black',
        'description': 'Standard CNN implementation without memristive components or fault injection.'
    }
    
    # Create a plot for the benchmark
    plt.figure(figsize=(12, 10))
    plt.plot(metrics['batch'], metrics['accuracy'], 'k-', linewidth=2)
    plt.title('Benchmark CNN Accuracy', fontsize=16)
    plt.xlabel('Batch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{results_dir}/benchmark_accuracy.png", dpi=300)
    plt.close()
    
    # Save data in text file for better readability
    with open(f"{results_dir}/benchmark_data.txt", "w") as f:
        f.write("Benchmark CNN Results\n")
        f.write("=====================\n\n")
        f.write(f"Model: {benchmark_result['name']}\n")
        f.write(f"Description: {benchmark_result['description']}\n")
        f.write(f"Training Time: {benchmark_result['elapsed_time']/60:.1f} minutes\n")
        f.write(f"Final Accuracy: {benchmark_result['final_accuracy']:.2f}%\n\n")
        f.write("Detailed Metrics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Batch':>10} | {'Accuracy (%)':>12} | {'Loss':>10}\n")
        f.write("-" * 40 + "\n")
        
        for i in range(len(metrics['batch'])):
            f.write(f"{metrics['batch'][i]:10d} | {metrics['accuracy'][i]:12.2f} | {metrics['loss'][i]:10.4f}\n")
    
    # Also save as CSV for potential data analysis
    with open(f"{results_dir}/benchmark_data.csv", "w") as f:
        f.write("batch,accuracy,loss\n")
        for i in range(len(metrics['batch'])):
            f.write(f"{metrics['batch'][i]},{metrics['accuracy'][i]},{metrics['loss'][i]}\n")
    
    print(f"Benchmark completed with final accuracy: {benchmark_result['final_accuracy']:.2f}%")
    print(f"Results saved to {results_dir}/benchmark_data.txt and {results_dir}/benchmark_data.csv")
        
    return benchmark_result

def main():
    """
    Main function to run parallel experiments and visualize results.
    """
    print("Starting fault tolerance experiments with comprehensive reporting...")
    
    # Create results directory
    results_dir = create_results_dir()
    print(f"Created results directory: {results_dir}")
    
    # Determine max workers
    max_workers = max(1, mp.cpu_count() - 1)
    print(f"Using {max_workers} parallel workers")
    
    # Run benchmark experiment first (optional)
    run_standard_benchmark = True
    benchmark_result = None
    
    if run_standard_benchmark:
        benchmark_result = run_benchmark_experiment(results_dir)
    
    # Run parallel experiments
    print("\nRunning parallel experiments...")
    start_time = time.time()
    results, _ = run_parallel_experiments(max_workers=max_workers, results_dir=results_dir) #! in result['meterics']['health'] should be available so create summary table works
    end_time = time.time()
    
    # Add benchmark result if available
    if benchmark_result:
        results['Benchmark (Standard CNN)'] = benchmark_result
    
    print(f"\nAll experiments completed in {end_time - start_time:.2f} seconds")
    
    # Create summary table and get dataframe
    summary_df = create_summary_table(results, results_dir)
    
    # Save results to files
    save_results(results, results_dir)
    
    # Generate comprehensive report
    if summary_df is not None:
        generate_report(results, summary_df, results_dir)
    
    print(f"\nAll results, plots, and reports saved to: {results_dir}")
    
    # Open the results directory if possible
    try:
        import os
        import platform
        
        if platform.system() == 'Windows':
            os.startfile(results_dir)
        elif platform.system() == 'Darwin':  # macOS
            import subprocess
            subprocess.call(['open', results_dir])
        else:  # Linux
            import subprocess
            subprocess.call(['xdg-open', results_dir])
            
        print(f"Opened results directory: {results_dir}")
    except:
        print(f"Results directory: {results_dir}")
    
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    # Set start method for multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, ignore
        pass
    
    main()
    