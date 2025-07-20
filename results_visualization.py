import os
import re
import numpy as np
import matplotlib.pyplot as plt
import glob

def parse_game_log(file_path):
    """Parse game_log.txt and return (converged, rounds, parsing_failure)"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        if "PARSING FAILURE" in content:
            return False, None, True
        elif "CORRECT!" in content:
            rounds = len(re.findall(r'Round \d+:', content))
            return True, rounds, False
        else:
            return False, None, False
    except:
        return False, None, True

def collect_results(base_path):
    """Collect all experiment results"""
    results = {}
    
    for config_dir in glob.glob(os.path.join(base_path, "sum_a*_t*")):
        # Extract agents and temp from folder name
        match = re.search(r'sum_a(\d+)_t(\d+\.\d+)', config_dir)
        if not match:
            continue
        
        agents, temp = int(match.group(1)), float(match.group(2))
        
        converged_count = 0
        not_converged_count = 0
        parsing_failure_count = 0
        convergence_rounds = []
        
        # Check all runs
        for run_dir in glob.glob(os.path.join(config_dir, "run_*")):
            log_file = os.path.join(run_dir, "game_log.txt")
            if os.path.exists(log_file):
                converged, rounds, parsing_failure = parse_game_log(log_file)
                
                if parsing_failure:
                    parsing_failure_count += 1
                elif converged:
                    converged_count += 1
                    convergence_rounds.append(rounds)
                else:
                    not_converged_count += 1
        
        results[(agents, temp)] = {
            'converged': converged_count,
            'not_converged': not_converged_count, 
            'parsing_failure': parsing_failure_count,
            'convergence_rounds': convergence_rounds
        }
    
    return results

def create_plots(results, base_path):
    """Create the plots and save each as separate PDF"""
    # Check if we have any results
    if not results:
        print(f"No results found in {base_path}")
        return
    
    # Create plots directory
    plots_dir = os.path.join(base_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get unique agents and temps
    agents_list = sorted(set(k[0] for k in results.keys()))
    temp_list = sorted(set(k[1] for k in results.keys()))
    
    # Check if we have valid data
    if not agents_list or not temp_list:
        print(f"No valid agent/temperature data found in {base_path}")
        return
    
    # Create data arrays
    n_agents, n_temps = len(agents_list), len(temp_list)
    converged_props = np.zeros((n_agents, n_temps))
    not_converged_props = np.zeros((n_agents, n_temps))
    parsing_failure_props = np.zeros((n_agents, n_temps))
    converged_counts = np.zeros((n_agents, n_temps))
    not_converged_counts = np.zeros((n_agents, n_temps))
    parsing_failure_counts = np.zeros((n_agents, n_temps))
    mean_rounds = np.zeros((n_agents, n_temps))
    std_errors = np.zeros((n_agents, n_temps))
    successful_runs = np.zeros((n_agents, n_temps))  # Track number of successful runs
    
    # Fill arrays
    for i, agents in enumerate(agents_list):
        for j, temp in enumerate(temp_list):
            if (agents, temp) in results:
                data = results[(agents, temp)]
                total = data['converged'] + data['not_converged'] + data['parsing_failure']
                
                # Store actual counts
                converged_counts[i, j] = data['converged']
                not_converged_counts[i, j] = data['not_converged']
                parsing_failure_counts[i, j] = data['parsing_failure']
                
                if total > 0:
                    converged_props[i, j] = data['converged'] / total
                    not_converged_props[i, j] = data['not_converged'] / total
                    parsing_failure_props[i, j] = data['parsing_failure'] / total
                
                if data['convergence_rounds']:
                    mean_rounds[i, j] = np.mean(data['convergence_rounds'])
                    std_errors[i, j] = np.std(data['convergence_rounds'])  # Full std, not std error
                    successful_runs[i, j] = len(data['convergence_rounds'])
    
    temp_mesh, agents_mesh = np.meshgrid(temp_list, agents_list)
    
    # Plot 1: 3D Stacked proportions
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    
    # Check if we have any non-zero data for 3D plots
    if np.any(converged_props) or np.any(not_converged_props) or np.any(parsing_failure_props):
        bottom = np.zeros_like(temp_mesh)
        ax1.bar3d(temp_mesh.ravel(), agents_mesh.ravel(), bottom.ravel(),
                  0.08, 0.8, converged_props.ravel(), color='green', alpha=0.8, label='Converged')
        
        bottom += converged_props
        ax1.bar3d(temp_mesh.ravel(), agents_mesh.ravel(), bottom.ravel(),
                  0.08, 0.8, not_converged_props.ravel(), color='orange', alpha=0.8, label='Not Converged')
        
        bottom += not_converged_props
        ax1.bar3d(temp_mesh.ravel(), agents_mesh.ravel(), bottom.ravel(),
                  0.08, 0.8, parsing_failure_props.ravel(), color='red', alpha=0.8, label='Parsing Failure')
    
    ax1.set_xlabel('Temperature')
    ax1.set_ylabel('Agents')
    ax1.set_zlabel('Proportion')
    ax1.set_title('Outcome Proportions (3D)')
    ax1.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot1_3d_proportions.pdf"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: 3D Mean convergence rounds
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    
    mask = mean_rounds > 0
    if np.any(mask):
        ax2.bar3d(temp_mesh[mask], agents_mesh[mask], np.zeros(np.sum(mask)),
                  0.08, 0.8, mean_rounds[mask], color='blue', alpha=0.7)
        
        # Add error bars
        for i in range(n_agents):
            for j in range(n_temps):
                if mask[i, j]:
                    ax2.plot([temp_list[j], temp_list[j]], 
                            [agents_list[i], agents_list[i]], 
                            [mean_rounds[i, j], mean_rounds[i, j] + std_errors[i, j]], 
                            'k-', alpha=0.6, linewidth=2)
    
    ax2.set_xlabel('Temperature')
    ax2.set_ylabel('Agents')
    ax2.set_zlabel('Mean Rounds to Convergence')
    ax2.set_title('Convergence Performance (3D)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot2_3d_convergence.pdf"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Count Heatmaps with actual values
    fig3, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Success count heatmap
    im1 = axes[0].imshow(converged_counts, aspect='auto', cmap='Greens', origin='lower', vmin=0, vmax=10)
    axes[0].set_title('Success Count')
    axes[0].set_xlabel('Temperature')
    axes[0].set_ylabel('Agents')
    plt.colorbar(im1, ax=axes[0])
    
    # Add numbers
    for i in range(n_agents):
        for j in range(n_temps):
            axes[0].text(j, i, f'{int(converged_counts[i, j])}', ha='center', va='center', fontsize=6)
    
    # Parsing failure count heatmap  
    im2 = axes[1].imshow(parsing_failure_counts, aspect='auto', cmap='Reds', origin='lower', vmin=0, vmax=10)
    axes[1].set_title('Parsing Failure Count')
    axes[1].set_xlabel('Temperature')
    axes[1].set_ylabel('Agents')
    plt.colorbar(im2, ax=axes[1])
    
    # Add numbers
    for i in range(n_agents):
        for j in range(n_temps):
            axes[1].text(j, i, f'{int(parsing_failure_counts[i, j])}', ha='center', va='center', fontsize=6)
    
    # No convergence count heatmap
    im3 = axes[2].imshow(not_converged_counts, aspect='auto', cmap='Oranges', origin='lower', vmin=0, vmax=10)
    axes[2].set_title('No Convergence Count')
    axes[2].set_xlabel('Temperature')
    axes[2].set_ylabel('Agents')
    plt.colorbar(im3, ax=axes[2])
    
    # Add numbers
    for i in range(n_agents):
        for j in range(n_temps):
            axes[2].text(j, i, f'{int(not_converged_counts[i, j])}', ha='center', va='center', fontsize=6)
    
    # Add temperature and agent labels
    for ax in axes:
        ax.set_xticks(range(0, len(temp_list), 5))
        ax.set_xticklabels([f'{temp_list[i]:.1f}' for i in range(0, len(temp_list), 5)])
        ax.set_yticks(range(0, len(agents_list), 2))
        ax.set_yticklabels([agents_list[i] for i in range(0, len(agents_list), 2)])
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot3_count_heatmaps.pdf"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: All agent counts in one figure - stacked bars
    fig4, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()
    
    for agent_idx, agents in enumerate(agents_list):
        ax = axes[agent_idx]
        
        x = np.arange(len(temp_list))
        success_data = converged_counts[agent_idx, :]
        no_conv_data = not_converged_counts[agent_idx, :]
        parsing_data = parsing_failure_counts[agent_idx, :]
        
        ax.bar(x, success_data, label='Success', color='green', alpha=0.8)
        ax.bar(x, no_conv_data, bottom=success_data, label='No Convergence', color='orange', alpha=0.8)
        ax.bar(x, parsing_data, bottom=success_data + no_conv_data, label='Parsing Failure', color='red', alpha=0.8)
        
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Count')
        ax.set_title(f'{agents} Agents')
        ax.set_xticks(range(0, len(temp_list), 5))
        ax.set_xticklabels([f'{temp_list[i]:.1f}' for i in range(0, len(temp_list), 5)], rotation=45)
        ax.set_ylim(0, 10)
        if agent_idx == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot4_all_agents_stacked_bars.pdf"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 5: Single large convergence table with mean ± std
    fig5 = plt.figure(figsize=(16, 12))
    ax = fig5.add_subplot(111)
    
    # Create combined mean ± std values
    combined_values = np.full((n_agents, n_temps), '', dtype=object)
    for i in range(n_agents):
        for j in range(n_temps):
            if mean_rounds[i, j] > 0:
                combined_values[i, j] = f'{mean_rounds[i, j]:.1f}±{std_errors[i, j]:.2f}'
    
    # Create heatmap based on mean values
    mean_rounds_masked = np.where(mean_rounds > 0, mean_rounds, np.nan)
    im = ax.imshow(mean_rounds_masked, aspect='auto', cmap='Blues', origin='lower')
    ax.set_title('Mean Rounds to Convergence ± Standard Error', fontsize=16)
    ax.set_xlabel('Temperature', fontsize=14)
    ax.set_ylabel('Number of Agents', fontsize=14)
    plt.colorbar(im, ax=ax)
    
    # Add text annotations
    for i in range(n_agents):
        for j in range(n_temps):
            if combined_values[i, j]:
                ax.text(j, i, combined_values[i, j], ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Add temperature and agent labels
    ax.set_xticks(range(len(temp_list)))
    ax.set_xticklabels([f'{temp:.1f}' for temp in temp_list], rotation=45)
    ax.set_yticks(range(len(agents_list)))
    ax.set_yticklabels([str(agents) for agents in agents_list])
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot5_convergence_mean_std.pdf"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 6: Convergence by agent count with error bars and sample size coloring
    fig6, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()
    
    for agent_idx, agents in enumerate(agents_list):
        ax = axes[agent_idx]
        
        # Get data for this agent count
        x_temps = []
        y_means = []
        y_stds = []
        colors = []
        
        for j, temp in enumerate(temp_list):
            if mean_rounds[agent_idx, j] > 0:  # Only include successful runs
                x_temps.append(temp)
                y_means.append(mean_rounds[agent_idx, j])
                y_stds.append(std_errors[agent_idx, j])
                # Color based on number of successful runs (1-10)
                n_runs = successful_runs[agent_idx, j]
                colors.append(n_runs / 10.0)  # Normalize to 0-1 for colormap
        
        if x_temps:  # Only plot if we have data
            scatter = ax.errorbar(x_temps, y_means, yerr=y_stds, fmt='o', capsize=5, capthick=2, linewidth=2)
            # Color the points based on sample size
            points = ax.scatter(x_temps, y_means, c=colors, cmap='viridis', s=50, zorder=5, vmin=0.1, vmax=1.0)
        
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Mean Rounds to Convergence')
        ax.set_title(f'{agents} Agents')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        # Add colorbar for the first subplot to show sample size legend
        if agent_idx == 0 and x_temps:
            cbar = plt.colorbar(points, ax=ax)
            cbar.set_label('Fraction of Successful Runs (out of 10)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot6_convergence_by_agents.pdf"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plots in {plots_dir}:")
    print(f"  - plot1_3d_proportions.pdf")
    print(f"  - plot2_3d_convergence.pdf") 
    print(f"  - plot3_count_heatmaps.pdf (actual counts with numbers)")
    print(f"  - plot4_all_agents_stacked_bars.pdf (all 19 agents in one figure)")
    print(f"  - plot5_convergence_mean_std.pdf (mean ± std in one table)")
    print(f"  - plot6_convergence_by_agents.pdf (convergence trends with sample sizes)")

# Usage - UPDATE THIS PATH
base_path = "results/massive_experiment_meta-llama_llama-3.3-70b-instruct_20250703_221915"
results = collect_results(base_path)
create_plots(results, base_path)