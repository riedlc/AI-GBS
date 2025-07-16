import os
import re
import csv

def extract_round_data(content):
    """Extract round data from game log content."""
    rounds_data = []
    pattern = r"Round (\d+): guesses=\[(.*?)\]"
    
    for match in re.finditer(pattern, content):
        round_num = int(match.group(1))
        guesses_str = match.group(2)
        guesses = [int(x.strip()) for x in guesses_str.split(',')]
        
        # Create row: [round_number, guess1, guess2, ...]
        row = [round_num] + guesses
        rounds_data.append(row)
    
    # Sort by round number
    rounds_data.sort(key=lambda x: x[0])
    return rounds_data

def save_to_csv(rounds_data, filename, exp_name, run_name, success_type):
    """Save rounds data to CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        num_agents = len(rounds_data[0]) - 1 if rounds_data else 10
        header = ['round'] + [f'agent_{i+1}' for i in range(num_agents)]
        writer.writerow(header)
        
        # Data rows
        writer.writerows(rounds_data)
    
    print(f"Saved: {filename} ({len(rounds_data)} rounds) - {exp_name}/{run_name} ({success_type})")

def extract_game_data(base_dir, experiment_names):
    """Extract game data from successful and unsuccessful experiments and save as CSV files."""
    
    for exp_name in experiment_names:
        print(f"\nProcessing experiment: {exp_name}")
        
        found_success = False
        found_failure = False
        run_num = 1
        
        # Search through runs until we find both types or exhaust runs
        while (not found_success or not found_failure) and run_num <= 20:
            run_name = f"run_{run_num:03d}"
            game_log_path = os.path.join(base_dir, exp_name, "sum_a10_t0.7", run_name, "game_log.txt")
            
            if not os.path.exists(game_log_path):
                run_num += 1
                continue
            
            # Read the game log
            try:
                with open(game_log_path, 'r') as f:
                    content = f.read()
            except Exception as e:
                print(f"Error reading {game_log_path}: {e}")
                run_num += 1
                continue
            
            # Extract round data
            rounds_data = extract_round_data(content)
            
            if not rounds_data:
                print(f"No round data found in {exp_name}/{run_name}")
                run_num += 1
                continue
            
            # Check if experiment was successful
            is_successful = "CORRECT" in content
            
            if is_successful and not found_success:
                # Save successful run
                csv_filename = f"{exp_name.replace('massive_experiment_', '')}_SUCCESS.csv"
                save_to_csv(rounds_data, csv_filename, exp_name, run_name, "SUCCESS")
                found_success = True
                
            elif not is_successful and not found_failure:
                # Save unsuccessful run
                csv_filename = f"{exp_name.replace('massive_experiment_', '')}_FAILURE.csv"
                save_to_csv(rounds_data, csv_filename, exp_name, run_name, "FAILURE")
                found_failure = True
            
            run_num += 1
        
        # Report what was found
        if not found_success:
            print(f"No successful runs found for {exp_name}")
        if not found_failure:
            print(f"No unsuccessful runs found for {exp_name}")

# Usage
experiment_names = [
    "massive_experiment_google_gemini-2.5-flash-preview-05-20_20250703_221915",
    "massive_experiment_meta-llama_llama-3.3-70b-instruct_20250703_221915", 
    "massive_experiment_openai_gpt-4o-mini_20250702_182448"
]

# Set your base directory path here
base_directory = "./results"  # Current directory, adjust as needed

extract_game_data(base_directory, experiment_names)