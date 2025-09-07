import random
import sys
import json
import os
import time
import csv
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict
from llm_run import chat
from reasoning_capture import (
    init_reasoning_capture,
    capture_reasoning as capture_reasoning_trace,
    save_and_display_reasoning,
)
import asyncio
import re
from prompt_capture import capture_prompt
from persona_wrapper import PersonaWrapper
from experiment import GameMaster, ParsingError, Agent, Round

class CleanInterventionAgent(Agent):
    
    def __init__(self, agent_id: int, model: str, temperature: float, intervention_type: str, persona_wrapper: PersonaWrapper = None):
        super().__init__(agent_id, model, temperature)
        self.intervention_type = intervention_type
        self.persona_wrapper = persona_wrapper
    
    async def make_guess(self, round_num: int, game_history, guess_range: tuple, mode: str):
        # Get original prompt using parent class logic
        if round_num == 1:
            game_desc = "sum guessing game" if mode == "sum" else "number guessing game"
            target_desc = "sum to the mystery number" if mode == "sum" else "average to the mystery number"
            
            # Clean base prompt for all interventions
            original_prompt = f"""You are playing a {game_desc}. Your goal is to help your group {target_desc}.
Your guess range is {guess_range[0]} to {guess_range[1]}.

Game History:
    This is the first round and you have not made any guesses yet.
Result: There is no feedback on the group guess yet

What is your guess this round? Always start with the efficient strategy in guessing games which is to use a binary search approach: guessing the midpoint of the current range. Always anchor your guess on the group feedback from previous rounds (too high / too low).

End your answer with: FINAL GUESS: [0-50]"""
        else:
            original_prompt = self._build_strategic_prompt(game_history, guess_range, mode)
        
        # Apply intervention based on type
        if self.intervention_type == "plain4":
            enhanced_prompt = original_prompt  # No enhancement
        elif self.intervention_type == "persona4":
            # Add persona only (no ToM reasoning)
            enhanced_prompt = self.persona_wrapper.enhance_prompt(self.agent_id, original_prompt)
        elif self.intervention_type == "tom4":
            # Add persona first
            enhanced_prompt = self.persona_wrapper.enhance_prompt(self.agent_id, original_prompt)
            # Then add the two yellow sentences (CoT reasoning & adaptation)
            tom4_text = "Only as a secondary approach, carefully think through step-by-step what others might guess and how the contributions of others contribute to the sum of the group guesses for the mystery number. Consider what roles other agents might be playing (e.g., guessing higher or lower) and adapt your own adjustment to complement the group. "
            
            # More robust replacement - insert before the target sentence
            target_sentence = "Always anchor your guess on the group feedback from previous rounds (too high / too low)."
            if target_sentence in enhanced_prompt:
                enhanced_prompt = enhanced_prompt.replace(
                    target_sentence,
                    f"{tom4_text}{target_sentence}"
                )
            else:
                # Fallback: append to the end if target sentence not found
                enhanced_prompt = enhanced_prompt.replace(
                    "End your answer with: FINAL GUESS: [0-50]",
                    f"{tom4_text}End your answer with: FINAL GUESS: [0-50]"
                )
        else:
            raise ValueError(f"Unknown intervention type: {self.intervention_type}")
        
        # Use parent class logic for the rest
        capture_prompt(round_num, self.agent_id, enhanced_prompt)
        
        response = await chat(
            model=self.model,
            prompt=enhanced_prompt,
            temperature=self.temperature,
            max_tokens=200
        )
        
        try:
            guess = self._extract_number(response, guess_range)
            self.guess_history.append(guess)
            self.last_successful_guess = guess
            # Capture full reasoning/response if enabled
            try:
                response_text = self._get_response_content(response)
                capture_reasoning_trace(round_num, self.agent_id, enhanced_prompt, response_text, guess)
            except Exception:
                pass
            
            return guess, enhanced_prompt, response, False
        except Exception as e:
            raise e

    def _build_strategic_prompt(self, game_history: List[Round], guess_range: tuple, mode: str) -> str:
        """Build clean base prompt for all interventions"""
        
        game_desc = "sum guessing game" if mode == "sum" else "number guessing game"
        target_desc = "sum to the mystery number" if mode == "sum" else "average to the mystery number"
        
        prompt = f"""You are playing a {game_desc}. Your goal is to help your group {target_desc}.
Your guess range is {guess_range[0]} to {guess_range[1]}.

Game History:
"""
        
        for round_data in game_history:
            prompt += f"Round {round_data.round_num}: "
            prompt += f"Your guess: {round_data.guesses[self.agent_id]}\n"
            prompt += f"Result: {round_data.feedback}\n\n"
        
        # Clean base prompt - interventions will be applied by the wrapper
        prompt += f"""What is your guess this round? Always start with the efficient strategy in guessing games which is to use a binary search approach: guessing the midpoint of the current range. Always anchor your guess on the group feedback from previous rounds (too high / too low).

End your answer with: FINAL GUESS: [0-50]"""
        
        return prompt

    def _get_response_content(self, response) -> str:
        # Handle OpenAI-like and our Localhost/Ollama adapters
        try:
            if hasattr(response, "choices") and response.choices:
                msg = getattr(response.choices[0], "message", None)
                if msg and hasattr(msg, "content"):
                    return msg.content
            # Fallback to str
            return str(response)
        except Exception:
            return str(response)

class CleanInterventionGameMaster(GameMaster):
    """GameMaster with clean intervention support"""
    
    def __init__(self, mode: str = "sum", mystery_range: tuple = None, temperature: float = 0.7, 
                 max_rounds: int = 20, num_agents: int = None, batch_folder: str = None, run_id: int = 1,
                 intervention_type: str = "plain4"):
        super().__init__(mode=mode, mystery_range=mystery_range, temperature=temperature, 
                        max_rounds=max_rounds, num_agents=num_agents, batch_folder=batch_folder, run_id=run_id)
        self.intervention_type = intervention_type
        
        # Setup persona wrapper if needed for Persona4 and ToM4
        if intervention_type in ["persona4", "tom4"]:
            self.persona_wrapper = PersonaWrapper("personas_gpt41.txt")
            self.persona_wrapper.assign_personas(num_agents)
        else:
            self.persona_wrapper = None
    
    def add_agent(self, model: str) -> CleanInterventionAgent:
        """Add a clean intervention agent to the game"""
        agent_id = len(self.agents)
        agent = CleanInterventionAgent(agent_id, model, self.temperature, self.intervention_type, self.persona_wrapper)
        self.agents.append(agent)
        return agent

async def run_single_config(agents, temp, run_id, batch_folder, model, client_type, mode, max_rounds, intervention_type):
    """Run a single configuration asynchronously"""
    config_id = f"{agents}a_t{temp:.1f}_r{run_id}_{intervention_type}"
    print(f"    🚀 Starting: {config_id}")
    
    try:
        game = CleanInterventionGameMaster(mode=mode, temperature=temp, max_rounds=max_rounds, 
                                         num_agents=agents, batch_folder=batch_folder, run_id=run_id,
                                         intervention_type=intervention_type)
        for i in range(agents):
            game.add_agent(model)
        # Initialize reasoning capture for this game's results directory
        try:
            if hasattr(game, "results_dir"):
                init_reasoning_capture(game.results_dir)
        except Exception:
            pass
        
        await game.play_game()
        
        # Save reasoning traces (if any)
        try:
            save_and_display_reasoning()
        except Exception:
            pass
        
        print(f"    ✅ Completed: {config_id}")
        return {"status": "success", "config": config_id, "agents": agents, "temp": temp, "run_id": run_id, "intervention_type": intervention_type}
        
    except ParsingError as e:
        print(f"    🔤 Parsing Failed: {config_id} - {str(e)[:100]}")
        return {"status": "parsing_failed", "config": config_id, "agents": agents, "temp": temp, "run_id": run_id, "intervention_type": intervention_type, "error": str(e)}
        
    except Exception as e:
        print(f"    ❌ Failed: {config_id} - {str(e)[:100]}")
        return create_fallback_result(config_id, agents, temp, run_id, intervention_type, str(e))

def create_fallback_result(config_id, agents, temp, run_id, intervention_type, error_msg):
    """Create a fallback result when game completely fails"""
    return {
        "status": "failed_with_fallback",
        "config": config_id,
        "agents": agents,
        "temp": temp,
        "run_id": run_id,
        "intervention_type": intervention_type,
        "error": error_msg,
        "fallback_data": {
            "total_rounds": 0,
            "solved": False,
            "completed_successfully": False,
            "note": "Game failed completely, using fallback data"
        }
    }

def create_config_batches(agents_list, temp_list, runs_per_config, intervention_types, batch_size=20):
    """Create batches of configurations to run"""
    all_configs = []
    
    for agents in agents_list:
        for temp in temp_list:
            for intervention_type in intervention_types:
                for run_id in range(1, runs_per_config + 1):
                    all_configs.append((agents, temp, run_id, intervention_type))
    
    # Split into batches
    batches = []
    for i in range(0, len(all_configs), batch_size):
        batches.append(all_configs[i:i + batch_size])
    
    return batches

def save_progress(batch_folder, batch_num, batch_results, total_batches):
    """Save progress to file"""
    progress_file = os.path.join(batch_folder, "progress.json")
    
    # Load existing progress
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
    else:
        progress = {"batches_completed": [], "total_batches": total_batches, "start_time": time.time()}
    
    # Add current batch results
    batch_summary = {
        "batch_num": batch_num,
        "total_configs": len(batch_results),
        "successful": sum(1 for r in batch_results if r["status"] == "success"),
        "failed": sum(1 for r in batch_results if r["status"] == "failed"),
        "parsing_failed": sum(1 for r in batch_results if r["status"] == "parsing_failed"),
        "failed_with_fallback": sum(1 for r in batch_results if r["status"] == "failed_with_fallback"),
        "completion_time": time.time(),
        "results": batch_results
    }
    
    progress["batches_completed"].append(batch_summary)
    
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)
    
    return progress

def load_progress(batch_folder):
    """Load existing progress from file"""
    progress_file = os.path.join(batch_folder, "progress.json")
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return None

def find_latest_experiment_folder(intervention_type, model_safe_name):
    """Find the most recent experiment folder with progress for the given intervention and model"""
    results_dir = "results"
    if not os.path.exists(results_dir):
        return None
    
    # Look for folders matching the pattern
    pattern = f"{intervention_type}_experiment_{model_safe_name}_"
    matching_folders = []
    
    for folder in os.listdir(results_dir):
        if folder.startswith(pattern) and os.path.isdir(os.path.join(results_dir, folder)):
            # Check if this folder has progress
            progress_file = os.path.join(results_dir, folder, "progress.json")
            if os.path.exists(progress_file):
                matching_folders.append(folder)
    
    if not matching_folders:
        return None
    
    # Sort by modification time (most recent first)
    matching_folders.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
    return os.path.join(results_dir, matching_folders[0])

def migrate_to_resumable(intervention_type, model_safe_name):
    """Migrate latest experiment to resumable format"""
    latest_folder = find_latest_experiment_folder(intervention_type, model_safe_name)
    if not latest_folder:
        return None
    
    resumable_folder = f"results/{intervention_type}_experiment_{model_safe_name}_resumable"
    
    # Check if latest folder has progress
    progress = load_progress(latest_folder)
    if not progress:
        return None
    
    # Copy progress to resumable folder
    os.makedirs(resumable_folder, exist_ok=True)
    progress_file = os.path.join(resumable_folder, "progress.json")
    
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)
    
    print(f"📋 Migrated progress from {latest_folder} to {resumable_folder}")
    return resumable_folder

async def run_batch(batch_configs, batch_num, total_batches, batch_folder, model, client_type, mode, max_rounds, max_concurrent, intervention_types):
    """Run a batch of configurations concurrently"""
    print(f"\n📦 Batch {batch_num}/{total_batches} - {len(batch_configs)} configs")
    
    # Create semaphore to limit concurrent executions
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def run_with_semaphore(config):
        async with semaphore:
            agents, temp, run_id, intervention_type = config
            return await run_single_config(agents, temp, run_id, batch_folder, model, client_type, mode, max_rounds, intervention_type)
    
    # Run all configs in the batch concurrently
    tasks = [run_with_semaphore(config) for config in batch_configs]
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions that occurred
    processed_results = []
    for i, result in enumerate(batch_results):
        if isinstance(result, Exception):
            config = batch_configs[i]
            error_result = create_fallback_result(config[0], config[1], config[2], config[3], str(result))
            processed_results.append(error_result)
        else:
            processed_results.append(result)
    
    # Save progress
    save_progress(batch_folder, batch_num, processed_results, total_batches)
    
    # Display batch summary
    successful = sum(1 for r in processed_results if r["status"] == "success")
    failed = sum(1 for r in processed_results if r["status"] == "failed")
    parsing_failed = sum(1 for r in processed_results if r["status"] == "parsing_failed")
    failed_with_fallback = sum(1 for r in processed_results if r["status"] == "failed_with_fallback")
    
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   🔤 Parsing failed: {parsing_failed}")
    print(f"   ⚠️  Failed with fallback: {failed_with_fallback}")

def save_failure_summary(batch_folder, progress):
    """Analyze and save parsing failure summary"""
    all_results = []
    for batch in progress["batches_completed"]:
        all_results.extend(batch["results"])
    
    # Count failures by type
    total_configs = len(all_results)
    parsing_failures = [r for r in all_results if r["status"] == "parsing_failed"]
    parsing_failure_rate = len(parsing_failures) / total_configs if total_configs > 0 else 0
    
    # Group failures by intervention type
    failure_by_intervention = {}
    for failure in parsing_failures:
        intervention = failure.get("intervention_type", "unknown")
        if intervention not in failure_by_intervention:
            failure_by_intervention[intervention] = 0
        failure_by_intervention[intervention] += 1
    
    failure_analysis = {
        "total_configs": total_configs,
        "parsing_failures": len(parsing_failures),
        "parsing_failure_rate": parsing_failure_rate,
        "failure_by_intervention": failure_by_intervention,
        "all_failures": parsing_failures
    }
    
    with open(os.path.join(batch_folder, "parsing_failures_analysis.json"), 'w') as f:
        json.dump(failure_analysis, f, indent=2)
    
    return failure_analysis

async def main():
    """Main function for clean intervention experiments"""
    if len(sys.argv) < 2:
        print("Usage: python clean_intervention_experiment.py <model_name> [intervention_type]")
        print("Example: python clean_intervention_experiment.py 'openai/gpt-4o-mini'")
        print("Example: python clean_intervention_experiment.py 'openai/gpt-4o-mini' plain4")
        print("Example: python clean_intervention_experiment.py 'openai/gpt-4o-mini' persona4")
        print("Example: python clean_intervention_experiment.py 'openai/gpt-4o-mini' tom4")
        print("\nAvailable interventions:")
        print("  plain4  - Clean base prompt (no enhancements)")
        print("  persona4 - Clean base prompt + persona only")
        print("  tom4    - Clean base prompt + persona + ToM reasoning")
        sys.exit(1)
    
    model = sys.argv[1]
    intervention_type = sys.argv[2] if len(sys.argv) > 2 else "plain4"
    client_type = "openai"  # Default client type
    
    # Validate intervention type
    if intervention_type not in ["plain4", "persona4", "tom4"]:
        print(f"❌ Invalid intervention type: {intervention_type}")
        print("Available types: plain4, persona4, tom4")
        sys.exit(1)
    
    # ===== EXPERIMENT SETTINGS =====
    mode = "sum"  # "sum" or "mean"
   
    
    # agents_list = list(range(2, 21))                    # 2 to 20
    # temp_list = [round(0.1 * i, 1) for i in range(0, 11)]  # 0.0 to 1.0
    # runs_per_config = 50
    
    # For testing - single configuration
    max_rounds = 30
    agents_list = [10]
    temp_list = [1.0]
    runs_per_config = 220
    intervention_types = [intervention_type]  # Only the specified intervention
    
    # ===== BATCHING SETTINGS =====
    batch_size = 2          # Configs per batch
    max_concurrent = 1      # Max concurrent configs within each batch
    resume_from_batch = None # Set to batch number to resume from, or None for auto-resume
    force_fresh_start = True # Set to True to force a fresh start (ignore existing experiments)


    # To resume: set resume_from_batch = <batch_number> (e.g., resume_from_batch = 5)
    # To force fresh start: set force_fresh_start = True
    
    # Calculate totals
    total_configs = len(agents_list) * len(temp_list) * len(intervention_types) * runs_per_config
    model_safe_name = model.replace("/", "_").replace(":", "_")
    
    print(f"🎯 CLEAN INTERVENTION EXPERIMENT SETUP:")
    print(f"   🤖 Model: {model}")
    print(f"   🎭 Intervention: {intervention_type}")
    print(f"   Agent counts: {len(agents_list)} ({min(agents_list)} to {max(agents_list)})")
    print(f"   Temperatures: {len(temp_list)} ({min(temp_list):.1f} to {max(temp_list):.1f})")
    print(f"   Runs per config: {runs_per_config}")
    print(f"   📊 TOTAL CONFIGS: {total_configs}")
    print(f"   📦 Batch size: {batch_size}")
    print(f"   ⚡ Max concurrent per batch: {max_concurrent}")
    
    # Create batch directory with model name and intervention type
    if force_fresh_start:
        # Force fresh start - create new timestamped folder
        batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_folder = f"results/{intervention_type}_experiment_{model_safe_name}_{batch_timestamp}"
        print(f"🆕 Force fresh start - using new experiment folder: {batch_folder}")
        existing_progress = None
    else:
        # Try to find existing progress in resumable folder
        resumable_folder = f"results/{intervention_type}_experiment_{model_safe_name}_resumable"
        existing_progress = load_progress(resumable_folder)
        
        if existing_progress:
            # Use existing resumable folder
            batch_folder = resumable_folder
            print(f"🔄 Found existing progress in: {batch_folder}")
        else:
            # No resumable folder - check if there are any existing experiments to migrate
            latest_experiment = find_latest_experiment_folder(intervention_type, model_safe_name)
            if latest_experiment:
                # Found existing experiment - migrate it to resumable format
                migrated_folder = migrate_to_resumable(intervention_type, model_safe_name)
                if migrated_folder:
                    batch_folder = migrated_folder
                    existing_progress = load_progress(batch_folder)
                    print(f"🔄 Found and migrated existing experiment to: {batch_folder}")
                else:
                    # Migration failed - create new experiment
                    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    batch_folder = f"results/{intervention_type}_experiment_{model_safe_name}_{batch_timestamp}"
                    print(f"🆕 Migration failed, using new experiment folder: {batch_folder}")
            else:
                # No existing experiments - create new one
                batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                batch_folder = f"results/{intervention_type}_experiment_{model_safe_name}_{batch_timestamp}"
                print(f"🆕 No existing experiments found, using new experiment folder: {batch_folder}")
    
    os.makedirs(batch_folder, exist_ok=True)
    
    # Save experiment configuration
    experiment_config = {
        "model": model,
        "intervention_type": intervention_type,
        "agents_list": agents_list,
        "temp_list": temp_list,
        "runs_per_config": runs_per_config,
        "intervention_types": intervention_types,
        "total_configs": total_configs,
        "batch_size": batch_size,
        "max_concurrent": max_concurrent,
        "client_type": client_type,
        "mode": mode,
        "max_rounds": max_rounds,
        "experiment_type": f"{intervention_type}_experiment",
        "start_time": datetime.now().isoformat()
    }
    
    with open(os.path.join(batch_folder, "experiment_config.json"), 'w') as f:
        json.dump(experiment_config, f, indent=2)
    
    print(f"📁 Results folder: {batch_folder}")
    
    # Create batches
    print(f"\n🔄 Creating batches...")
    config_batches = create_config_batches(agents_list, temp_list, runs_per_config, intervention_types, batch_size)
    total_batches = len(config_batches)
    print(f"📦 Created {total_batches} batches")
    
    # Determine start batch based on existing progress
    start_batch = 1
    
    if existing_progress:
        completed_batches = len(existing_progress["batches_completed"])
        
        if resume_from_batch:
            # Manual resume from specific batch
            if resume_from_batch <= completed_batches:
                print(f"⚠️  Warning: Batch {resume_from_batch} already completed. Starting from next batch.")
                start_batch = completed_batches + 1
            else:
                start_batch = resume_from_batch
            print(f"🔄 Manual resume from batch {start_batch}")
        else:
            # Automatic resume from last completed batch + 1
            if completed_batches < total_batches:
                start_batch = completed_batches + 1
                print(f"🔄 Auto-resume from batch {start_batch} (last completed: {completed_batches})")
            else:
                print(f"✅ All batches already completed!")
                return
        
        # Show progress summary
        total_completed_configs = sum(batch["total_configs"] for batch in existing_progress["batches_completed"])
        print(f"📊 Progress: {completed_batches}/{total_batches} batches completed ({total_completed_configs} configs)")
    else:
        print(f"🆕 No existing progress found. Starting fresh from batch 1.")
    
    # Run batches sequentially
    experiment_start_time = time.time()
    
    for batch_num in range(start_batch, total_batches + 1):
        batch_configs = config_batches[batch_num - 1]  # Convert to 0-based index
        
        try:
            await run_batch(
                batch_configs, batch_num, total_batches, 
                batch_folder, model, client_type, mode, max_rounds, max_concurrent, intervention_types
            )
            
            # Calculate and display overall progress
            elapsed_time = time.time() - experiment_start_time
            progress_pct = (batch_num / total_batches) * 100
            
            if batch_num < total_batches:
                estimated_total_time = elapsed_time * (total_batches / batch_num)
                remaining_time = estimated_total_time - elapsed_time
                print(f"🎯 Overall Progress: {batch_num}/{total_batches} batches ({progress_pct:.1f}%)")
                print(f"⏱️  Elapsed: {elapsed_time/60:.1f}m | Estimated remaining: {remaining_time/60:.1f}m")
            
        except KeyboardInterrupt:
            print(f"\n🛑 Interrupted at batch {batch_num}")
            print(f"💾 Progress saved. To resume:")
            print(f"   1. Edit line 386: resume_from_batch = {batch_num}")
            print(f"   2. Run the experiment again")
            print(f"   📁 Progress file: {os.path.join(batch_folder, 'progress.json')}")
            break
        except Exception as e:
            print(f"❌ Error in batch {batch_num}: {e}")
            print(f"💾 Progress saved. You can resume from this batch.")
            # Continue to next batch instead of stopping
            continue
    
    # Final summary
    final_progress = load_progress(batch_folder)
    if final_progress:
        total_successful = sum(batch["successful"] for batch in final_progress["batches_completed"])
        total_failed = sum(batch["failed"] for batch in final_progress["batches_completed"])
        total_parsing_failed = sum(batch["parsing_failed"] for batch in final_progress["batches_completed"])
        total_duration = time.time() - final_progress["start_time"]
        
        # Generate parsing failure analysis
        failure_analysis = save_failure_summary(batch_folder, final_progress)
        
        print(f"\n🎉 EXPERIMENT COMPLETED!")
        print(f"🤖 Model: {model}")
        print(f"📊 Final Results:")
        print(f"   ✅ Successful configs: {total_successful}")
        print(f"   ❌ Failed configs: {total_failed}")
        print(f"   🔤 Parsing failed configs: {total_parsing_failed}")
        print(f"   📈 Parsing failure rate: {failure_analysis['parsing_failure_rate']:.1%}")
        print(f"   📦 Batches completed: {len(final_progress['batches_completed'])}/{total_batches}")
        print(f"   ⏱️  Total duration: {total_duration/3600:.1f} hours")
        print(f"📁 Results saved in: {batch_folder}")
        print(f"📋 Parsing analysis saved in: parsing_failures_analysis.json")

if __name__ == "__main__":
    import time
    total_start_time = time.time()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Experiment interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        total_end_time = time.time()
        total_elapsed = total_end_time - total_start_time
        print(f"\n🏁 Total program execution time: {total_elapsed:.2f} seconds ({total_elapsed/60:.1f} minutes)")
