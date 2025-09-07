from typing import List, Dict
from experiment import GameMaster, ParsingError, Agent
from persona_wrapper import PersonaWrapper
from datetime import datetime
import os
import asyncio
import json
import time
import sys
from llm_run import chat

class PersonaAgent(Agent):
    
    def __init__(self, agent_id: int, model: str, temperature: float, persona_wrapper: PersonaWrapper):
        super().__init__(agent_id, model, temperature)
        self.persona_wrapper = persona_wrapper
    
    async def make_guess(self, round_num: int, game_history, guess_range: tuple, mode: str):
        # Get original prompt using parent class logic
        if round_num == 1:
            game_desc = "sum guessing game" if mode == "sum" else "number guessing game"
            target_desc = "sum to the mystery number" if mode == "sum" else "average to the mystery number"
            original_prompt = f"""You are playing a {game_desc}. Your goal is to help your group {target_desc}.
Your guess range is {guess_range[0]} to {guess_range[1]}.

Game History:
    This is the first round and you have not made any guesses yet.
Result: There is no feedback on the group guess yet

Based on this information, what should your first guess be?

Respond with only an integer between {guess_range[0]} and {guess_range[1]}."""
        else:
            original_prompt = self._build_strategic_prompt(game_history, guess_range, mode)
        
        # Enhance with persona
        enhanced_prompt = self.persona_wrapper.enhance_prompt(self.agent_id, original_prompt)
        
        # Use parent class logic for the rest
        from prompt_capture import capture_prompt
        capture_prompt(round_num, self.agent_id, enhanced_prompt)
        
        response = await chat(
            model=self.model,
            prompt=enhanced_prompt,
            temperature=self.temperature,
            max_tokens=2
        )
        
        try:
            guess = self._extract_number(response, guess_range)
            self.guess_history.append(guess)
            self.last_successful_guess = guess
            return guess, enhanced_prompt, response, False
        except Exception as e:
            # For research: let parsing failures propagate naturally
            raise e

class PersonaGameMaster(GameMaster):
    """GameMaster with persona support - inherits from original GameMaster"""
    
    def __init__(self, mode: str = "mean", mystery_range: tuple = None, temperature: float = 0.7, 
                 max_rounds: int = 20, num_agents: int = None, batch_folder: str = None, run_id: int = 1,
                 persona_wrapper: PersonaWrapper = None):
        # Use exact same parameters as original GameMaster
        super().__init__(mode, mystery_range, temperature, max_rounds, num_agents, batch_folder, run_id)
        self.persona_wrapper = persona_wrapper
    
    def add_agent(self, model: str) -> PersonaAgent:
        """Add a persona agent to the game"""
        agent_id = len(self.agents)
        agent = PersonaAgent(agent_id, model, self.temperature, self.persona_wrapper)
        self.agents.append(agent)
        return agent
    
    def _save_config(self):
        config = {
            "mystery_number": self.mystery_number,
            "mystery_range": self.mystery_range,
            "guess_range": self.guess_range,
            "mode": self.mode,
            "temperature": self.temperature,
            "max_rounds": self.max_rounds,
            "timestamp": datetime.now().isoformat(),
            "agents": [
                {
                    "agent_id": i,
                    "model": agent.model,
                    "client_type": "openai",
                    "persona": self.persona_wrapper.agent_personas.get(i) if self.persona_wrapper else None
                }
                for i, agent in enumerate(self.agents)
            ]
        }
        
        with open(os.path.join(self.results_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)

async def run_single_config(agents, temp, run_id, batch_folder, model, client_type, mode, max_rounds):
    config_id = f"{agents}a_t{temp:.1f}_r{run_id}"
    print(f"    🚀 Starting: {config_id}")
    
    try:
        # Setup personas for this simulation
        persona_wrapper = PersonaWrapper("personas_gpt41.txt")
        persona_wrapper.assign_personas(agents)
        
        game = PersonaGameMaster(mode=mode, temperature=temp, max_rounds=max_rounds, 
                               num_agents=agents, batch_folder=batch_folder, run_id=run_id,
                               persona_wrapper=persona_wrapper)
        
        # Add agents
        for i in range(agents):
            game.add_agent(model)
        
        await game.play_game()
        
        print(f"    ✅ Completed: {config_id}")
        return {"status": "success", "config": config_id, "agents": agents, "temp": temp, "run_id": run_id}
        
    except ParsingError as e:
        print(f"    🔤 Parsing Failed: {config_id} - {str(e)[:100]}")
        return {"status": "parsing_failed", "config": config_id, "agents": agents, "temp": temp, "run_id": run_id, "error": str(e)}
        
    except Exception as e:
        print(f"    ❌ Failed: {config_id} - {str(e)[:100]}")
        return {"status": "failed", "config": config_id, "agents": agents, "temp": temp, "run_id": run_id, "error": str(e)}

def create_fallback_result(config_id, agents, temp, run_id, error_msg):
    """Create a fallback result when game completely fails"""
    return {
        "status": "failed_with_fallback",
        "config": config_id,
        "agents": agents,
        "temp": temp,
        "run_id": run_id,
        "error": error_msg,
        "fallback_data": {
            "total_rounds": 0,
            "solved": False,
            "completed_successfully": False,
            "note": "Game failed completely, using fallback data"
        }
    }

def create_config_batches(agents_list, temp_list, runs_per_config, batch_size=20):
    """Create batches of configurations to run"""
    all_configs = []
    
    for agents in agents_list:
        for temp in temp_list:
            for run_id in range(1, runs_per_config + 1):
                all_configs.append((agents, temp, run_id))
    
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
    progress_file = os.path.join(batch_folder, "progress.json")
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return None

async def run_batch(batch_configs, batch_num, total_batches, batch_folder, model, client_type, mode, max_rounds, max_concurrent=10):
    """Run a single batch of configurations"""
    batch_start_time = time.time()
    
    print(f"\n📦 BATCH {batch_num}/{total_batches}")
    print(f"🔢 Configs in this batch: {len(batch_configs)}")
    print(f"⚡ Max concurrent: {max_concurrent}")
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def limited_run_config(config):
        async with semaphore:
            agents, temp, run_id = config
            return await run_single_config(agents, temp, run_id, batch_folder, model, client_type, mode, max_rounds)
    
    # Run all configs in this batch
    batch_results = await asyncio.gather(
        *[limited_run_config(config) for config in batch_configs],
        return_exceptions=True
    )
    
    # Handle any exceptions
    processed_results = []
    for result in batch_results:
        if isinstance(result, Exception):
            processed_results.append({
                "status": "failed", 
                "config": "unknown", 
                "error": str(result)
            })
        else:
            processed_results.append(result)
    
    # Calculate batch statistics
    successful = sum(1 for r in processed_results if r["status"] == "success")
    failed = sum(1 for r in processed_results if r["status"] == "failed")
    parsing_failed = sum(1 for r in processed_results if r["status"] == "parsing_failed")
    batch_duration = time.time() - batch_start_time
    
    print(f"  ✅ Successful: {successful}/{len(batch_configs)}")
    print(f"  ❌ Failed: {failed}/{len(batch_configs)}")
    print(f"  🔤 Parsing Failed: {parsing_failed}/{len(batch_configs)}")
    print(f"  ⏱️  Duration: {batch_duration:.1f}s")
    
    # Save progress
    progress = save_progress(batch_folder, batch_num, processed_results, total_batches)
    
    return processed_results

async def main():
    # Parse command line arguments
    if len(sys.argv) != 2:
        print("Usage: python persona_experiment.py <model_name>")
        print("Example: python persona_experiment.py 'openai/gpt-4o-mini'")
        sys.exit(1)
    
    model = sys.argv[1]
    
    # ===== EXPERIMENT PARAMETERS =====
    client_type = "openai"
    mode = "sum"  # "sum" or "mean"
    # max_rounds = 20
    # max_rounds = 5
    
    # ===== EXPERIMENT SETTINGS =====
    # agents_list = list(range(2, 21))                    # 2 to 20
    # temp_list = [round(0.1 * i, 1) for i in range(0, 11)]  # 0.0 to 1.0
    # runs_per_config = 50

    # #checking if code works 
    # agents_list = [2,3]
    # temp_list = [0.7]
    # runs_per_config = 3

    max_rounds = 20
    runs_per_config = 100
    # agents_list = list(range(11, 11)) # just one experiment with N=10
    agents_list = [10]
    temp_list = [1.0]
    
    # ===== BATCHING SETTINGS =====
    batch_size = 20          # Configs per batch
    max_concurrent = 8       # Max concurrent configs within each batch
    resume_from_batch = None # Set to batch number to resume from, or None to start fresh
    
    # Calculate totals
    total_configs = len(agents_list) * len(temp_list) * runs_per_config
    model_safe_name = model.replace("/", "_").replace(":", "_")
    
    print(f"🎯 PERSONA EXPERIMENT SETUP:")
    print(f"   🤖 Model: {model}")
    print(f"   🎭 Using personas for all agents")
    print(f"   Agent counts: {len(agents_list)} ({min(agents_list)} to {max(agents_list)})")
    print(f"   Temperatures: {len(temp_list)} ({min(temp_list):.1f} to {max(temp_list):.1f})")
    print(f"   Runs per config: {runs_per_config}")
    print(f"   📊 TOTAL CONFIGS: {total_configs}")
    print(f"   📦 Batch size: {batch_size}")
    print(f"   ⚡ Max concurrent per batch: {max_concurrent}")
    
    # Create batch directory with model name
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_folder = f"results/persona_experiment_{model_safe_name}_{batch_timestamp}"
    os.makedirs(batch_folder, exist_ok=True)
    
    # Save experiment configuration
    experiment_config = {
        "model": model,
        "agents_list": agents_list,
        "temp_list": temp_list,
        "runs_per_config": runs_per_config,
        "total_configs": total_configs,
        "batch_size": batch_size,
        "max_concurrent": max_concurrent,
        "client_type": client_type,
        "mode": mode,
        "max_rounds": max_rounds,
        "experiment_type": "persona_experiment",
        "persona_file": "personas_gpt41.txt",
        "start_time": datetime.now().isoformat()
    }
    
    with open(os.path.join(batch_folder, "experiment_config.json"), 'w') as f:
        json.dump(experiment_config, f, indent=2)
    
    print(f"📁 Results folder: {batch_folder}")
    
    # Create batches
    print(f"\n🔄 Creating batches...")
    config_batches = create_config_batches(agents_list, temp_list, runs_per_config, batch_size)
    total_batches = len(config_batches)
    print(f"📦 Created {total_batches} batches")
    
    # Check for existing progress
    existing_progress = load_progress(batch_folder)
    start_batch = 1
    
    if existing_progress and resume_from_batch:
        start_batch = resume_from_batch
        print(f"🔄 Resuming from batch {start_batch}")
    elif existing_progress:
        completed_batches = len(existing_progress["batches_completed"])
        if completed_batches < total_batches:
            start_batch = completed_batches + 1
            print(f"🔄 Found existing progress. Resuming from batch {start_batch}")
        else:
            print(f"✅ All batches already completed!")
            return
    
    # Run batches sequentially
    experiment_start_time = time.time()
    
    for batch_num in range(start_batch, total_batches + 1):
        batch_configs = config_batches[batch_num - 1]  # Convert to 0-based index
        
        try:
            await run_batch(
                batch_configs, batch_num, total_batches, 
                batch_folder, model, client_type, mode, max_rounds, max_concurrent
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
            print(f"💾 Progress saved. Resume with: resume_from_batch = {batch_num}")
            break
        except Exception as e:
            print(f"❌ Error in batch {batch_num}: {e}")
            print(f"💾 Progress saved. You can resume from this batch.")
            continue
    
    # Final summary
    final_progress = load_progress(batch_folder)
    if final_progress:
        total_successful = sum(batch["successful"] for batch in final_progress["batches_completed"])
        total_failed = sum(batch["failed"] for batch in final_progress["batches_completed"])
        total_parsing_failed = sum(batch["parsing_failed"] for batch in final_progress["batches_completed"])
        total_duration = time.time() - final_progress["start_time"]
        
        print(f"\n🎉 PERSONA EXPERIMENT COMPLETED!")
        print(f"🤖 Model: {model}")
        print(f"🎭 Experiment Type: Persona-enhanced agents")
        print(f"📊 Final Results:")
        print(f"   ✅ Successful configs: {total_successful}")
        print(f"   ❌ Failed configs: {total_failed}")
        print(f"   🔤 Parsing failed configs: {total_parsing_failed}")
        print(f"   📦 Batches completed: {len(final_progress['batches_completed'])}/{total_batches}")
        print(f"   ⏱️  Total duration: {total_duration/3600:.1f} hours")
        print(f"📁 Results saved in: {batch_folder}")

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