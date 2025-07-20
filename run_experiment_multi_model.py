# Updated run_experiment.py to handle multiple models via command line
from experiment import GameMaster, ParsingError  # Import ParsingError
from datetime import datetime
import os
import asyncio
import json
import time
import sys

async def run_single_config(agents, temp, run_id, batch_folder, model, client_type, mode, max_rounds):
    """Run a single configuration asynchronously"""
    config_id = f"{agents}a_t{temp:.1f}_r{run_id}"
    print(f"    🚀 Starting: {config_id}")
    
    try:
        game = GameMaster(mode=mode, temperature=temp, max_rounds=max_rounds, 
                         num_agents=agents, batch_folder=batch_folder, run_id=run_id)
        for i in range(agents):
            game.add_agent(model, client_type)
        
        await game.play_game()
        
        print(f"    ✅ Completed: {config_id}")
        return {"status": "success", "config": config_id, "agents": agents, "temp": temp, "run_id": run_id}
        
    except ParsingError as e:
        print(f"    🔤 Parsing Failed: {config_id} - {str(e)[:100]}")
        return {"status": "parsing_failed", "config": config_id, "agents": agents, "temp": temp, "run_id": run_id, "error": str(e)}
        
    except Exception as e:
        print(f"    ❌ Failed: {config_id} - {str(e)[:100]}")
        return {"status": "failed", "config": config_id, "agents": agents, "temp": temp, "run_id": run_id, "error": str(e)}

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
        "parsing_failed": sum(1 for r in batch_results if r["status"] == "parsing_failed"),  # Track parsing failures separately
        "completion_time": time.time(),
        "results": batch_results
    }
    
    progress["batches_completed"].append(batch_summary)
    
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)
    
    return progress

def save_parsing_failure_summary(batch_folder, progress):
    """Save summary of all parsing failures for analysis"""
    parsing_failures = []
    
    for batch in progress["batches_completed"]:
        for result in batch["results"]:
            if result["status"] == "parsing_failed":
                parsing_failures.append({
                    "batch_num": batch["batch_num"],
                    "config": result["config"],
                    "agents": result["agents"],
                    "temp": result["temp"],
                    "run_id": result["run_id"],
                    "error": result["error"]
                })
    
    failure_summary = {
        "total_parsing_failures": len(parsing_failures),
        "failure_rate": len(parsing_failures) / sum(batch["total_configs"] for batch in progress["batches_completed"]) if sum(batch["total_configs"] for batch in progress["batches_completed"]) > 0 else 0,
        "failures_by_agent_count": {},
        "failures_by_temperature": {},
        "detailed_failures": parsing_failures
    }
    
    # Analyze patterns
    for failure in parsing_failures:
        # By agent count
        agents = failure["agents"]
        if agents not in failure_summary["failures_by_agent_count"]:
            failure_summary["failures_by_agent_count"][agents] = 0
        failure_summary["failures_by_agent_count"][agents] += 1
        
        # By temperature
        temp = failure["temp"]
        if temp not in failure_summary["failures_by_temperature"]:
            failure_summary["failures_by_temperature"][temp] = 0
        failure_summary["failures_by_temperature"][temp] += 1
    
    # Save to file
    with open(os.path.join(batch_folder, "parsing_failures_analysis.json"), 'w') as f:
        json.dump(failure_summary, f, indent=2)
    
    return failure_summary

def load_progress(batch_folder):
    """Load existing progress if available"""
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
        print("Usage: python run_experiment_multi_model.py <model_name>")
        print("Example: python run_experiment_multi_model.py 'openai/gpt-4o-mini'")
        sys.exit(1)
    
    model = sys.argv[1]
    
    # ===== EXPERIMENT PARAMETERS =====
    client_type = "openai"
    mode = "sum"  # "sum" or "mean"
    max_rounds = 15
    
    # ===== MASSIVE BATCH SETTINGS =====
    agents_list = list(range(2, 21))                    # 2 to 20 included (19 values)
    temp_list = [float(i/10) for i in range(0, 21)]     # 0.0 to 2.0 in 0.1 steps (21 values)
    runs_per_config = 10                                 # 10 runs per config
    
    # ===== BATCHING SETTINGS =====
    batch_size = 20          # Configs per batch
    max_concurrent = 8       # Max concurrent configs within each batch
    resume_from_batch = None # Set to batch number to resume from, or None to start fresh
    
    # Calculate totals
    total_configs = len(agents_list) * len(temp_list) * runs_per_config
    model_safe_name = model.replace("/", "_").replace(":", "_")
    
    print(f"🎯 MASSIVE EXPERIMENT SETUP:")
    print(f"   🤖 Model: {model}")
    print(f"   Agent counts: {len(agents_list)} ({min(agents_list)} to {max(agents_list)})")
    print(f"   Temperatures: {len(temp_list)} ({min(temp_list):.1f} to {max(temp_list):.1f})")
    print(f"   Runs per config: {runs_per_config}")
    print(f"   📊 TOTAL CONFIGS: {total_configs}")
    print(f"   📦 Batch size: {batch_size}")
    print(f"   ⚡ Max concurrent per batch: {max_concurrent}")
    
    # Create batch directory with model name
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_folder = f"results/massive_experiment_{model_safe_name}_{batch_timestamp}"
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
        parsing_analysis = save_parsing_failure_summary(batch_folder, final_progress)
        
        print(f"\n🎉 EXPERIMENT COMPLETED!")
        print(f"🤖 Model: {model}")
        print(f"📊 Final Results:")
        print(f"   ✅ Successful configs: {total_successful}")
        print(f"   ❌ Failed configs: {total_failed}")
        print(f"   🔤 Parsing failed configs: {total_parsing_failed}")
        print(f"   📈 Parsing failure rate: {parsing_analysis['failure_rate']:.1%}")
        print(f"   📦 Batches completed: {len(final_progress['batches_completed'])}/{total_batches}")
        print(f"   ⏱️  Total duration: {total_duration/3600:.1f} hours")
        print(f"📁 Results saved in: {batch_folder}")
        print(f"📋 Parsing analysis saved in: parsing_failures_analysis.json")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Experiment interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()