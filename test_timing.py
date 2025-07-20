#!/usr/bin/env python3
"""
Test script to demonstrate timing functionality for different experiment configurations.
"""

import asyncio
import time
from experiment import run_async_test

async def test_timing_configurations():
    """Test timing for different experiment configurations"""
    
    print("⏱️  TIMING TEST - Different Experiment Configurations")
    print("=" * 60)
    
    # Test configurations
    configs = [
        {"name": "Small (2 agents, low temp)", "agents": 2, "temp": 0.0},
        {"name": "Medium (5 agents, medium temp)", "agents": 5, "temp": 0.7},
        {"name": "Large (10 agents, high temp)", "agents": 10, "temp": 1.5},
    ]
    
    total_start_time = time.time()
    
    for i, config in enumerate(configs, 1):
        print(f"\n🧪 Test {i}: {config['name']}")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            await run_async_test(
                num_agents=config["agents"],
                model="gpt-4o-mini",
                temperature=config["temp"],
                mode="sum"
            )
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            print(f"✅ Completed in {elapsed:.2f} seconds")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    total_elapsed = time.time() - total_start_time
    print(f"\n" + "=" * 60)
    print(f"🏁 Total time for all tests: {total_elapsed:.2f} seconds ({total_elapsed/60:.1f} minutes)")
    print(f"📊 Average per test: {total_elapsed/len(configs):.2f} seconds")

if __name__ == "__main__":
    asyncio.run(test_timing_configurations()) 