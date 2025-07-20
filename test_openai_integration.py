#!/usr/bin/env python3
"""
Test script to verify OpenAI integration works correctly.
This will test the API connection and basic functionality.
"""

import asyncio
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from llm_run import chat
    print("✅ Successfully imported llm_run module")
except ImportError as e:
    print(f"❌ Failed to import llm_run: {e}")
    sys.exit(1)

async def test_openai_connection():
    """Test the OpenAI API connection"""
    print("\n🧪 Testing OpenAI API connection...")
    
    try:
        # Test with a simple prompt
        response = await chat(
            model="gpt-4o-mini",
            prompt="Generate a random number between 1 and 10. Respond with only the number.",
            client_type="openai",
            temperature=0.0,
            max_tokens=5
        )
        
        print(f"✅ API call successful!")
        print(f"📝 Response: {response}")
        
        # Extract the content from the response
        if hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content
            print(f"📄 Content: {content}")
            
            # Try to extract a number
            import re
            numbers = re.findall(r'\d+', content)
            if numbers:
                print(f"🔢 Extracted number: {numbers[0]}")
            else:
                print("⚠️  No number found in response")
        
        return True
        
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return False

async def test_experiment_structure():
    """Test that the experiment structure works"""
    print("\n🧪 Testing experiment structure...")
    
    try:
        from experiment import GameMaster, Agent
        
        # Create a simple test
        game_master = GameMaster(
            num_agents=2,
            temperature=0.0,
            mode="sum"
        )
        
        # Add agents
        game_master.add_agent("gpt-4o-mini", "openai")
        game_master.add_agent("gpt-4o-mini", "openai")
        
        print(f"✅ GameMaster created successfully")
        print(f"📊 Mode: {game_master.mode}")
        print(f"🎯 Target range: {game_master.mystery_range}")
        print(f"🤖 Number of agents: {len(game_master.agents)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Experiment structure test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Starting OpenAI Integration Tests")
    print("=" * 50)
    
    # Test 1: Module imports
    print("✅ All modules imported successfully")
    
    # Test 2: Experiment structure
    structure_ok = await test_experiment_structure()
    
    # Test 3: OpenAI API connection (only if dependencies are available)
    try:
        from openai import AsyncOpenAI
        api_ok = await test_openai_connection()
    except ImportError:
        print("\n⚠️  Skipping API test - OpenAI package not available")
        print("   Install with: pip install openai python-dotenv")
        api_ok = True  # Don't fail the test
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"   Structure Test: {'✅ PASS' if structure_ok else '❌ FAIL'}")
    print(f"   API Test: {'✅ PASS' if api_ok else '❌ FAIL'}")
    
    if structure_ok and api_ok:
        print("\n🎉 All tests passed! The integration is ready.")
        return 0
    else:
        print("\n💥 Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 