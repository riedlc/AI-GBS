#!/usr/bin/env python3
import asyncio
import time
from llm_run import chat

async def test_call(model, prompt, call_num):
    start = time.time()
    try:
        response = await chat(model=model, prompt=prompt, temperature=0.7, max_tokens=50)
        end = time.time()
        
        content = response.choices[0].message.content
        print(f"Call {call_num}: {end-start:.2f}s - {content[:50]}...")
        return True, end-start
    except Exception as e:
        end = time.time()
        print(f"Call {call_num}: {end-start:.2f}s - ERROR: {e}")
        return False, end-start

async def main():
    model = "localhost:5001/gpt-oss:120b" 
    
    prompt = "Write a haiku about coding"
    
    print(f"Testing {model}")
    print(f"Prompt: {prompt}")
    print("-" * 50)
    
    # Test 1 call
    print("1 call:")
    await test_call(model, prompt, 1)
    
    # Test 10 calls
    print("\n10 calls:")
    times = []
    for i in range(1, 111):
        success, duration = await test_call(model, prompt, i)
        times.append(duration)
    
    print(f"Average time: {sum(times)/len(times):.2f}s")
    
    # Test 20 calls
    print("\n20 calls:")
    times = []
    for i in range(1, 21):
        success, duration = await test_call(model, prompt, i)
        times.append(duration)
    
    print(f"Average time: {sum(times)/len(times):.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
