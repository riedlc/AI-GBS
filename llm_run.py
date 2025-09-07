
import asyncio
import random
import aiohttp
import json
from openai import AsyncOpenAI

_openai_client = None
_http_session = None

async def get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(
            api_key="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",
        )
    return _openai_client

async def get_http_session():
    """Create/reuse a single aiohttp session with connection pooling to avoid FD exhaustion."""
    global _http_session
    if _http_session is None or _http_session.closed:
        connector = aiohttp.TCPConnector(
            limit=10,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
        )
        timeout = aiohttp.ClientTimeout(total=120, sock_connect=10, sock_read=60)
        _http_session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    return _http_session

async def chat_ollama(model_name, prompt, system_prompt=None, temperature=0.0, max_tokens=None):
    """Chat with Ollama models via localhost:11434"""
    url = "http://localhost:11434/api/generate"
    
    # Build messages
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': prompt})
    
    payload = {
        'model': model_name,
        'messages': messages,
        'stream': False,
        'options': {
            'temperature': temperature
        }
    }
    
    if max_tokens:
        payload['options']['num_predict'] = max_tokens
    
    try:
        session = await get_http_session()
        async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    # Create OpenAI-compatible response format
                    class OllamaResponse:
                        def __init__(self, content):
                            self.choices = [type('obj', (object,), {
                                'message': type('obj', (object,), {'content': content})()
                            })()]
                            self.is_ollama = True
                    
                    return OllamaResponse(result['message']['content'])
                else:
                    raise Exception(f"Ollama API error: {response.status}")
    except Exception as e:
        raise Exception(f"Ollama API call failed: {e}")

async def chat_localhost(model_name, prompt, system_prompt=None, temperature=0.0, max_tokens=None):
    """Chat with localhost:5000/5001 API models (Ollama-compatible format)"""
    # Extract port from model_name if it contains localhost:port
    if "localhost:" in model_name:
        port = model_name.split(":")[1].split("/")[0]
        url = f"http://localhost:{port}/api/generate"
        model_name = model_name.split("/")[-1]  # Extract just the model name
    else:
        url = f"http://localhost:5000/api/generate"
    
    # Build the full prompt with system message if provided
    full_prompt = prompt
    if system_prompt:
        full_prompt = f"{system_prompt}\n\n{prompt}"
    
    payload = {
        'model': model_name,
        'prompt': full_prompt,
        'stream': False,
        'options': {
            'temperature': temperature
        }
    }
    
    if max_tokens:
        payload['options']['num_predict'] = max_tokens
    
    try:
        session = await get_http_session()
        async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    # Create OpenAI-compatible response format
                    class LocalhostResponse:
                        def __init__(self, content):
                            self.choices = [type('obj', (object,), {
                                'message': type('obj', (object,), {'content': content})()
                            })()]
                            self.is_localhost = True
                    
                    return LocalhostResponse(result['response'])
                else:
                    raise Exception(f"Localhost API error: {response.status}")
    except Exception as e:
        raise Exception(f"Localhost API call failed: {e}")

async def cleanup_clients():
    """Clean up async clients"""
    global _openai_client
    global _http_session
    if _openai_client:
        await _openai_client.close()
        _openai_client = None
    if _http_session and not _http_session.closed:
        await _http_session.close()
        _http_session = None

async def chat(model, prompt, system_prompt=None, temperature=0.0, max_tokens=None, max_retries=3):
    # Detect model type and route to appropriate API
    if model.startswith("ollama/"):
        model_name = model.replace("ollama/", "")
        return await chat_ollama(model_name, prompt, system_prompt, temperature, max_tokens)
    elif model.startswith("localhost:"):
        # Pass full string (e.g., "localhost:5001/model") so the port is preserved
        return await chat_localhost(model, prompt, system_prompt, temperature, max_tokens)
    else:
        # OpenAI API (existing logic)
        messages = []
        
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})
        
        # Retry logic for API failures
        for attempt in range(max_retries):
            try:
                client = await get_openai_client()
                kwargs = {
                    'model': model,
                    'messages': messages,
                    'temperature': temperature
                }
                if max_tokens is not None:
                    kwargs['max_tokens'] = max_tokens
                    
                response = await client.chat.completions.create(**kwargs)
                return response
                    
            except Exception as e:
                print(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"Retrying in {wait_time:.1f} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"All {max_retries} attempts failed")
                    # Return a mock response instead of crashing
                    class MockResponse:
                        def __init__(self):
                            self.choices = [type('obj', (object,), {'message': type('obj', (object,), {'content': str(random.randint(1, 100))})()})()]
                            self.is_fallback = True  # Flag to identify fallback responses
                    return MockResponse()

# Usage:
if __name__ == "__main__":
    async def test():
        # OpenAI
        result = await chat('gpt-4o-mini', 'Generate a number between 1 and 100', temperature=0, max_tokens=5)
        print("OpenAI:", result)
        
        # Ollama (if running)
        try:
            result = await chat('ollama/gemma2:2b', 'Generate a number between 1 and 100', temperature=0, max_tokens=5)
            print("Ollama:", result)
        except Exception as e:
            print("Ollama test failed:", e)
        
        # Localhost (if running)
        try:
            result = await chat('localhost:5000/gemma2', 'Generate a number between 1 and 100', temperature=0, max_tokens=5)
            print("Localhost:", result)
        except Exception as e:
            print("Localhost test failed:", e)
    
    asyncio.run(test())