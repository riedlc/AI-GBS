import os
import asyncio
import random
import aiohttp
import json
from openai import AsyncOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_openai_client = None
_http_session = None
_vllm_clients = {}  # base_url -> AsyncOpenAI
_openrouter_client = None

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
    """Chat with Ollama models via localhost:5001"""
    url = "http://localhost:5001/api/generate"
    
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
        url = f"http://localhost:5001/api/generate"
    
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
        raise Exception(f"Localhost API call failed: {e}. Use localhost:5001/nvidia/Llama-3.1-70B-Instruct-FP8 (or vllm:5001/...) for TensorRT-LLM.")


def _parse_vllm_model(model_string):
    """
    Parse vLLM model string into (base_url, model_name).
    Examples:
      vllm:8000/my-model           -> (http://localhost:8000/v1, my-model)
      vllm:155.33.199.239:8000/xy  -> (http://155.33.199.239:8000/v1, xy)
      vllm/my-model                -> (VLLM_BASE_URL or http://localhost:8000/v1, my-model)
    """
    if model_string.startswith("vllm:"):
        rest = model_string[5:]
        if "/" not in rest:
            return os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/"), rest
        left, model_name = rest.split("/", 1)
        if ":" in left:
            host, port = left.rsplit(":", 1)
            base_url = f"http://{host}:{port}/v1"
        else:
            # Use 127.0.0.1 so SSH -L (which binds IPv4) is used; localhost can resolve to ::1
            base_url = f"http://127.0.0.1:{left}/v1"
        return base_url, model_name
    if model_string.startswith("vllm/"):
        model_name = model_string[5:]
        base_url = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
        return base_url, model_name
    raise ValueError(f"Invalid vLLM model string: {model_string}")


def _get_vllm_client(base_url):
    """Get or create AsyncOpenAI client for vLLM (OpenAI-compatible API)."""
    global _vllm_clients
    if base_url not in _vllm_clients:
        _vllm_clients[base_url] = AsyncOpenAI(
            base_url=base_url,
            api_key="dummy",
        )
    return _vllm_clients[base_url]


async def get_openrouter_client():
    """OpenRouter is OpenAI-compatible; use base_url and OPENROUTER_API_KEY."""
    global _openrouter_client
    if _openrouter_client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Get a key at https://openrouter.ai/keys and run:\n"
                "  export OPENROUTER_API_KEY=sk-or-..."
            )
        _openrouter_client = AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
        )
    return _openrouter_client


async def chat_openrouter(model_id, prompt, system_prompt=None, temperature=0.0, max_tokens=None, max_retries=3):
    """Chat via OpenRouter (OpenAI-compatible). model_id is e.g. qwen/qwen-2.5-72b-instruct."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(max_retries):
        try:
            client = await get_openrouter_client()
            kwargs = {
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            response = await client.chat.completions.create(**kwargs)
            return response
        except Exception as e:
            print(f"OpenRouter API call attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Retrying in {wait_time:.1f} seconds...")
                await asyncio.sleep(wait_time)
            else:
                raise


async def chat_vllm(model_string, prompt, system_prompt=None, temperature=0.0, max_tokens=None, max_retries=3):
    """Chat with vLLM via OpenAI-compatible /v1/chat/completions API."""
    base_url, model_name = _parse_vllm_model(model_string)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(max_retries):
        try:
            client = _get_vllm_client(base_url)
            kwargs = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            response = await client.chat.completions.create(**kwargs)
            return response
        except Exception as e:
            print(f"API call attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Retrying in {wait_time:.1f} seconds...")
                await asyncio.sleep(wait_time)
            else:
                print(f"All {max_retries} attempts failed")
                class MockResponse:
                    def __init__(self):
                        self.choices = [type("obj", (object,), {"message": type("obj", (object,), {"content": str(random.randint(1, 100))})()})()]
                        self.is_fallback = True
                return MockResponse()


async def cleanup_clients():
    """Clean up async clients"""
    global _openai_client
    global _http_session
    global _vllm_clients
    global _openrouter_client
    if _openai_client:
        await _openai_client.close()
        _openai_client = None
    for client in _vllm_clients.values():
        await client.close()
    _vllm_clients = {}
    if _openrouter_client:
        await _openrouter_client.close()
        _openrouter_client = None
    if _http_session and not _http_session.closed:
        await _http_session.close()
        _http_session = None

async def chat(model, prompt, system_prompt=None, temperature=0.0, max_tokens=None, max_retries=3):
    # Detect model type and route to appropriate API
    if model.startswith("ollama/"):
        model_name = model.replace("ollama/", "")
        return await chat_ollama(model_name, prompt, system_prompt, temperature, max_tokens)
    elif model.startswith("localhost:"):
        # localhost:port/model-id -> OpenAI-compatible API (vLLM, TensorRT-LLM, etc.) using /v1/chat/completions
        rest = model[len("localhost:"):]
        if "/" in rest:
            port_and_model = rest.split("/", 1)  # e.g. "9000" and "qwen-30b" or "nvidia/Llama-3.1-70B-Instruct-FP8"
            port, model_id = port_and_model[0], port_and_model[1]
            vllm_style = f"vllm:{port}/{model_id}"
            return await chat_vllm(vllm_style, prompt, system_prompt, temperature, max_tokens, max_retries)
        # Ollama-style: localhost:5001 only (no path) uses /api/generate
        return await chat_localhost(model, prompt, system_prompt, temperature, max_tokens)
    elif model.startswith("vllm:") or model.startswith("vllm/"):
        return await chat_vllm(model, prompt, system_prompt, temperature, max_tokens, max_retries)
    elif model.startswith("openrouter/"):
        model_id = model[len("openrouter/"):].strip()
        if not model_id:
            raise ValueError("openrouter/ model string must include a model id, e.g. openrouter/qwen/qwen-2.5-72b-instruct")
        return await chat_openrouter(model_id, prompt, system_prompt, temperature, max_tokens, max_retries)
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
        
        # vLLM (if running, e.g. vllm:8000/model-name)
        try:
            result = await chat('vllm:8000/my-model', 'Generate a number between 1 and 100', temperature=0, max_tokens=5)
            print("vLLM:", result)
        except Exception as e:
            print("vLLM test failed:", e)
    
    asyncio.run(test())