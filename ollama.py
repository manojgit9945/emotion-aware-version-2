from flask import Flask, request, jsonify, Response
import os
import logging
import json
import time
import hashlib
import requests
from dotenv import load_dotenv
from flask_cors import CORS
from threading import Thread
from typing import Dict, List, Optional, Union, Generator
from functools import lru_cache

# Load environment variables
load_dotenv()

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_PRIMARY_MODEL = os.getenv("OPENROUTER_PRIMARY_MODEL", "microsoft/phi-4-reasoning:free")
OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_API_CHAT = f"{OPENROUTER_API_BASE}/chat/completions"
SITE_URL = os.getenv("SITE_URL", "http://localhost:5003")
SITE_NAME = os.getenv("SITE_NAME", "Local Chat API")

# Models configuration
PRIMARY_PROVIDER = os.getenv("PRIMARY_PROVIDER", "openrouter")  # Either "openrouter" or "ollama"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")  # Default to llama2 if not specified
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "mistral")  # Default fallback model
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "512"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
RESPONSE_TIMEOUT = int(os.getenv("RESPONSE_TIMEOUT", "10"))  # Timeout in seconds
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "True").lower() == "true"
CACHE_SIZE = int(os.getenv("CACHE_SIZE", "100"))  # Number of responses to cache

# Ollama API configuration
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
OLLAMA_API_GENERATE = f"{OLLAMA_API_BASE}/api/generate"
OLLAMA_API_CHAT = f"{OLLAMA_API_BASE}/api/chat"
OLLAMA_API_EMBEDDINGS = f"{OLLAMA_API_BASE}/api/embeddings"
OLLAMA_API_LIST = f"{OLLAMA_API_BASE}/api/tags"

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Response cache
response_cache = {}

def check_ollama_model_available(model_name):
    """Check if a model is available in Ollama"""
    try:
        response = requests.get(OLLAMA_API_LIST)
        if response.status_code == 200:
            models = response.json().get('models', [])
            available_models = [model['name'] for model in models]
            return model_name in available_models
        return False
    except Exception as e:
        logger.error(f"Error checking Ollama model availability: {str(e)}")
        return False

def format_chat_messages_ollama(messages):
    """Format chat messages for Ollama API"""
    # Convert from our format to Ollama's expected format
    ollama_messages = []
    
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "user":
            ollama_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            ollama_messages.append({"role": "assistant", "content": content})
        elif role == "system":
            # Add system message as the first message
            ollama_messages.insert(0, {"role": "system", "content": content})
    
    return ollama_messages

def format_chat_messages_openrouter(messages):
    """Format chat messages for OpenRouter API"""
    # OpenRouter uses the same format as OpenAI
    formatted_messages = []
    
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        if role in ["user", "assistant", "system"]:
            formatted_messages.append({"role": role, "content": content})
    
    return formatted_messages

def get_cache_key(provider, model_name, messages, max_tokens, temperature):
    """Generate a cache key for the request"""
    message_str = json.dumps(messages)
    key_str = f"{provider}:{model_name}:{message_str}:{max_tokens}:{temperature}"
    return hashlib.md5(key_str.encode()).hexdigest()

def generate_openrouter_response(
    model_name: str, 
    messages: List[Dict[str, str]], 
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    stream: bool = False
) -> Union[Dict, Generator]:
    """
    Generate a response using OpenRouter API.
    
    Args:
        model_name: The name of the OpenRouter model to use
        messages: List of message dictionaries with 'role' and 'content'
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling (higher = more random)
        stream: Whether to stream the response
        
    Returns:
        Either a dictionary with the response or a generator for streaming
    """
    if not OPENROUTER_API_KEY:
        logger.error("OpenRouter API key is not set")
        return None
        
    # For non-streaming responses, check cache first
    if not stream and ENABLE_CACHE:
        cache_key = get_cache_key("openrouter", model_name, messages, max_tokens, temperature)
        if cache_key in response_cache:
            logger.info(f"Cache hit for query: {messages[0]['content'][:50]}...")
            return response_cache[cache_key]
    
    try:
        logger.info(f"Generating response from OpenRouter model: {model_name}")
        
        # Format the messages for OpenRouter API
        formatted_messages = format_chat_messages_openrouter(messages)
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": SITE_URL,
            "X-Title": SITE_NAME,
        }
        
        # Prepare the request payload
        payload = {
            "model": model_name,
            "messages": formatted_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        
        # For streaming responses
        if stream:
            response = requests.post(
                OPENROUTER_API_CHAT,
                headers=headers,
                json=payload,
                stream=True
            )
            
            if response.status_code != 200:
                logger.error(f"OpenRouter API returned error: {response.text}")
                raise Exception(f"OpenRouter API error: {response.status_code}")
                
            def stream_generator():
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            line = line[6:]  # Remove 'data: ' prefix
                            if line == "[DONE]":
                                break
                            try:
                                data = json.loads(line)
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta and delta["content"]:
                                        yield delta["content"]
                            except json.JSONDecodeError:
                                logger.error(f"Failed to decode JSON: {line}")
            
            return stream_generator()
        
        # For non-streaming responses
        else:
            response = requests.post(
                OPENROUTER_API_CHAT,
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"OpenRouter API returned error: {response.text}")
                raise Exception(f"OpenRouter API error: {response.status_code}")
                
            result = response.json()
            
            # Extract the response content
            response_content = result["choices"][0]["message"]["content"]
            model_used = result.get("model", model_name)
            
            result_dict = {
                "model": model_used,
                "response": response_content,
                "provider": "openrouter"
            }
            
            # Cache the result if caching is enabled
            if ENABLE_CACHE:
                cache_key = get_cache_key("openrouter", model_name, messages, max_tokens, temperature)
                
                # Manage cache size - remove oldest entries if needed
                if len(response_cache) >= CACHE_SIZE:
                    oldest_key = next(iter(response_cache))
                    del response_cache[oldest_key]
                
                response_cache[cache_key] = result_dict
            
            return result_dict
    
    except Exception as e:
        logger.error(f"OpenRouter model {model_name} failed: {str(e)}")
        return None

def generate_ollama_response(
    model_name: str, 
    messages: List[Dict[str, str]], 
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    stream: bool = False
) -> Union[Dict, Generator]:
    """
    Generate a response using Ollama API.
    
    Args:
        model_name: The name of the Ollama model to use
        messages: List of message dictionaries with 'role' and 'content'
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling (higher = more random)
        stream: Whether to stream the response
        
    Returns:
        Either a dictionary with the response or a generator for streaming
    """
    # For non-streaming responses, check cache first
    if not stream and ENABLE_CACHE:
        cache_key = get_cache_key("ollama", model_name, messages, max_tokens, temperature)
        if cache_key in response_cache:
            logger.info(f"Cache hit for query: {messages[0]['content'][:50]}...")
            return response_cache[cache_key]
    
    try:
        logger.info(f"Generating response from Ollama model: {model_name}")
        
        # Check if model is available
        if not check_ollama_model_available(model_name):
            logger.error(f"Model {model_name} is not available in Ollama")
            return None
        
        # Format the messages for Ollama API
        ollama_messages = format_chat_messages_ollama(messages)
        
        # Prepare the request payload
        payload = {
            "model": model_name,
            "messages": ollama_messages,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature
            },
            "stream": stream
        }
        
        # For streaming responses
        if stream:
            # Make streaming request to Ollama API
            response = requests.post(OLLAMA_API_CHAT, json=payload, stream=True)
            
            if response.status_code != 200:
                logger.error(f"Ollama API returned error: {response.text}")
                raise Exception(f"Ollama API error: {response.status_code}")
            
            def stream_generator():
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "message" in data:
                                # Extract just the content part
                                content = data["message"]["content"]
                                yield content
                        except json.JSONDecodeError:
                            logger.error(f"Failed to decode JSON: {line}")
            
            return stream_generator()
            
        else:
            # Make non-streaming request to Ollama API
            response = requests.post(OLLAMA_API_CHAT, json=payload)
            
            if response.status_code != 200:
                logger.error(f"Ollama API returned error: {response.text}")
                raise Exception(f"Ollama API error: {response.status_code}")
            
            result = response.json()
            
            # Extract the response content
            response_content = result["message"]["content"]
            
            result_dict = {
                "model": model_name, 
                "response": response_content,
                "provider": "ollama"
            }
            
            # Cache the result if caching is enabled
            if ENABLE_CACHE:
                cache_key = get_cache_key("ollama", model_name, messages, max_tokens, temperature)
                
                # Manage cache size - remove oldest entries if needed
                if len(response_cache) >= CACHE_SIZE:
                    oldest_key = next(iter(response_cache))
                    del response_cache[oldest_key]
                
                response_cache[cache_key] = result_dict
            
            return result_dict
            
    except Exception as e:
        logger.error(f"Ollama model {model_name} failed: {str(e)}")
        return None

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    """
    Chat completion endpoint.
    
    Accepts:
        POST with JSON body:
        {
            "text": "User message",
            "max_tokens": 512,  # optional
            "temperature": 0.7,  # optional
            "stream": false      # optional
        }
    """
    global PRIMARY_PROVIDER
    start_time = time.time()
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing JSON body"}), 400
            
        if "text" not in data:
            return jsonify({"error": "Missing 'text' in request body"}), 400

        user_input = data["text"]
        max_tokens = int(data.get("max_tokens", DEFAULT_MAX_TOKENS))
        temperature = float(data.get("temperature", DEFAULT_TEMPERATURE))
        should_stream = bool(data.get("stream", False))
        
        # Performance optimization: Limit max tokens if input is very short
        if len(user_input.split()) < 5 and max_tokens > 100:
            max_tokens = min(max_tokens, 100)  # Shorter response for simple queries
        
        messages = [{"role": "user", "content": user_input}]
        
        logger.info(f"Chat request: '{user_input[:50]}...' (stream={should_stream})")
        
        # Determine which provider to use as primary
        # "openrouter" or "ollama"
        
        # Streaming response
        if should_stream:
            def generate():
                # Try primary provider
                try:
                    if PRIMARY_PROVIDER == "openrouter" and OPENROUTER_API_KEY:
                        streamer = generate_openrouter_response(
                            OPENROUTER_PRIMARY_MODEL, messages, max_tokens, temperature, stream=True
                        )
                        provider_model = OPENROUTER_PRIMARY_MODEL
                        provider_name = "openrouter"
                    else:
                        streamer = generate_ollama_response(
                            OLLAMA_MODEL, messages, max_tokens, temperature, stream=True
                        )
                        provider_model = OLLAMA_MODEL
                        provider_name = "ollama"
                    
                    if streamer:
                        # Stream the SSE response
                        yield f"data: {json.dumps({'model': provider_model, 'provider': provider_name})}\n\n"
                        
                        # Batch tokens for faster response
                        buffer = ""
                        last_send_time = time.time()
                        
                        for text in streamer:
                            if text:
                                buffer += text
                                current_time = time.time()
                                
                                # Send if buffer has enough tokens or enough time has passed
                                if len(buffer) >= 10 or (current_time - last_send_time) >= 0.1:
                                    yield f"data: {json.dumps({'token': buffer})}\n\n"
                                    buffer = ""
                                    last_send_time = current_time
                        
                        # Send any remaining tokens
                        if buffer:
                            yield f"data: {json.dumps({'token': buffer})}\n\n"
                                
                        yield "data: [DONE]\n\n"
                        return
                    else:
                        raise Exception("Failed to initialize streamer")
                    
                except Exception as e:
                    logger.warning(f"Primary provider {PRIMARY_PROVIDER} streaming failed: {str(e)}")
                    # If primary fails, try fallback immediately
                    try:
                        # Use Ollama as fallback
                        fallback_streamer = generate_ollama_response(
                            FALLBACK_MODEL, messages, max_tokens, temperature, stream=True
                        )
                        
                        if fallback_streamer:
                            yield f"data: {json.dumps({'model': FALLBACK_MODEL, 'provider': 'ollama', 'fallback': True})}\n\n"
                            
                            # Batch tokens for faster response
                            buffer = ""
                            last_send_time = time.time()
                            
                            for text in fallback_streamer:
                                if text:
                                    buffer += text
                                    current_time = time.time()
                                    
                                    # Send if buffer has enough tokens or enough time has passed
                                    if len(buffer) >= 10 or (current_time - last_send_time) >= 0.1:
                                        yield f"data: {json.dumps({'token': buffer})}\n\n"
                                        buffer = ""
                                        last_send_time = current_time
                            
                            # Send any remaining tokens
                            if buffer:
                                yield f"data: {json.dumps({'token': buffer})}\n\n"
                                    
                            yield "data: [DONE]\n\n"
                        else:
                            raise Exception("Failed to initialize fallback streamer")
                        
                    except Exception as fallback_e:
                        logger.error(f"Fallback model also failed: {str(fallback_e)}")
                        yield f"data: {json.dumps({'error': 'Both providers failed'})}\n\n"            
            return Response(generate(), mimetype='text/event-stream')
        
        # Non-streaming response
        else:
            # Try primary provider
            result = None
            try:
                # Start timer for timeout tracking
                primary_start = time.time()
                
                # Try primary provider based on configuration
                if PRIMARY_PROVIDER == "openrouter" and OPENROUTER_API_KEY:
                    result = generate_openrouter_response(
                        OPENROUTER_PRIMARY_MODEL, messages, max_tokens, temperature, stream=False
                    )
                else:
                    result = generate_ollama_response(
                        OLLAMA_MODEL, messages, max_tokens, temperature, stream=False
                    )
                
                # Check if we should fall back due to timeout
                if result is None and (time.time() - primary_start) > RESPONSE_TIMEOUT:
                    logger.warning(f"Primary provider {PRIMARY_PROVIDER} timed out after {RESPONSE_TIMEOUT}s")
                    raise TimeoutError("Primary provider response took too long")
                
            except Exception as e:
                logger.warning(f"Primary provider {PRIMARY_PROVIDER} failed: {str(e)}")
            
            # If primary fails, try fallback (always Ollama)
            if result is None:
                logger.warning("Primary provider failed. Switching to Ollama fallback...")
                result = generate_ollama_response(
                    FALLBACK_MODEL, messages, max_tokens, temperature, stream=False
                )
                if result:
                    result["fallback"] = True

            if result:
                elapsed = time.time() - start_time
                logger.info(f"Response generated in {elapsed:.2f}s using {result['provider']}")
                return jsonify(result)
            else:
                return jsonify({"error": "Both providers failed to generate a response"}), 500

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint that verifies model availability."""
    logger.info("Health check requested")
    start_time = time.time()

    openrouter_available = False
    ollama_available = False
    ollama_models = []

    try:
        # Check OpenRouter availability if configured
        if OPENROUTER_API_KEY:
            try:
                # Simple non-streaming request to check if OpenRouter is working
                headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": SITE_URL,
                    "X-Title": SITE_NAME,
                }
                response = requests.get(
                    f"{OPENROUTER_API_BASE}/models",
                    headers=headers
                )
                openrouter_available = response.status_code == 200
                logger.info(f"OpenRouter API check: {'Available' if openrouter_available else 'Unavailable'}")
            except Exception as e:
                logger.error(f"OpenRouter health check failed: {str(e)}")
                openrouter_available = False
        
        # Check Ollama availability
        try:
            response = requests.get(OLLAMA_API_LIST)
            if response.status_code == 200:
                ollama_available = True
                models = response.json().get('models', [])
                ollama_models = [model['name'] for model in models]
                logger.info(f"Ollama API check: Available with {len(ollama_models)} models")
            else:
                logger.error(f"Ollama API returned status code: {response.status_code}")
                ollama_available = False
        except Exception as e:
            logger.error(f"Ollama health check failed: {str(e)}")
            ollama_available = False
        
        # Determine overall status
        if PRIMARY_PROVIDER == "openrouter" and openrouter_available:
            status = "healthy"
            primary_status = "available"
        elif PRIMARY_PROVIDER == "ollama" and ollama_available and OLLAMA_MODEL in ollama_models:
            status = "healthy"
            primary_status = "available"
        elif (PRIMARY_PROVIDER == "openrouter" and not openrouter_available and 
              ollama_available and FALLBACK_MODEL in ollama_models):
            status = "degraded"
            primary_status = "unavailable"
        elif (PRIMARY_PROVIDER == "ollama" and (not ollama_available or OLLAMA_MODEL not in ollama_models) and
              openrouter_available):
            status = "degraded"
            primary_status = "unavailable"
        else:
            status = "down"
            primary_status = "unavailable"
    
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        status = "down"
        primary_status = "unavailable"

    response_time = time.time() - start_time
    
    return jsonify({
        "status": status,
        "PRIMARY_PROVIDER": PRIMARY_PROVIDER,
        "primary_status": primary_status,
        "openrouter": {
            "available": openrouter_available,
            "model": OPENROUTER_PRIMARY_MODEL if openrouter_available else None
        },
        "ollama": {
            "available": ollama_available,
            "primary_model": OLLAMA_MODEL,
            "fallback_model": FALLBACK_MODEL,
            "available_models": ollama_models if ollama_available else []
        },
        "cache_enabled": ENABLE_CACHE,
        "cache_size": len(response_cache) if ENABLE_CACHE else 0,
        "service": "Hybrid Chat Completion API",
        "response_time_seconds": round(response_time, 3),
        "streaming_supported": True,
        "server_time": time.ctime()
    })
@app.route("/models", methods=["GET"])
def list_models():
    """Returns the available models from both providers."""
    result = {
        "PRIMARY_PROVIDER": PRIMARY_PROVIDER,
        "openrouter": {
            "primary_model": OPENROUTER_PRIMARY_MODEL,
            "available_models": []
        },
        "ollama": {
            "primary_model": OLLAMA_MODEL,
            "fallback_model": FALLBACK_MODEL,
            "available_models": []
        }
    }
    
    # Get OpenRouter models if configured
    if OPENROUTER_API_KEY:
        try:
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": SITE_URL,
                "X-Title": SITE_NAME,
            }
            response = requests.get(
                f"{OPENROUTER_API_BASE}/models",
                headers=headers
            )
            if response.status_code == 200:
                models_data = response.json()
                # Format depends on API response structure
                if "data" in models_data:
                    result["openrouter"]["available_models"] = [model["id"] for model in models_data["data"]]
                else:
                    result["openrouter"]["available_models"] = [model["id"] for model in models_data]
        except Exception as e:
            logger.error(f"Failed to get OpenRouter models: {str(e)}")
    
    # Get Ollama models
    try:
        response = requests.get(OLLAMA_API_LIST)
        if response.status_code == 200:
            ollama_models = response.json().get('models', [])
            result["ollama"]["available_models"] = [model['name'] for model in ollama_models]
        else:
            logger.error(f"Failed to get models from Ollama: {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to connect to Ollama: {str(e)}")
    
    return jsonify(result)
@app.route("/", methods=["GET"])
def home():
    """Provides basic documentation as an HTML page."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hybrid Chat Completion API (OpenRouter + Ollama)</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; color: #333; max-width: 800px; }}
            code {{ background: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-family: monospace; }}
            pre {{ background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
            h1, h2 {{ color: #444; }}
            .endpoint {{ margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 20px; }}
            .method {{ display: inline-block; padding: 3px 6px; border-radius: 3px; font-size: 0.8em; color: white; }}
            .post {{ background-color: #49cc90; }}
            .get {{ background-color: #61affe; }}
        </style>
    </head>
    <body>
        <h1>Hybrid Chat Completion API (OpenRouter + Ollama)</h1>
        <p>This service provides access to both OpenRouter and locally running Ollama language models for chat completion.</p>
        
        <div class="endpoint">
            <h2><span class="method post">POST</span> /chat</h2>
            <p>Send a message to get a completion from the configured model.</p>
            <h3>Request Format:</h3>
            <pre>
{{
  "text": "Your message to the model",
  "max_tokens": 512,      // optional (default: 512)
  "temperature": 0.7,     // optional (default: 0.7)
  "stream": false         // optional (default: false)
}}
            </pre>
            
            <h3>Response Format (non-streaming):</h3>
            <pre>
{{
  "model": "model_name",
  "provider": "openrouter|ollama",
  "response": "The model's response to your message"
}}
            </pre>
            
            <p>If streaming is enabled (stream=true), the response will be sent as server-sent events (SSE).</p>
        </div>
        
        <div class="endpoint">
            <h2><span class="method get">GET</span> /health</h2>
            <p>Check the health status of the service and models.</p>
        </div>
        
        <div class="endpoint">
            <h2><span class="method get">GET</span> /models</h2>
            <p>List the available models from both providers.</p>
        </div>
        
        <div class="endpoint">
            <h2><span class="method post">POST</span> /clear-cache</h2>
            <p>Clear the response cache.</p>
        </div>
        
        <p><strong>Primary Provider:</strong> {PRIMARY_PROVIDER}</p>
        <p><strong>OpenRouter Model:</strong> {OPENROUTER_PRIMARY_MODEL}</p>
        <p><strong>Ollama Primary Model:</strong> {OLLAMA_MODEL}</p>
        <p><strong>Ollama Fallback Model:</strong> {FALLBACK_MODEL}</p>
        <p><strong>Caching:</strong> {'Enabled' if ENABLE_CACHE else 'Disabled'}</p>
    </body>
    </html>
    """

@app.route("/clear-cache", methods=["POST"])
def clear_cache():
    """Clear the response cache"""
    global response_cache
    response_cache = {}
    return jsonify({"status": "success", "message": "Cache cleared"})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5003))
    debug_mode = os.getenv("DEBUG", "False").lower() == "true"
    
    logger.info(f"Starting server on port {port} (debug={debug_mode})")
    
    # Check OpenRouter API key
    if PRIMARY_PROVIDER == "openrouter" and not OPENROUTER_API_KEY: 
        logger.warning("OpenRouter is set as primary provider but API key is not configured")
        logger.warning("Set OPENROUTER_API_KEY in your environment or .env file")
        logger.warning("Falling back to Ollama as primary provider")
        PRIMARY_PROVIDER = "ollama" 
    
    # Log configuration
    logger.info(f"Primary provider: {PRIMARY_PROVIDER}")
    if OPENROUTER_API_KEY:
        logger.info(f"OpenRouter model: {OPENROUTER_PRIMARY_MODEL}")
        logger.info(f"OpenRouter API: {OPENROUTER_API_BASE}")
    
    logger.info(f"Ollama API: {OLLAMA_API_BASE}")
    logger.info(f"Ollama primary model: {OLLAMA_MODEL}")
    logger.info(f"Ollama fallback model: {FALLBACK_MODEL}")
    logger.info(f"Response caching: {'Enabled' if ENABLE_CACHE else 'Disabled'}")
    
    # Check if Ollama server is running
    try:
        response = requests.get(OLLAMA_API_LIST)
        if response.status_code == 200:
            logger.info("Successfully connected to Ollama server")
            
            # List available models
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            logger.info(f"Available Ollama models: {', '.join(model_names)}")
            
            # Check if Ollama models are available
            if OLLAMA_MODEL not in model_names:
                logger.warning(f"Ollama primary model '{OLLAMA_MODEL}' not found in Ollama. Available models: {', '.join(model_names)}")
            
            if FALLBACK_MODEL not in model_names:
                logger.warning(f"Fallback model '{FALLBACK_MODEL}' not found in Ollama. Available models: {', '.join(model_names)}")
        else:
            logger.error(f"Failed to connect to Ollama API: Status code {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to connect to Ollama server: {str(e)}")
        if PRIMARY_PROVIDER == "ollama":
     
            logger.error("Primary provider is set to Ollama, but Ollama server is not available")
            logger.error("Make sure Ollama is running on your machine (default: http://localhost:11434)")
    
    app.run(host="0.0.0.0", port=port, debug=debug_mode)