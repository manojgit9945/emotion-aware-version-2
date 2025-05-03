from flask import Flask, request, jsonify, render_template_string
import logging
import time
import os
import traceback
from transformers import pipeline
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("emotion_detector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Dictionary of response messages for different emotions
emotion_responses = {
    "admiration": "I can sense your admiration. What impressed you?",
    "amusement": "That seems to have brought you joy!",
    "anger": "I notice you're feeling angry. Would you like to talk about it?",
    "annoyance": "Something seems to be bothering you.",
    "approval": "You seem to approve of this.",
    "caring": "Your compassion is evident.",
    "confusion": "You seem a bit confused. Can I help clarify something?",
    "curiosity": "I can sense your curiosity. What would you like to know more about?",
    "desire": "You seem to have a strong desire for something.",
    "disappointment": "I'm sorry to see you're disappointed.",
    "disapproval": "You don't seem to approve of this.",
    "disgust": "That seems to have evoked a strong negative reaction.",
    "embarrassment": "No need to feel embarrassed.",
    "excitement": "I can sense your excitement!",
    "fear": "I notice you might be feeling afraid. Is everything okay?",
    "gratitude": "Your gratitude is heartwarming.",
    "grief": "I'm truly sorry for your loss or pain.",
    "joy": "I'm glad to see you're feeling joyful!",
    "love": "There's a lot of love in your words.",
    "nervousness": "You seem a bit nervous. Is there something on your mind?",
    "optimism": "I appreciate your positive outlook.",
    "pride": "You have every reason to feel proud.",
    "realization": "It seems you've had an insight or realization.",
    "relief": "You seem relieved. I'm glad things worked out.",
    "remorse": "I can sense your remorse. We all make mistakes.",
    "sadness": "I notice you're feeling sad. Would you like to talk about it?",
    "surprise": "That seems to have surprised you!",
    "neutral": "I understand what you're saying."
}

# Model loading with status tracking
model_loading = True
emotion_pipeline = None

def load_model():
    """Load the Hugging Face model"""
    global emotion_pipeline, model_loading
    
    try:
        logger.info("Loading emotion detection model from Hugging Face...")
        emotion_pipeline = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")
        logger.info("Emotion detection model loaded successfully")
        model_loading = False
        return True
    except Exception as e:
        logger.error(f"Failed to load emotion detection model: {str(e)}")
        traceback.print_exc()
        model_loading = False
        return False

# Start loading the model
import threading
model_thread = threading.Thread(target=load_model)
model_thread.daemon = True
model_thread.start()

def check_model_loaded():
    """Check if the model is loaded and return appropriate error if not."""
    if model_loading:
        return {"error": "Model is still loading. Please try again in a moment."}, 503
    if emotion_pipeline is None:
        return {"error": "Emotion detection model failed to load. Please check logs."}, 500
    return None

# Function to detect emotions using the Hugging Face model
def detect_emotion(text):
    """Detect emotions using the Hugging Face model."""
    logger.info(f"Detecting emotions for: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    
    # Check if model is loaded
    check_result = check_model_loaded()
    if check_result:
        return check_result[0]
    
    try:
        start_time = time.time()
        
        # Input validation
        if not text or not isinstance(text, str):
            return {"error": "Invalid input text"}
            
        if len(text) > 1000:  # Set a reasonable text length limit
            logger.warning(f"Text length exceeds limit: {len(text)} characters")
            text = text[:1000]  # Truncate to prevent model overload
        
        # Get emotion predictions from the model
        result = emotion_pipeline(text)
        
        # Sort emotions by their score in descending order
        top_emotions = sorted(result, key=lambda x: x['score'], reverse=True)[:3]
        
        processing_time = time.time() - start_time
        logger.info(f"Emotion detection completed in {processing_time:.2f}s")
        
        # Format the result
        primary_emotion = top_emotions[0]['label'].lower()
        
        return {
            "primary_emotion": primary_emotion,
            "primary_score": float(top_emotions[0]['score']),
            "secondary_emotions": [{"emotion": emotion['label'].lower(), "score": float(emotion['score'])} for emotion in top_emotions[1:]],
            "message": emotion_responses.get(primary_emotion, "I'm here to listen.")
        }
    
    except Exception as e:
        logger.error(f"ERROR in emotion detection: {str(e)}")
        traceback.print_exc()
        return {"error": str(e)}

@app.route('/detect-emotion', methods=['POST'])
def detect_emotion_endpoint():
    """API endpoint to detect emotions in text."""
    try:
        logger.info("Received request to /detect-emotion endpoint")
        
        # Check for JSON content type
        if not request.is_json:
            logger.warning("Request does not contain JSON")
            return jsonify({"error": "Request must be JSON"}), 415
            
        data = request.get_json()
        if not data:
            logger.warning("Missing request body")
            return jsonify({"error": "Missing request body"}), 400
        
        user_input = data.get("text")
        if not user_input:
            logger.warning("Missing 'text' field in request")
            return jsonify({"error": "Missing 'text' field in request"}), 400
        
        logger.info(f"Processing text for emotions: '{user_input[:50]}{'...' if len(user_input) > 50 else ''}'")
        
        # Check if model is loaded
        check_result = check_model_loaded()
        if check_result:
            return jsonify(check_result[0]), check_result[1]
        
        # Get emotions
        result = detect_emotion(user_input)
        
        if "error" in result:
            logger.error(f"Error processing emotion: {result['error']}")
            return jsonify({"error": result["error"]}), 500
            
        logger.info(f"Sending emotion response: {result}")
        return jsonify(result)
    except Exception as e:
        logger.error(f"ERROR in detect-emotion endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    logger.info("Health check requested")
    
    if model_loading:
        status = "loading"
        message = "Model is currently loading"
    elif emotion_pipeline is None:
        status = "error"
        message = "Model failed to load"
    else:
        status = "ready"
        message = "Model is loaded and ready"
    
    return jsonify({
        "status": status,
        "service": "Emotion Detection",
        "model": "SamLowe/roberta-base-go_emotions",
        "message": message,
        "timestamp": time.time()
    })

@app.route('/', methods=['GET'])
def home():
    """Homepage with interactive UI."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Emotion Detector</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                max-width: 800px;
                margin: 0 auto;
                color: #333;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }
            .container {
                background-color: #f9f9f9;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            textarea {
                width: 100%;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                height: 120px;
                margin-bottom: 10px;
                font-family: inherit;
                resize: vertical;
            }
            button {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                transition: background-color 0.3s;
            }
            button:hover {
                background-color: #2980b9;
            }
            button:disabled {
                background-color: #95a5a6;
                cursor: not-allowed;
            }
            .results {
                margin-top: 20px;
                display: none;
            }
            .emotion {
                margin-bottom: 5px;
            }
            .primary {
                font-weight: bold;
                color: #2c3e50;
            }
            .secondary {
                color: #7f8c8d;
            }
            .message {
                margin-top: 15px;
                padding: 10px;
                background-color: #eaf2f8;
                border-left: 4px solid #3498db;
                border-radius: 4px;
            }
            .char-count {
                text-align: right;
                font-size: 12px;
                color: #7f8c8d;
            }
            .error {
                color: #c0392b;
                background-color: #fadbd8;
                padding: 10px;
                border-radius: 4px;
                margin-top: 10px;
                display: none;
            }
            .loading {
                display: none;
                margin-top: 10px;
                text-align: center;
                color: #7f8c8d;
            }
            .status {
                padding: 10px;
                border-radius: 4px;
                margin-bottom: 20px;
            }
            .status.loading {
                background-color: #fef9e7;
                border-left: 4px solid #f1c40f;
                display: block;
            }
            .status.error {
                background-color: #fadbd8;
                border-left: 4px solid #c0392b;
                display: block;
            }
            .status.ready {
                background-color: #d5f5e3;
                border-left: 4px solid #2ecc71;
            }
            @media (max-width: 600px) {
                body {
                    padding: 10px;
                }
                .container {
                    padding: 15px;
                }
            }
        </style>
    </head>
    <body>
        <h1>Emotion Detector</h1>
        
        <div class="status" id="modelStatus">
            Checking model status...
        </div>
        
        <div class="container">
            <div>
                <label for="text-input">Enter your text:</label>
                <textarea id="text-input" placeholder="Type something here to analyze emotions..." maxlength="1000"></textarea>
                <div class="char-count"><span id="charCount">0</span>/1000 characters</div>
            </div>
            
            <button id="analyzeBtn" disabled>Analyze Emotions</button>
            
            <div class="loading" id="loadingIndicator">
                Analyzing emotions... Please wait.
            </div>
            
            <div class="error" id="errorDisplay"></div>
            
            <div class="results" id="resultsContainer">
                <h3>Detected Emotions:</h3>
                <div class="emotion primary">
                    Primary: <span id="primaryEmotion"></span> (<span id="primaryScore"></span>)
                </div>
                <div class="emotion secondary" id="secondaryEmotions">
                    <!-- Secondary emotions will be inserted here -->
                </div>
                <div class="message" id="responseMessage"></div>
            </div>
        </div>

        <script>
            // Check model status on page load
            document.addEventListener('DOMContentLoaded', function() {
                checkModelStatus();
                
                // Setup character counter
                const textInput = document.getElementById('text-input');
                const charCount = document.getElementById('charCount');
                
                textInput.addEventListener('input', function() {
                    charCount.textContent = this.value.length;
                    if (this.value.length > 950) {
                        charCount.style.color = '#c0392b';
                    } else {
                        charCount.style.color = '#7f8c8d';
                    }
                });
                
                // Setup analyze button
                document.getElementById('analyzeBtn').addEventListener('click', analyzeText);
            });
            
            // Check if the model is loaded
            function checkModelStatus() {
                fetch('/health')
                    .then(response => response.json())
                    .then(data => {
                        const statusElement = document.getElementById('modelStatus');
                        
                        if (data.status === 'loading') {
                            statusElement.className = 'status loading';
                            statusElement.textContent = 'Model is currently loading. Please wait...';
                            setTimeout(checkModelStatus, 2000); // Check again in 2 seconds
                        } else if (data.status === 'error') {
                            statusElement.className = 'status error';
                            statusElement.textContent = 'Error: Model failed to load. Please refresh the page or try again later.';
                        } else if (data.status === 'ready') {
                            statusElement.className = 'status ready';
                            statusElement.textContent = 'Model is loaded and ready to use!';
                            document.getElementById('analyzeBtn').disabled = false;
                        }
                    })
                    .catch(error => {
                        console.error('Error checking model status:', error);
                        document.getElementById('modelStatus').className = 'status error';
                        document.getElementById('modelStatus').textContent = 'Error connecting to the server. Please check if the service is running.';
                    });
            }
            
            // Analyze text for emotions
            function analyzeText() {
                const text = document.getElementById('text-input').value.trim();
                
                if (!text) {
                    showError('Please enter some text to analyze.');
                    return;
                }
                
                // Show loading indicator
                document.getElementById('loadingIndicator').style.display = 'block';
                document.getElementById('analyzeBtn').disabled = true;
                document.getElementById('resultsContainer').style.display = 'none';
                document.getElementById('errorDisplay').style.display = 'none';
                
                // Send request to API
                fetch('/detect-emotion', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text })
                })
                .then(response => response.json())
                .then(data => {
                    // Hide loading indicator
                    document.getElementById('loadingIndicator').style.display = 'none';
                    document.getElementById('analyzeBtn').disabled = false;
                    
                    if (data.error) {
                        showError(data.error);
                        return;
                    }
                    
                    // Display results
                    document.getElementById('primaryEmotion').textContent = data.primary_emotion;
                    document.getElementById('primaryScore').textContent = data.primary_score.toFixed(4);
                    
                    // Display secondary emotions
                    const secondaryContainer = document.getElementById('secondaryEmotions');
                    secondaryContainer.innerHTML = '';
                    
                    data.secondary_emotions.forEach(emotion => {
                        const div = document.createElement('div');
                        div.className = 'emotion secondary';
                        div.textContent = `${emotion.emotion} (${emotion.score.toFixed(4)})`;
                        secondaryContainer.appendChild(div);
                    });
                    
                    // Display response message
                    document.getElementById('responseMessage').textContent = data.message;
                    
                    // Show results container
                    document.getElementById('resultsContainer').style.display = 'block';
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('loadingIndicator').style.display = 'none';
                    document.getElementById('analyzeBtn').disabled = false;
                    showError('An error occurred while connecting to the server. Please try again later.');
                });
            }
            
            // Show error message
            function showError(message) {
                const errorElement = document.getElementById('errorDisplay');
                errorElement.textContent = message;
                errorElement.style.display = 'block';
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html_content)

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 5002))
    
    # Debug mode based on environment
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    
    logger.info(f"Starting Emotion Detection service on port {port}, debug mode: {debug_mode}")
    app.run(debug=debug_mode, host='0.0.0.0', port=port)