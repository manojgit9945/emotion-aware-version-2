import requests
import os
import json
import logging
import random
from dotenv import load_dotenv
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("content_query_generator")

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    logger.error("API key not found! Please set OPENROUTER_API_KEY in your .env file")
    raise EnvironmentError("Missing API key")

OPENROUTER_API = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "deepseek/deepseek-prover-v2:free"

class ContentType(Enum):
    """Enum for different types of content platforms"""
    VIDEO = "video"
    ARTICLE = "article"
    PODCAST = "podcast"
    BOOK = "book"
    MUSIC = "music"
    COURSE = "course"
    COMMUNITY = "community"
    APP = "app"

# Content type mapping for search platforms
PLATFORM_MAP = {
    ContentType.VIDEO: ["YouTube", "Vimeo", "TED Talks"],
    ContentType.ARTICLE: ["Medium", "Psychology Today", "The Guardian", "Harvard Health"],
    ContentType.PODCAST: ["Spotify podcasts", "Apple podcasts", "Google podcasts"],
    ContentType.BOOK: ["Goodreads", "Amazon books", "Audible"],
    ContentType.MUSIC: ["Spotify playlists", "YouTube Music", "Apple Music"],
    ContentType.COURSE: ["Coursera", "Udemy", "Khan Academy", "edX"],
    ContentType.COMMUNITY: ["Reddit", "Quora", "Discord communities"],
    ContentType.APP: ["App Store", "Google Play Store", "meditation apps"]
}

def clean_query(raw_text):
    """
    Removes unwanted characters and formats the query appropriately.
    
    Args:
        raw_text (str): The raw text from the AI response
        
    Returns:
        str: A clean, formatted search query
    """
    if not raw_text or not isinstance(raw_text, str):
        return "emotional support"  # Default fallback
        
    q = raw_text.strip()
    
    # Remove wrapping quotes and any leading/trailing spaces
    q = q.strip('"\'')
    
    # Remove any unwanted prefixes like "user:", etc.
    if ":" in q:
        q = q.split(":")[-1].strip()
    
    # Ensure lowercase for consistency in searching
    q = q.lower().strip()
    
    # Remove any special characters that might affect search results
    for char in ['?', '!', '.', ',']:
        q = q.replace(char, '')
    
    return q

def determine_best_content_types(emotion, message_text):
    """
    Determine the most appropriate content types based on emotion and message.
    
    Args:
        emotion (str): The identified emotion
        message_text (str): The user's message
        
    Returns:
        list: List of ContentType enums ordered by relevance
    """
    # Basic mapping of emotions to likely helpful content types
    emotion_content_map = {
        "sad": [ContentType.MUSIC, ContentType.VIDEO, ContentType.ARTICLE, ContentType.COMMUNITY],
        "anxious": [ContentType.VIDEO, ContentType.APP, ContentType.ARTICLE, ContentType.PODCAST],
        "angry": [ContentType.MUSIC, ContentType.ARTICLE, ContentType.VIDEO, ContentType.APP],
        "stressed": [ContentType.APP, ContentType.MUSIC, ContentType.PODCAST, ContentType.VIDEO],
        "happy": [ContentType.MUSIC, ContentType.VIDEO, ContentType.COMMUNITY, ContentType.PODCAST],
        "scared": [ContentType.ARTICLE, ContentType.COMMUNITY, ContentType.VIDEO, ContentType.PODCAST],
        "confused": [ContentType.ARTICLE, ContentType.COURSE, ContentType.VIDEO, ContentType.COMMUNITY],
        "lonely": [ContentType.COMMUNITY, ContentType.PODCAST, ContentType.VIDEO, ContentType.APP],
        "overwhelmed": [ContentType.APP, ContentType.ARTICLE, ContentType.VIDEO, ContentType.MUSIC],
        "motivated": [ContentType.COURSE, ContentType.BOOK, ContentType.PODCAST, ContentType.VIDEO]
    }
    
    # Default content types if emotion isn't in our map
    default_types = [ContentType.VIDEO, ContentType.ARTICLE, ContentType.PODCAST, ContentType.APP]
    
    # Get content types for this emotion, or use defaults
    return emotion_content_map.get(emotion.lower(), default_types)

def get_search_queries(user_text, emotion, num_results=3):
    """
    Generates search queries for multiple content platforms based on user text and emotional state.
    
    Args:
        user_text (str): The user's input message
        emotion (str): The detected emotion (e.g., "sad", "happy", "anxious")
        num_results (int): Number of different content sources to suggest
        
    Returns:
        list: A list of dictionaries containing content type, platform, and search query
    """
    logger.info(f"Generating queries for: '{user_text}' (emotion: {emotion})")
    
    # Guard clauses for invalid inputs
    if not user_text or not emotion:
        logger.warning("Missing user text or emotion, using fallback")
        return [{"type": "video", "platform": "YouTube", "query": f"{emotion or 'relaxing'} support video"}]
    
    # Determine best content types for this emotion
    content_types = determine_best_content_types(emotion, user_text)
    
    # Take the top types based on num_results
    selected_types = content_types[:min(num_results, len(content_types))]
    
    results = []
    
    for content_type in selected_types:
        # Select a platform for this content type
        platform = random.choice(PLATFORM_MAP[content_type])
        
        # System message: Clear and targeted instructions for the model
        system_msg = {
            "role": "system",
            "content": (
                f"You generate search queries for finding {content_type.value}s on {platform}. "
                f"Given the user's emotional state and message, generate ONLY a short, focused "
                f"search phrase (3-5 words) that would find {content_type.value} content to support their emotional needs. "
                f"Output the raw search phrase with no quotes, explanations, or additional text. "
                f"Focus on finding appropriate content that addresses their emotional state."
            )
        }
        
        # User message: just the text and emotion for context
        user_msg = {
            "role": "user",
            "content": f"User's message: {user_text} | Emotion: {emotion} | Platform: {platform}"
        }
        
        try:
            # Send the request to OpenRouter AI
            resp = requests.post(
                OPENROUTER_API,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://emotional-support-app.com",  # Identifying your application
                    "X-Title": "Multi-Platform Content Recommender"  # App name
                },
                json={  # Use json parameter instead of data+dumps
                    "model": MODEL_NAME,
                    "messages": [system_msg, user_msg],
                    "temperature": 0.2,  # Slightly more creative but still focused
                    "max_tokens": 12,    # Short output
                    "stop": ["\n", "."]
                },
                timeout=8  # Reasonable timeout
            )
            
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            logger.info(f"Raw AI output for {content_type.value}: {raw!r}")
            
            # Clean up the raw output from the AI
            query = clean_query(raw)
            logger.info(f"Cleaned query for {platform}: {query!r}")
            
            # Validate: the query should be 2–6 words
            words = query.split()
            if len(words) < 2:
                query = f"{emotion} {content_type.value}"
            elif len(words) > 6:
                query = " ".join(words[:5])  # Take first 5 words
            
            results.append({
                "type": content_type.value,
                "platform": platform,
                "query": query
            })
            
        except Exception as e:
            logger.error(f"Error generating query for {content_type.value}: {e}")
            # Add fallback query
            results.append({
                "type": content_type.value,
                "platform": platform,
                "query": f"{emotion} {content_type.value}"
            })
    
    return results

def format_recommendations(query_results):
    """
    Formats the query results into a user-friendly recommendations list.
    
    Args:
        query_results (list): List of query result dictionaries
        
    Returns:
        str: Formatted recommendations text
    """
    output = "Here are some content recommendations that might help:\n\n"
    
    for i, result in enumerate(query_results, 1):
        content_type = result["type"]
        platform = result["platform"]
        query = result["query"]
        
        # Add emoji based on content type
        emoji = {
            "video": "📺",
            "article": "📝",
            "podcast": "🎧",
            "book": "📚",
            "music": "🎵",
            "course": "🎓",
            "community": "👥",
            "app": "📱"
        }.get(content_type, "✨")
        
        output += f"{i}. {emoji} {platform}: \"{query}\"\n"
    
    output += "\nThese search terms are designed to find content that addresses your current emotional needs."
    return output

# Example usage
if __name__ == "__main__":
    test_cases = [
        ("I feel so alone and nobody understands me", "sad"),
        ("I can't stop worrying about my exam tomorrow", "anxious"),
        ("I'm so excited about my new job!", "happy"),
        ("I'm feeling overwhelmed with all my responsibilities", "overwhelmed")
    ]
    
    print("=== TESTING MULTI-CONTENT QUERY GENERATOR ===")
    for text, emotion in test_cases:
        print(f"\nInput: '{text}' ({emotion})")
        print("-" * 40)
        
        # Get multiple content recommendations
        recommendations = get_search_queries(text, emotion, num_results=3)
        
        # Format and display recommendations
        print(format_recommendations(recommendations))
        print("=" * 60)