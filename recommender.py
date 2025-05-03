from flask import Flask, request, jsonify
import sys
import traceback
import logging
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("content_recommender")

# Import the AI-based query generator
try:
    from query_agent import get_search_queries
    logger.info("✅ Successfully imported query_agent module")
except ImportError as e:
    logger.error(f"❌ ERROR importing query_agent: {e}")
    logger.error("Make sure query_agent.py is in the same directory")
    sys.exit(1)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Platform-specific URL builders
PLATFORM_URL_BUILDERS = {
    "YouTube": lambda q: f"https://www.youtube.com/results?search_query={q.replace(' ', '+')}",
    "Vimeo": lambda q: f"https://vimeo.com/search?q={q.replace(' ', '+')}",
    "TED Talks": lambda q: f"https://www.ted.com/search?q={q.replace(' ', '+')}",
    "Medium": lambda q: f"https://medium.com/search?q={q.replace(' ', '%20')}",
    "Psychology Today": lambda q: f"https://www.psychologytoday.com/us/search/site/{q.replace(' ', '%20')}",
    "The Guardian": lambda q: f"https://www.theguardian.com/search?q={q.replace(' ', '+')}",
    "Harvard Health": lambda q: f"https://www.health.harvard.edu/search?q={q.replace(' ', '+')}",
    "Spotify podcasts": lambda q: f"https://open.spotify.com/search/{q.replace(' ', '%20')}/podcasts",
    "Apple podcasts": lambda q: f"https://podcasts.apple.com/search?term={q.replace(' ', '+')}",
    "Google podcasts": lambda q: f"https://podcasts.google.com/search/{q.replace(' ', '%20')}",
    "Goodreads": lambda q: f"https://www.goodreads.com/search?q={q.replace(' ', '+')}",
    "Amazon books": lambda q: f"https://www.amazon.com/s?k={q.replace(' ', '+')}&i=stripbooks",
    "Audible": lambda q: f"https://www.audible.com/search?keywords={q.replace(' ', '+')}",
    "Spotify playlists": lambda q: f"https://open.spotify.com/search/{q.replace(' ', '%20')}/playlists",
    "YouTube Music": lambda q: f"https://music.youtube.com/search?q={q.replace(' ', '+')}",
    "Apple Music": lambda q: f"https://music.apple.com/search?term={q.replace(' ', '+')}",
    "Coursera": lambda q: f"https://www.coursera.org/search?query={q.replace(' ', '%20')}",
    "Udemy": lambda q: f"https://www.udemy.com/courses/search/?q={q.replace(' ', '+')}",
    "Khan Academy": lambda q: f"https://www.khanacademy.org/search?page_search_query={q.replace(' ', '+')}",
    "edX": lambda q: f"https://www.edx.org/search?q={q.replace(' ', '+')}",
    "Reddit": lambda q: f"https://www.reddit.com/search/?q={q.replace(' ', '%20')}",
    "Quora": lambda q: f"https://www.quora.com/search?q={q.replace(' ', '%20')}",
    "Discord communities": lambda q: f"https://disboard.org/servers/search?keyword={q.replace(' ', '%20')}",
    "App Store": lambda q: f"https://apps.apple.com/search?term={q.replace(' ', '+')}",
    "Google Play Store": lambda q: f"https://play.google.com/store/search?q={q.replace(' ', '%20')}&c=apps"
}

# Default URL builders for unknown platforms
DEFAULT_URL_BUILDERS = {
    "video": lambda q: f"https://www.youtube.com/results?search_query={q.replace(' ', '+')}",
    "article": lambda q: f"https://medium.com/search?q={q.replace(' ', '%20')}",
    "podcast": lambda q: f"https://open.spotify.com/search/{q.replace(' ', '%20')}/podcasts",
    "book": lambda q: f"https://www.goodreads.com/search?q={q.replace(' ', '+')}",
    "music": lambda q: f"https://open.spotify.com/search/{q.replace(' ', '%20')}/playlists",
    "course": lambda q: f"https://www.coursera.org/search?query={q.replace(' ', '%20')}",
    "community": lambda q: f"https://www.reddit.com/search/?q={q.replace(' ', '%20')}",
    "app": lambda q: f"https://play.google.com/store/search?q={q.replace(' ', '%20')}&c=apps"
}

# Friendly descriptions for content types
CONTENT_DESCRIPTIONS = {
    "video": "Videos selected to help with your current emotional state",
    "article": "Articles with insights and advice for your situation",
    "podcast": "Podcasts that discuss similar emotional experiences",
    "book": "Books that might provide guidance and understanding",
    "music": "Music playlists to match and improve your mood",
    "course": "Online courses to develop skills for handling your emotions",
    "community": "Communities where you can connect with others feeling similar",
    "app": "Apps designed to help with your emotional wellbeing"
}

# Emoji indicators for content types
CONTENT_EMOJIS = {
    "video": "📺",
    "article": "📝",
    "podcast": "🎧",
    "book": "📚",
    "music": "🎵",
    "course": "🎓",
    "community": "👥",
    "app": "📱"
}

def build_recommendation(content_type, platform, query):
    """
    Builds a content recommendation from a dynamic AI-generated query.
    
    Args:
        content_type (str): Type of content (video, article, etc.)
        platform (str): Platform name (YouTube, Medium, etc.)
        query (str): Search query string
        
    Returns:
        dict: Recommendation object with URL, title, etc.
    """
    logger.info(f"Building {content_type} recommendation for: {query} on {platform}")
    
    # Get the appropriate URL builder for this platform
    url_builder = PLATFORM_URL_BUILDERS.get(platform)
    
    # If platform not found, use default URL builder for this content type
    if not url_builder and content_type in DEFAULT_URL_BUILDERS:
        url_builder = DEFAULT_URL_BUILDERS[content_type]
        logger.warning(f"No URL builder for {platform}, using default for {content_type}")
    
    # Fallback to YouTube if we can't find any appropriate URL builder
    if not url_builder:
        url_builder = lambda q: f"https://www.youtube.com/results?search_query={q.replace(' ', '+')}"
        logger.warning(f"No URL builders found, defaulting to YouTube")
    
    # Build the search URL
    search_url = url_builder(query)
    
    # Get emoji and description
    emoji = CONTENT_EMOJIS.get(content_type, "✨")
    description = CONTENT_DESCRIPTIONS.get(content_type, "Resources selected to help with your current emotional state")
    
    # Build recommendation object
    return {
        "title": f"{emoji} {query.title()}",
        "url": search_url,
        "snippet": description,
        "platform": platform,
        "type": content_type,
        "query": query,
        "emoji": emoji
    }

@app.route("/recommend", methods=["POST"])
def recommend():
    logger.info("Received /recommend request")

    # Validate request
    if not request.is_json:
        logger.error("Request is not JSON")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    user_text = data.get("text", "")
    emotion = data.get("emotion", "neutral")
    
    # Optional parameter for number of recommendations (default: 3)
    num_recommendations = data.get("num_recommendations", 3)
    
    # Ensure num_recommendations is an integer between 1 and 5
    try:
        num_recommendations = int(num_recommendations)
        num_recommendations = max(1, min(5, num_recommendations))  # Clamp between 1 and 5
    except (ValueError, TypeError):
        num_recommendations = 3

    if not user_text:
        logger.error("Missing 'text' field in request")
        return jsonify({"error": "Missing 'text' field"}), 400

    logger.info(f"Processing with text='{user_text}', emotion='{emotion}', num_recommendations={num_recommendations}")

    try:
        # 1️⃣ Get AI-generated search recommendations for multiple platforms
        query_results = get_search_queries(user_text, emotion, num_results=num_recommendations)
        logger.info(f"Queries generated by AI: {query_results}")

        # 2️⃣ Build dynamic recommendations for each result
        recommendations = []
        for result in query_results:
            content_type = result["type"]
            platform = result["platform"]
            query = result["query"]
            
            recommendation = build_recommendation(content_type, platform, query)
            recommendations.append(recommendation)
        
        logger.info(f"Generated {len(recommendations)} recommendations")

        return jsonify({
            "recommendations": recommendations,
            "emotion": emotion,
            "text": user_text
        })

    except Exception as e:
        logger.error(f"ERROR in /recommend endpoint: {str(e)}")
        traceback.print_exc()

        return jsonify({
            "error": "Content recommendation generation failed.",
            "recommendations": None
        }), 500

@app.route("/test", methods=["GET"])
def test_endpoint():
    logger.info("/test endpoint called")
    return jsonify({
        "status": "ok", 
        "message": "Multi-Platform Content Recommender API is running",
        "supported_platforms": list(PLATFORM_URL_BUILDERS.keys()),
        "supported_content_types": list(DEFAULT_URL_BUILDERS.keys())
    })

if __name__ == "__main__":
    logger.info("🚀 Starting Multi-Platform Content Recommender API on http://localhost:5004")
    app.run(host="0.0.0.0", port=5004, debug=True)