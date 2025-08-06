from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np
import gc
from threading import Lock
import os
import logging

# Initialize Flask app
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Configuration
MAX_INPUT_LENGTH = 1000
MODEL_NAME = 'paraphrase-MiniLM-L6-v2'
MODEL_CACHE = './model_cache'

# Ensure model cache directory exists
os.makedirs(MODEL_CACHE, exist_ok=True)

# Sample questions
stored_questions = [
    "What is your name?",
    "How can I reset my password?",
    "Where is your office located?",
    "Tell me about your services.",
    "How do I make a payment?"
]

# Model loading setup
model_lock = Lock()
model = None

def get_model():
    global model
    with model_lock:
        if model is None:
            try:
                app.logger.info("Loading sentence transformer model...")
                model = SentenceTransformer(
                    MODEL_NAME,
                    device='cpu',
                    cache_folder=MODEL_CACHE
                )
                # Reduce memory footprint
                model.max_seq_length = 128
                app.logger.info("Model loaded successfully")
            except Exception as e:
                app.logger.error(f"Model loading failed: {str(e)}")
                raise
    return model

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy", "service": "text-similarity"})

@app.route('/model-info', methods=['GET'])
def model_info():
    """Endpoint to verify model loading"""
    try:
        m = get_model()
        return jsonify({
            "status": "loaded",
            "model": MODEL_NAME,
            "device": str(m.device),
            "max_seq_length": m.max_seq_length
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    """Main similarity analysis endpoint"""
    try:
        app.logger.info("Received analysis request")
        
        # Validate input
        if not request.is_json:
            app.logger.warning("Invalid content type")
            return jsonify({"error": "Content-Type must be application/json"}), 400

        data = request.get_json()
        if not data or 'text' not in data:
            app.logger.warning("Missing text field")
            return jsonify({"error": "Missing 'text' in request body"}), 400

        input_text = data['text']
        if len(input_text) > MAX_INPUT_LENGTH:
            app.logger.warning(f"Input too long ({len(input_text)} chars)")
            return jsonify({
                "error": f"Input exceeds maximum length of {MAX_INPUT_LENGTH} characters"
            }), 413

        # Process request
        current_model = get_model()
        app.logger.info("Encoding input text...")
        
        input_embedding = current_model.encode(
            input_text,
            convert_to_tensor=False,
            normalize_embeddings=True,
            batch_size=1,
            show_progress_bar=False
        )

        best_score = -1
        best_match = ""
        best_question = ""

        app.logger.info(f"Comparing against {len(stored_questions)} questions...")
        for question in stored_questions:
            question_embedding = current_model.encode(
                question,
                convert_to_tensor=False,
                batch_size=1,
                show_progress_bar=False
            )
            
            # Manual cosine similarity to save memory
            score = np.dot(input_embedding, question_embedding) / (
                np.linalg.norm(input_embedding) * np.linalg.norm(question_embedding)
            )

            if score > best_score:
                best_score = float(score)
                best_match = question

        app.logger.info(f"Best match found: {best_match[:50]}... (score: {best_score:.2f})")
        
        return jsonify({
            "input": input_text,
            "most_similar": best_match,
            "similarity_score": round(best_score, 4),
            "status": "success"
        })

    except Exception as e:
        app.logger.error(f"Processing failed: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "message": str(e),
            "status": "error"
        }), 500
    finally:
        gc.collect()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port)
