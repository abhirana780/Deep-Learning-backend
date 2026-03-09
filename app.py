from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import time
import os
import webbrowser
from threading import Timer

app = Flask(__name__, static_folder="../frontend", static_url_path="/")
CORS(app)

# ─── Simple CNN Simulation ───────────────────────────────────────────────────
# Maps image keys to predicted labels with confidence scores
IMAGE_PREDICTIONS = {
    # Shape images
    "circle":    {"label": "Circle",    "category": "Shape",      "confidence": 97, "description": "Detected curved edges and round symmetry."},
    "square":    {"label": "Square",    "category": "Shape",      "confidence": 94, "description": "Detected four equal straight edges and right angles."},
    "triangle":  {"label": "Triangle",  "category": "Shape",      "confidence": 96, "description": "Detected three edges meeting at sharp angles."},
    # Product images
    "laptop":    {"label": "Laptop",    "category": "Electronics","confidence": 93, "description": "Detected keyboard grid, screen boundary and hinge patterns."},
    "shirt":     {"label": "T-Shirt",   "category": "Clothing",   "confidence": 91, "description": "Detected fabric texture, collar shape and sleeve patterns."},
    "pizza":     {"label": "Pizza",     "category": "Food",       "confidence": 95, "description": "Detected circular dough shape, cheese texture and topping colours."},
    "phone":     {"label": "Smartphone","category": "Electronics","confidence": 98, "description": "Detected rectangular screen, camera lens and button patterns."},
    "sneaker":   {"label": "Sneaker",   "category": "Footwear",   "confidence": 92, "description": "Detected sole shape, lace pattern and curved upper."},
    "cat":       {"label": "Cat",       "category": "Animal",     "confidence": 96, "description": "Detected fur texture, pointed ears and whisker patterns."},
    "dog":       {"label": "Dog",       "category": "Animal",     "confidence": 94, "description": "Detected snout shape, ear position and fur texture."},
}

# ─── Simple RNN / Next-Word Simulation ───────────────────────────────────────
# Business-context sentence completions
SENTENCE_COMPLETIONS = {
    "our company sales are increasing because of": ["marketing", "innovation", "strategy", "customers", "quality"],
    "the customer satisfaction score improved after": ["training", "feedback", "upgrades", "support", "changes"],
    "our new product launch was successful due to": ["branding", "research", "pricing", "teamwork", "planning"],
    "the quarterly revenue declined because of": ["competition", "costs", "delays", "shortages", "inflation"],
    "our employees are more productive when they": ["collaborate", "train", "communicate", "innovate", "focus"],
    "artificial intelligence is transforming the way we": ["work", "learn", "sell", "communicate", "operate"],
    "deep learning helps businesses by": ["automating", "predicting", "classifying", "improving", "optimizing"],
    "customers love our new": ["product", "service", "features", "design", "experience"],
    "our sales increased after launching our new marketing": ["campaign", "strategy", "initiative", "approach", "program"],
    "the best way to grow a business is to": ["innovate", "listen", "invest", "plan", "adapt"],
}

# Fallback words for unknown sentences
FALLBACK_WORDS = ["strategy", "growth", "innovation", "performance", "results", "impact", "value", "success"]


@app.route("/", methods=["GET"])
def index():
    return app.send_static_file('index.html')

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "Deep Learning Explorer API is running", "version": "1.0"})


@app.route("/image_predict", methods=["POST"])
def image_predict():
    """
    Accept an image key and return CNN-style prediction with confidence.
    Body: { "image": "laptop" }
    """
    data = request.get_json(force=True, silent=True) or {}
    image_key = data.get("image", "").lower().strip()

    if image_key in IMAGE_PREDICTIONS:
        result = IMAGE_PREDICTIONS[image_key]
        # Simulate slight randomness in confidence (±2%)
        confidence = max(80, min(99, result["confidence"] + random.randint(-2, 2)))
        return jsonify({
            "success": True,
            "image": image_key,
            "label": result["label"],
            "category": result["category"],
            "confidence": confidence,
            "description": result["description"],
            "pipeline": [
                {"stage": "Input Layer",        "detail": f"Raw pixels of {result['label']} image loaded"},
                {"stage": "Edge Detection",      "detail": "CNN Layer 1 detected basic edges and outlines"},
                {"stage": "Pattern Recognition", "detail": "CNN Layer 2 identified shapes and textures"},
                {"stage": "Feature Mapping",     "detail": "CNN Layer 3 combined features into patterns"},
                {"stage": "Classification",      "detail": f"Output Layer predicted: {result['label']} ({confidence}% confidence)"},
            ]
        })
    else:
        return jsonify({
            "success": False,
            "error": f"Image '{image_key}' not found. Try: circle, square, triangle, laptop, shirt, pizza, phone, sneaker, cat, dog"
        }), 400


@app.route("/next_word", methods=["POST"])
def next_word():
    """
    Accept a sentence and return RNN-style next-word prediction.
    Body: { "sentence": "Our company sales are increasing because of" }
    """
    data = request.get_json(force=True, silent=True) or {}
    sentence = data.get("sentence", "").lower().strip().rstrip(".")

    # Try to find exact or partial match
    predicted_word = None
    matched_key = None

    for key, words in SENTENCE_COMPLETIONS.items():
        if key in sentence or sentence.endswith(key[-20:]):
            predicted_word = words[0]
            matched_key = key
            all_predictions = words
            break

    if not predicted_word:
        # Partial keyword matching
        sentence_words = set(sentence.split())
        best_score = 0
        for key, words in SENTENCE_COMPLETIONS.items():
            key_words = set(key.split())
            score = len(sentence_words & key_words)
            if score > best_score:
                best_score = score
                predicted_word = words[0]
                all_predictions = words
                matched_key = key

    if not predicted_word:
        predicted_word = random.choice(FALLBACK_WORDS)
        all_predictions = random.sample(FALLBACK_WORDS, min(4, len(FALLBACK_WORDS)))

    # Build memory chain (shows how RNN remembers context)
    words_in_sentence = sentence.split()
    memory_chain = []
    for i, word in enumerate(words_in_sentence[-5:]):  # last 5 words
        memory_chain.append({
            "word": word,
            "memory_weight": round(0.4 + (i * 0.12), 2),
            "influence": "high" if i >= len(words_in_sentence) - 3 else "medium"
        })

    return jsonify({
        "success": True,
        "input_sentence": data.get("sentence", ""),
        "predicted_word": predicted_word,
        "all_predictions": all_predictions[:5],
        "memory_chain": memory_chain,
        "explanation": f"The RNN analysed the sequence of words and found '{predicted_word}' is the most likely next word based on the business context pattern."
    })


@app.route("/explain", methods=["GET"])
def explain():
    """Return simple explanations for CNN and RNN."""
    topic = request.args.get("topic", "all").lower()

    explanations = {
        "cnn": {
            "title": "Convolutional Neural Network (CNN)",
            "simple_name": "The Vision Expert",
            "analogy": "CNNs work like your eyes and brain together. Just as you recognise a friend's face by noticing eyes, nose, and mouth — CNNs scan images layer by layer to find patterns.",
            "how_it_works": [
                "Step 1: The image is broken into tiny pixel grids.",
                "Step 2: Layer 1 detects simple edges (horizontal, vertical, diagonal).",
                "Step 3: Layer 2 combines edges into shapes (circles, corners, curves).",
                "Step 4: Layer 3 combines shapes into objects (a face, a product, an animal).",
                "Step 5: The final layer makes a decision — 'This is a laptop!'",
            ],
            "business_uses": [
                "E-commerce product categorisation",
                "Medical scan analysis",
                "Quality control in manufacturing",
                "Facial recognition for security",
                "Self-driving car vision systems",
            ],
        },
        "rnn": {
            "title": "Recurrent Neural Network (RNN)",
            "simple_name": "The Memory Expert",
            "analogy": "RNNs work like reading a book. To understand the last sentence, you need to remember what you read before. RNNs have a built-in memory that carries information forward.",
            "how_it_works": [
                "Step 1: The first word enters the network.",
                "Step 2: The network processes it and stores a memory.",
                "Step 3: The next word enters — and the network uses the stored memory.",
                "Step 4: This continues word by word, building context.",
                "Step 5: After reading the whole sentence, the network predicts the next word.",
            ],
            "business_uses": [
                "Chatbots and virtual assistants",
                "Email auto-complete suggestions",
                "Sales forecasting from historical data",
                "Sentiment analysis of customer reviews",
                "Fraud detection in transaction sequences",
            ],
        },
        "deep_learning": {
            "title": "Deep Learning",
            "simple_name": "The AI Brain",
            "analogy": "Deep Learning is like training a very smart intern. You show them thousands of examples (data), and they gradually learn patterns without you explaining every rule.",
            "how_it_works": [
                "Step 1: Feed large amounts of data into the network.",
                "Step 2: The network tries to find patterns (like a child learning rules).",
                "Step 3: It makes mistakes and corrects itself (learning).",
                "Step 4: After millions of examples, it becomes very accurate.",
                "Step 5: It can now predict or classify new, unseen data.",
            ],
            "business_uses": [
                "Personalised product recommendations (Netflix, Amazon)",
                "Voice assistants (Siri, Alexa, Google)",
                "Automated customer service chatbots",
                "Financial fraud detection",
                "Medical diagnosis and drug discovery",
            ],
        },
    }

    if topic == "all":
        return jsonify({"success": True, "explanations": explanations})
    elif topic in explanations:
        return jsonify({"success": True, "explanation": explanations[topic]})
    else:
        return jsonify({"success": False, "error": f"Topic '{topic}' not found. Use: cnn, rnn, deep_learning, all"}), 400


def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    print("=" * 60)
    print("  Deep Learning Explorer API")
    print("  Running on: http://localhost:5000")
    print("=" * 60)
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        Timer(1, open_browser).start()
    app.run(debug=True, port=5000)
