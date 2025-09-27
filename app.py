# app.py - Railway deployment version
import os
from flask import Flask, render_template, request, jsonify
from local_rag_chromadb2 import LocalRAGChatbot

app = Flask(__name__)

# Initialize chatbot
try:
    chatbot = LocalRAGChatbot()
    print("Chatbot initialized successfully")
except Exception as e:
    print(f"Chatbot initialization failed: {e}")
    chatbot = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    if not chatbot:
        return jsonify({'error': 'Chatbot not available'}), 500
        
    data = request.get_json()
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    print(f"Received query: {query}")
    
    try:
        result = chatbot.chat_with_routing(query)
        print(f"Sending response: {result}")
        return jsonify(result)
    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Railway deployment configuration
    port = int(os.environ.get('PORT', 5000))
    
    print(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)