# app.py - Railway deployment version
import os
from flask import Flask, render_template, request, jsonify
from local_rag_chromadb2 import LocalRAGChatbot
from werkzeug.utils import secure_filename
from collections import defaultdict


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit



# Per-IP rate limiting
ip_usage = defaultdict(int)

@app.before_request 
def limit_per_ip():
    client_ip = request.remote_addr
    if ip_usage[client_ip] >= 100:  # 100 requests per IP
        return jsonify({'error': 'Personal limit reached.'}), 429
    ip_usage[client_ip] += 1


# Per-IP rate limiting
ip_usage = defaultdict(int)



# Initialize chatbot
try:
    chatbot = LocalRAGChatbot()
    print("Chatbot initialized successfully")
except Exception as e:
    print(f"Chatbot initialization failed: {e}")
    chatbot = None

# Process PDFs on startup if collection is empty
try:
    if chatbot and chatbot.get_collection_info() == 0:
        print("Collection is empty. Processing PDFs from repository...")
        chatbot.process_pdf_folder("./pdfs")
        print(f"Loaded {chatbot.get_collection_info()} documents")
except Exception as e:
    print(f"Error processing PDFs on startup: {e}")



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
        # Rate limiting for chat endpoint only
    client_ip = request.remote_addr
    if ip_usage[client_ip] >= 100:  # 100 chat requests per IP
        return jsonify({'error': 'Personal limit reached.'}), 429
    ip_usage[client_ip] += 1
    print("=== CHAT ENDPOINT HIT ===")
    print(f"Request method: {request.method}")
    print(f"Content-Type: {request.content_type}")
    
    if not chatbot:
        print("ERROR: Chatbot not available")
        return jsonify({'error': 'Chatbot not available'}), 500
    
    try:
        data = request.get_json()
        print(f"Received data: {data}")
        
        query = data.get('query', '')
        print(f"Query: {query}")
        
        if not query:
            print("ERROR: No query provided")
            return jsonify({'error': 'No query provided'}), 400
        
        print("Calling chatbot.chat_with_routing...")
        result = chatbot.chat_with_routing(query)
        print(f"Result: {result}")
        
        return jsonify(result)
    except Exception as e:
        print(f"EXCEPTION in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500



@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

@app.route('/upload', methods=['POST'])
def upload_pdfs():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files selected'}), 400
    
    files = request.files.getlist('files[]')
    processed = 0
    
    for file in files:
        if file.filename.endswith('.pdf'):
            # Save temporarily and process
            temp_path = f"/tmp/{file.filename}"
            file.save(temp_path)
            
            if chatbot.process_pdf_file(temp_path):
                processed += 1
            
            os.remove(temp_path)  # Clean up
    
    return jsonify({
        'message': f'Processed {processed} documents',
        'total_docs': chatbot.get_collection_info()
    })




@app.route('/collection-info')
def collection_info():
    if not chatbot:
        return jsonify({'error': 'Chatbot not available'}), 500
    
    try:
        count = chatbot.get_collection_info()
        return jsonify({
            'total_documents': count,
            'status': 'Documents loaded and ready for queries'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Railway deployment configuration
    port = int(os.environ.get('PORT', 5000))
    
    print(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)