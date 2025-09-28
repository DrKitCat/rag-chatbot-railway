# app.py - Railway deployment version
import os
from flask import Flask, render_template, request, jsonify
from local_rag_chromadb2 import LocalRAGChatbot
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit

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

# @app.route('/chat', methods=['POST'])
# def chat():
#     if not chatbot:
#         return jsonify({'error': 'Chatbot not available'}), 500
        
#     data = request.get_json()
#     query = data.get('query', '')
    
#     if not query:
#         return jsonify({'error': 'No query provided'}), 400
    
#     print(f"Received query: {query}")
    
#     try:
#         result = chatbot.chat_with_routing(query)
#         print(f"Sending response: {result}")
#         return jsonify(result)
#     except Exception as e:
#         print(f"Error processing query: {e}")
#         return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
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

# @app.route('/upload')
# def upload_page():
#     return '''
#     <html><body>
#     <h2>Upload CIRD Manual PDFs</h2>
#     <form action="/upload" method="post" enctype="multipart/form-data">
#         <input type="file" name="files[]" multiple accept=".pdf">
#         <input type="submit" value="Upload PDFs">
#     </form>
#     </body></html>
#     '''


# @app.route('/collection-info')
# def collection_info():
#     if not chatbot:
#         return jsonify({'error': 'Chatbot not available'}), 500
    
#     try:
#         count = chatbot.get_collection_info()
#         return jsonify({
#             'total_documents': count,
#             'status': 'Documents loaded and ready for queries'
#         })
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Railway deployment configuration
    port = int(os.environ.get('PORT', 5000))
    
    print(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)