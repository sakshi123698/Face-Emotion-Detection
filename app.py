from flask import Flask, request, jsonify
from deepface import DeepFace
import base64
from io import BytesIO
from PIL import Image
import numpy as np

from flask_cors import CORS



app = Flask(__name__)
CORS(app)
@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    try:
        # Get the image from the frontend (base64)
        data = request.get_json()
        img_data = data['image']
        
        # Decode the image from base64
        img_data = img_data.split(',')[1]  # Remove the 'data:image/png;base64,' part
        img_bytes = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_bytes))
        
        # Convert the image to RGB and resize
        img = img.convert("RGB")
        img = img.resize((224, 224))  # Resize image to standard size (DeepFace expects this)
        
        # Convert image to numpy array (required by DeepFace)
        img_array = np.array(img)
        
        # Use DeepFace to analyze the emotion of the image
        analysis = DeepFace.analyze(img_array, actions=['emotion'], enforce_detection=False)
        
        # Extract the dominant emotion
        emotion = analysis[0]['dominant_emotion']
        return jsonify({"emotion": emotion})
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
