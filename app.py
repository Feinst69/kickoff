from flask import Flask, request, jsonify, render_template_string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io

app = Flask(__name__)
model = load_model('model.h5')

HTML = """
<!doctype html>
<title>Digit Classifier</title>
<h1>Upload an image</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
{% if pred is not none %}
<p>Predicted digit: {{ pred }}</p>
{% endif %}
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    pred = None
    if request.method == 'POST':
        f = request.files['file']
        if f:
            img = image.load_img(io.BytesIO(f.read()), target_size=(28,28), color_mode='grayscale')
            img_array = image.img_to_array(img).reshape(1,28,28,1) / 255.0
            out = model.predict(img_array)
            pred = int(np.argmax(out))
    return render_template_string(HTML, pred=pred)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'no file provided'}), 400
    img = image.load_img(io.BytesIO(file.read()), target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img).reshape(1, 28, 28, 1) / 255.0
    out = model.predict(img_array)
    pred = int(np.argmax(out))
    return jsonify({'prediction': pred, 'probabilities': out[0].tolist()})

if __name__ == '__main__':
    app.run(debug=True)
