# import configparser,os
# import tensorflow as tf
# import tensorflow_hub as hub
# from flask import Flask, request, render_template


# app = Flask(__name__)

# # Load configuration from config.ini
# config = configparser.ConfigParser()
# config.read('config/config.ini')
# model_path = config['Model']['model_path']

# # General Settings
# ALLOWED_EXTENSIONS = set(config['General']['allowed_extensions'].split(','))
# threshold = float(config['General']['threshold'])  # Convert threshold to float

# # Load the trained model weights
# trained_model_weights = model_path
# module_url = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
# m = tf.keras.Sequential([
#     hub.KerasLayer(module_url, output_shape=[2048], trainable=False),
#     tf.keras.layers.Dense(1, activation="sigmoid")
# ])
# m.build([None, 299, 299, 3])
# m.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
# m.load_weights(trained_model_weights)

# def is_allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def predict_image_class(img_path, model, threshold=threshold):
#     img = tf.keras.preprocessing.image.load_img(img_path, target_size=(299, 299))
#     img = tf.keras.preprocessing.image.img_to_array(img)
#     img = tf.expand_dims(img, 0)
#     img = tf.keras.applications.inception_v3.preprocess_input(img)
#     img = tf.image.convert_image_dtype(img, tf.float32)
#     predictions = model.predict(img)
#     score = predictions.squeeze()
#     if score >= threshold:
#         result = f"{100 * score:.2f}% malignant"
#     else:
#         result = f"{100 * (1 - score):.2f}% benign"
    
#     return result

# @app.route('/', methods=['GET'])
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return "No file part"

#     file = request.files['file']

#     if file.filename == '':
#         return "No selected file"

#     if not is_allowed_file(file.filename):
#         return "Invalid file type. Allowed types: jpeg, jpg"

#     # Save the file
#     file_path = 'image.jpg'
#     img_path = os.path.join("media/image", file_path)
#     os.makedirs(os.path.dirname(img_path), exist_ok=True)
#     file.save(img_path)


#     result = predict_image_class(img_path, m)

#     # Render the HTML template with prediction and image path
#     return render_template('result.html', prediction=result, image_path=file_path)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=80, debug=False)

import os
import tensorflow as tf
import tensorflow_hub as hub
import configparser
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Load configuration from config.ini
config = configparser.ConfigParser()
config.read('config/config.ini')
model_path = config['Model']['model_path']

# General Settings
ALLOWED_EXTENSIONS = set(config['General']['allowed_extensions'].split(','))
threshold = float(config['General']['threshold'])  # Convert threshold to float

# Load the trained model weights
trained_model_weights = model_path
module_url = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
m = tf.keras.Sequential([
    hub.KerasLayer(module_url, output_shape=[2048], trainable=False),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
m.build([None, 299, 299, 3])
m.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
m.load_weights(trained_model_weights)

def is_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image_class(img_path, model, threshold=threshold):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(299, 299))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.expand_dims(img, 0)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    predictions = model.predict(img)
    score = predictions.squeeze()
    if score >= threshold:
        result = f"{100 * score:.2f}% malignant"
    else:
        result = f"{100 * (1 - score):.2f}% benign"
    
    return result

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part"})
        
        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"})

        if not is_allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Allowed types: jpeg, jpg"})

        # حفظ الملف
        file_path = 'image.jpg'
        img_path = os.path.join("static/media/image", file_path)
        img_path = img_path.replace("\\", "/")  # استخدم "/" بدلاً من "\"
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        file.save(img_path)

        result = predict_image_class(img_path, m)

        # 
        return jsonify({"prediction": result, "image_path": img_path})
    else:
        # عند الوصول من المتصفح باستخدام GET، عرض صفحة "result.html"
        return render_template('result.html', prediction=result, image_path=img_path), 200, {
    'Cache-Control': 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=False)

