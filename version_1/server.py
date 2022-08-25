from flask import render_template
import connexion
import pipeline
from flask import Flask, jsonify, request, abort
from base64 import b64encode, b64decode
import io
import tensorflow as tf
#from werkzeug import secure_filename
import json
import numpy as np
#app = Flask(__name__)
import pickle

# Create the application instance
app = connexion.App(__name__, specification_dir='./')

app.add_api('swagger.yml')
# Read the swagger.yml file to configure the endpoints
#app.add_api('swagger.yml')
# Create a URL route in our application for "/"
@app.route('/')
def home():
 """
 This function just responds to the browser ULR
 localhost:5000/
 :return: the rendered template 'home.html'
 """
 return render_template('home.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(f.filename)
      pred_colored = pipeline.predict(f.filename)
      file_object = io.BytesIO()
      img= tf.keras.utils.array_to_img(pred_colored.astype('uint8'))
      img.save(file_object, 'PNG')
      #encoded_string= b64encode(file_object.getvalue())
      base64img = "data:image/png;base64,"+b64encode(file_object.getvalue()).decode('utf-8')
      return render_template('visualization.html', img_data=base64img)


@app.route('/predict', methods = ['POST'])
def make_prediction():
    # get the base64 encoded string
    im_b64 = request.json['image']
    #deserialized_from_json = pickle.loads(json.loads(im_b64))
    deserialized_from_json = pickle.loads(im_b64.encode('latin-1'))
    img= tf.keras.utils.array_to_img(deserialized_from_json)
    img.save("file_traitement.jpg")
    # convert bytes data to PIL Image object
    pred_colored = pipeline.predict("file_traitement.jpg")
    #file_object = io.BytesIO()
    img= tf.keras.utils.img_to_array(pred_colored.astype('uint8'))
    serialized_as_json = json.dumps(pickle.dumps(img).decode('latin-1'))
    #img.save(file_object, 'PNG')
    #encoded_string= b64encode(file_object.getvalue())
    #base64img = "data:image/png;base64,"+b64encode(file_object.getvalue()).decode('utf-8')
    #base64_image = b64encode(file_object.getvalue())
    #jsonify(image=base64_image, total_count=1)
    return serialized_as_json

@app.route("/test", methods=['POST'])
def test_method():         
    # print(request.json)      
    if not request.json or 'image' not in request.json: 
        abort(400)
             
    # get the base64 encoded string
    im_b64 = request.json['image']

    # convert it into bytes  
    img_bytes = b64decode(im_b64.encode('utf-8'))

    # convert bytes data to PIL Image object
    img = tf.keras.utils.img_to_array(io.BytesIO(img_bytes))
    return print('img shape', img.shape)

 
if __name__ == '__main__':
    app.run()


# If we're running in stand alone mode, run the application
if __name__ == '__main__':
 app.run(host='0.0.0.0', port=5000, debug=True)