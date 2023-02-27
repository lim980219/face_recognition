import os
import shutil
import flask.scaffold
flask.helpers._endpoint_from_view_func = flask.scaffold._endpoint_from_view_func
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
from werkzeug.datastructures import FileStorage
from flask import Flask, jsonify, request
from flask_restplus import Api, Resource
from train_predict import *


application = app = Flask(__name__)
api = Api(app, version="1.0", title="Face recognition")
ns = api.namespace(
    "ArtificialIntelligence",
    description="Represents the image category by the AI."
)


# Use Flask-RESTPlus argparser to process user-uploaded images
arg_parser = api.parser()
arg_parser.add_argument('file', location='files',
                           type=FileStorage, required=True)


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

MODEL_PATH = 'trained_knn_model.clf'
OUTPUT_PATH = 'knn_examples/output/'
TEST_PATH = "knn_examples/test/"
TRAIN_PATH = 'knn_examples/train/'


# @ns.route('/predict')
# class Predict(Resource):
#     @api.doc(parser=arg_parser)
#     def post(self):
#         # Get the uploaded file
#         file = request.files['file']
#
#         # Save the uploaded file to the test directory
#         file_path = os.path.join(TEST_PATH, file.filename)
#         file.save(file_path)
#
#         # Loop through each person in the train directory
#         for person in os.listdir(TRAIN_PATH):
#             output_dir = os.path.join(OUTPUT_PATH, person)
#             if not os.path.exists(output_dir):
#                 os.makedirs(output_dir)
#
#             # Make predictions for the uploaded file
#             predictions = predict(file_path, model_path=MODEL_PATH, allowed_extensions=ALLOWED_EXTENSIONS)
#
#             # Copy the uploaded file to the appropriate output directory
#             for name, (top, right, bottom, left) in predictions:
#                 if name == person:
#                     shutil.copy2(file_path, os.path.join(output_dir, file.filename))
#
#         return jsonify({'result': 'success'})
#
#
# if __name__ == '__main__':
#     app.run(debug=True, port=5000)
#


@ns.route('/predict')
class Predict(Resource):
    @api.doc(parser=arg_parser)
    def post(self):
        # Get the uploaded files
        files = request.files.getlist('file')

        # Loop through each uploaded file
        for file in files:
            # Save the uploaded file to the test directory
            file_path = os.path.join(TEST_PATH, file.filename)
            file.save(file_path)

            # Loop through each person in the train directory
            for person in os.listdir(TRAIN_PATH):
                output_dir = os.path.join(OUTPUT_PATH, person)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Make predictions for the uploaded file
                predictions = predict(file_path, model_path=MODEL_PATH, allowed_extensions=ALLOWED_EXTENSIONS)

                # Copy the uploaded file to the appropriate output directory
                for name, (top, right, bottom, left) in predictions:
                    if name == person:
                        shutil.copy2(file_path, os.path.join(output_dir, file.filename))

        return jsonify({'result': 'success'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)