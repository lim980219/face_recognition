Face Recognition API
This is a simple Flask-based RESTful API for face recognition. The API uses a trained KNN model to recognize faces in uploaded images. The recognized faces are then categorized and saved in corresponding directories.

Getting Started
To get started with the API, you'll need to have Python and Flask installed on your system.

Clone this repository: git clone https://github.com/your_username/face-recognition-api.git
Install the required packages: pip install -r requirements.txt
Run the API: python app.py
The API should now be running on http://localhost:5000. You can use a tool like Postman to test the API.

API Endpoints
The API has one endpoint for uploading and processing images:

/ArtificialIntelligence/predict
Method: POST

Request Parameters:

file: The file to be uploaded. Only PNG, JPG, and JPEG formats are allowed.
Response:

result: A string indicating whether the request was successful.
