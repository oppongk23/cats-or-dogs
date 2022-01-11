from flask import Flask, jsonify, request, render_template, url_for
from flask_ngrok import run_with_ngrok
import torch
from torch._C import device
import torchvision.transforms as transforms
from PIL import Image
from model import CNN
import os

app = Flask(__name__)

def transform_image(image):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    return my_transforms(image).unsqueeze(0)


labelDict = {0:"Cat", 1: "Dog"}
device = torch.device('cpu')
#model = torch.load("dogs_or_cats.pth", map_location=device)

model = CNN(num_classes=2)
model.load_state_dict(torch.load("dogs_or_cats_state.pt", map_location=device))
model.eval()

images_folder = os.path.join('static', 'images')
app.config["UPLOAD FOLDER"] = images_folder

def get_prediction(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform_image(image)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    prediction = y_hat.item()
    return prediction

def render_prediction(prediction):
    return labelDict[prediction]

image_folder = os.path.join('static', 'images')
app.config["UPLOAD_FOLDER"] = image_folder

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
            file = request.files['imagefile']
            if file is not None:
                class_ID = get_prediction(file)
                class_name = render_prediction(class_ID)

                pic = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

                return render_template('prediction_index.html', user_image=pic, prediction_text=' We have a {} '.format(class_name))
            


if __name__ == "__main__":
    app.run()



##https://getwallpapers.com/collection/cute-cat-wallpapers
##https://www.hdnicewallpapers.com/Wallpaper-Download/Dog/Dog-Running-on-Grass-Image

##commands to remember: 
# ps fA | grep python
## run the server: FLASK_APP=app.py flask run

## send the request : curl -X POST -H "Content-Type: multipart/form-data" http://localhost:5000/predict -F "file=@random_dog.jpg"