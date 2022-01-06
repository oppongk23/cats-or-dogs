# cats-or-dogs
Is it a cat or a dog?

This project uses a simple CNN on an image datasets to classify an image as one of a cat or a dog.

The notebook "cats_or_dogs.ipynb" outlines the entire training and validation process, and the prediction process.

The model is saved and deployed on a local server using Flask. The model state file, has been excluded from the repo due to its large size. 

The flask folder contains the "app.py" file which shows the steps followed in deploying the model.

The two test images which were used to test the deployed model were downloaded from:

"https://getwallpapers.com/collection/cute-cat-wallpapers"

"https://www.hdnicewallpapers.com/Wallpaper-Download/Dog/Dog-Running-on-Grass-Image"

Libraries: Pytorch, Flask, Matplotlib
