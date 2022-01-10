# cats-or-dogs 
Is it a cat or a dog?

This project uses a simple CNN on an image datasets to classify an image as one of a cat or a dog. The project is thereafter **deployed using Flask**.

The notebook "cats_or_dogs.ipynb" outlines the entire training and validation process, and the prediction process.

The model is saved and deployed on a local server using Flask. The model state file, has been excluded from the repo due to its large size. 

The flask folder contains the "app.py" file which shows the steps followed in deploying the model. 

More information can be found in the [Pytorch documentation](https://pytorch.org/tutorials/recipes/deployment_with_flask.html) 

The two test images which were used to test the deployed model were downloaded from:

"https://getwallpapers.com/collection/cute-cat-wallpapers"

"https://www.hdnicewallpapers.com/Wallpaper-Download/Dog/Dog-Running-on-Grass-Image"

I played around with a webpage template I found from [bryantsmith.com](http://www.bryantsmith.com) for the front-end interface.

***Libraries: Pytorch, Flask, Matplotlib***
