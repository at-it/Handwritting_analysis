# Handwritting_analysis
Simple ML implementation based on MNIST dataset with GUI using TKinter and OpenCV libs.

<b>Current functionality:</b>
- Recognition of numerous digits drawn by the user using GUI at the same time 

<b>Known issues:</b>
- '9' recognition accuracy to be checked
- If the number is too big in drawn size, reshaping to 28x28 pixels loses the shape and therefore recognition accuracy
- Croping window size to save the image and use it for recognition is not always appriopriate (losing information in the corners)
