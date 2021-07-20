import tensorflow as tf
from tensorflow.keras.models import load_model
import utils.gui


model = load_model("src/model/mnist2.h5")
root = utils.gui.create_main_window()
cv = utils.gui.create_canvas(root)
utils.gui.bind_events(cv)
utils.gui.add_buttons_labels(cv)
root.mainloop()