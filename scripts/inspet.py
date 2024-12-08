from tensorflow.keras.models import load_model # type: ignore

# Load the trained model
model = load_model('../models/best.model.h5')

# Print the model summary
model.summary()
