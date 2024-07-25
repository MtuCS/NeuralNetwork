import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

loaded_model = load_model('my_model.keras')

new_data = pd.read_csv('Silver_data.csv')

predictions = loaded_model.predict(new_data)
print(predictions)
