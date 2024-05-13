from sklearn.model_selection import train_test_split
import re
import numpy as np

from utils.constrains import CLASSES, IMAGE_SIZE

def load_dataset(directory_path):

  def read_file(filename):

    with open(filename, 'r') as file:
      lines = file.readlines()

      # Remove newLines
      for i, line in enumerate(lines):
        lines[i] = line.replace('\n', '')

      # We assume these are integers
      EXAMPLES_NR = int(lines[0])
      PIXELS_NR = int(lines[1])

      inputs = list()
      labels = np.zeros(EXAMPLES_NR, dtype=int)

      examples_raw = lines[2:EXAMPLES_NR+2]

      for i, example_raw in enumerate(examples_raw):
        # Split by spaces (treats multiple as one)
        tokens = re.split('\s+', example_raw)

        pixel_values = np.array(tokens[0:PIXELS_NR])
        attributes = tokens[PIXELS_NR:]

        pixel_values = np.array(pixel_values, dtype=float)
        pixel_values = pixel_values.reshape([IMAGE_SIZE, IMAGE_SIZE])

        inputs.append(pixel_values)
        labels[i] = int(attributes[2])

      inputs = np.array(inputs)
    return inputs, labels

  # classes 0-15
  X_0, y_0 = read_file(f'{directory_path}/x24x24.txt')
  # classes 16-31
  X_1, y_1 = read_file(f'{directory_path}/y24x24.txt')
  # # classes 32-48
  X_2, y_2 = read_file(f'{directory_path}/z24x24.txt')

  # Concatenate train and test images
  X = np.concatenate((X_0, X_1, X_2))
  y = np.concatenate((y_0, y_1, y_2))

  N_TRAIN_EXAMPLES=int(len(X) * 0.8)
  N_TEST_EXAMPLES=len(X) - N_TRAIN_EXAMPLES


  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=N_TRAIN_EXAMPLES, test_size=N_TEST_EXAMPLES, random_state=1)

  # One-Hot encoding
  # Getting dummy variables
  y_train_fixed = np.zeros((y_train.shape[0], CLASSES))
  y_test_fixed = np.zeros((y_test.shape[0], CLASSES))

  for i, value in enumerate(y_train):
    y_train_fixed[i][value] = 1

  for i, value in enumerate(y_test):
    y_test_fixed[i][value] = 1

  return X_train, X_test, y_train_fixed, y_test_fixed