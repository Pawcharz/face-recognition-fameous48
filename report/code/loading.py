CLASSES = 48
IMAGE_SIZE = 24

def read_file(filename):

  with open(filename, 'r') as file:
    file.readline() # we skip the first line as it is not needed
    number_of_pixels = int(file.readline())

    features = []
    labels = []

    for line in file.readlines():
      elements = line.split()

      # add features
      pixels = np.array(elements[:number_of_pixels], dtype=float)
      pixels = np.reshape(pixels, [IMAGE_SIZE, IMAGE_SIZE])
      features.append(pixels)

      # add labels
      labels.append(elements[number_of_pixels+2])

    features = np.array(features)
    labels = np.array(labels, dtype=int)

  return features, labels


X_0, y_0 = read_file('./data/x24x24.txt')
X_1, y_1 = read_file('./data/y24x24.txt')
X_2, y_2 = read_file('./data/z24x24.txt')

X = np.concatenate((X_0, X_1, X_2))
y = np.concatenate((y_0, y_1, y_2))

X_train, X_test, y_train_raw, y_test_raw = train_test_split(X, y,
                                                train_size=N_TRAIN_EXAMPLES,
                                                test_size=N_TEST_EXAMPLES,
                                                random_state=42)

y_train = np.zeros((y_train_raw.shape[0], CLASSES))
y_test = np.zeros((y_test_raw.shape[0], CLASSES))

for i, value in enumerate(y_train_raw):
  y_train[i][value] = 1

for i, value in enumerate(y_test_raw):
  y_test[i][value] = 1
