# Classifier
classifiers = ['SVM']

# Features settings
image_size = (64, 128)  # (width, height)
hog_descriptor = True
color_histogram = False

# HOG features
orientations = 8
pixels_per_cell = 8
cells_per_block = 2
all_channels = False
transform_sqrt = False
feature_vector = True

# SVM settings
C = 5000.0
decision_function_shape = 'ovo'
kernel = 'rbf'

# Sliding window settings
window_size = (128, 64)  # (height, width)
step = (50, 25)  # (y, x)
