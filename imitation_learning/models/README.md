# aac.py

### Conv layers:

3 -> 16 -> 32 -> 64 -> 128 -> 256 -> 512

### Fully-connected layers:

2048 -> 256 -> BatchNorm -> 128 -> 32

# aac_lstm.py

### Conv layers:

3 -> 16 -> 32 -> 64 -> 128 -> 256 -> 512

### LSTM:

1 layer, 512 units

### Fully-connected layers:

512 -> 128 -> 32

# basic_models.py

## Model 1 (AFC):

### Conv layers:

AlexNet feature extractor

### Fully-connected layers:

BatchNorm (size: 4096) -> 2048 -> 512 -> 32 -> softmax

## Model 2 (ALSTM):

### Conv layers:

AlexNet feature extractor

### BatchNorm Layer:

size: 4096

### LSTM:

1 layer, 4096 input features, 512 hidden units

### Fully-connected layers:

512 -> 32 -> softmax

# models_2rooms.py

## ALSTM2rooms

### Conv layers:

AlexNet feature extractor + Conv2d(256 -> 512)

### BatchNorm Layer:

size: 2048

### LSTM:

2 layers
2048 input features, 1024 hidden units
1024 input features, 512 hidden units

### Fully-connected layers:

512 -> 128 -> 32 -> softmax
