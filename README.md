# hackathon

## first run

- accuracy : 30 %
- best val loss (epoch 3): 2.25
- last val loss: 4.9
- last train loss : 0.18
- epochs : 7

architecture:
- Conv(3,16)
- RelU
- MaxPooling(2)
- Conv(3,16)
- RelU
- MaxPooling(2)
- Linear(128)
- Relu
- Linear(128)

hyper parameters:
- learning rate : 0.001
- batch size : 32
- kernel size : 5
- padding : 1
- criterion : CrossEntropy
- optimizer: Adam


## second run

- accuracy : 62 %
- best val loss (epoch 3): 1.25
- last val loss:   1.3
- last train loss : 0.18
- epochs : 10

architecture:
- Conv(3, 16)
- BatchNorm(16)
- ReLU
- MaxPooling(2)
- Conv(16, 32)
- BatchNorm(32)
- ReLU
- MaxPooling(2)
- Conv(32, 64)
- BatchNorm(64)
- ReLU
- MaxPooling(2)
- Conv(64, 128)
- BatchNorm(128)
- ReLU
- MaxPooling(2)
- AdaptiveAvgPool(7, 7)
- Linear(6272, 256)
- ReLU
- Dropout(0.3)
- Linear(256, 20)

hyper parameters:
- learning rate : 0.001
- batch size : 64
- kernel size : 5
- padding : 1
- criterion : CrossEntropy
- optimizer: Adam