# hackathon

## fisrt run

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
