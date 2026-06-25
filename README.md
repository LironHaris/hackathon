# hackathon
Liron Haris 322400649
Michal Sacharen 324143668
Noam Lin 322845488
Yotam Kashai 315143297

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


## third run

- accuracy : 60.5 %
- best val loss (epoch ): 1.29
- last val loss: 1.29
- last train loss : 1.58
- epochs : 15

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
- MaxPooling(4)
- Linear(6272, 256)
- ReLU
- Dropout(0.3)
- Linear(256, 20)

augmentations:
- RandomHorizontalFlip(p=0.5)
- RandomRotation(15)
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
- RandomGrayscale(p=0.1)
- GaussianBlur(p=0.2)
- RandomErasing(p=0.2)

hyper parameters:
- learning rate : 0.001
- batch size : 64
- kernel size : 3
- padding : 1
- criterion : CrossEntropy
- optimizer: Adam


## fourth run

- accuracy : 67.25 %
- best val loss (epoch ): 1.05
- last val loss: 1.18
- last train loss : 1.26
- epochs : 15
- resumed from : best_model_20260625_122735.pth

architecture: same as third run

augmentations: same as third run

hyper parameters:
- learning rate : 0.001
- batch size : 64
- kernel size : 3
- padding : 1
- criterion : CrossEntropy
- optimizer: Adam


## fifth run

- accuracy : 73 %
- best val loss (epoch ): 0.88
- last val loss: 0.88
- last train loss : 1.15
- epochs : 15

data split: 56% train / 14% val / 30% test (stratified, test kept as clean held-out set)

architecture: same as third run

augmentations: same as third run

hyper parameters:
- learning rate : 0.001
- batch size : 64
- kernel size : 3
- padding : 1
- criterion : CrossEntropy
- optimizer: AdamW


## sixth run

first part
- accuracy : 69.75 %
- best val loss (epoch ): 1
- last val loss: 1.03
- last train loss : 1.32
- epochs : 15

second part
- accuracy : 75.82 %
- best val loss (epoch ): 0.8
- last val loss: 0.89
- last train loss : 0.97
- epochs : 15

third part - less augentation settings
- accuracy : 79.14 %
- best val loss (epoch ): 0.68
- last val loss: 0.72
- last train loss : 0.8
- epochs : 10

data split: 56% train / 14% val / 30% test (stratified, test kept as clean held-out set)

architecture: 4 residual blocks (each = 2x [Conv -> BatchNorm -> ReLU] + skip connection)
- ResidualBlock(3, 16)
- MaxPooling(2)
- ResidualBlock(16, 32)
- MaxPooling(2)
- ResidualBlock(32, 64)
- MaxPooling(2)
- ResidualBlock(64, 128)
- MaxPooling(4)
- Linear(6272, 256)
- ReLU
- Dropout(0.3)
- Linear(256, 20)

note: skip connection uses element-wise add; 1x1 conv projection on the skip path when channel count changes

augmentations: same as third run

hyper parameters:
- learning rate : 0.001
- batch size : 64
- kernel size : 3
- padding : 1
- dropout : 0.3
- criterion : CrossEntropy
- optimizer: AdamW (weight decay 1e-2)


## seventh run
first part
- accuracy : 69.75 %
- best val loss (epoch ): 1
- last val loss: 1.03
- last train loss : 1.32
- epochs : 15

second part
- accuracy : 75.82 %
- best val loss (epoch ): 0.8
- last val loss: 0.89
- last train loss : 0.97
- epochs : 15

second part
- accuracy : 75.82 %
- best val loss (epoch ): 0.8
- last val loss: 0.89
- last train loss : 0.97
- epochs : 5

## Final Architecture
all kernels- 5X5 , stride-1/2 in last layer, padding- 2, dropout-0.3 for MLP head,hidden size- 512  
Layer 1->4:
- Con2d 3->16->32->64->128
- BatchNorm
- Relu
- MaxPool(2)
layer 5:
- Conv2d 128->128 stride-2
- BatchNorm
- Relu

MLP Head:
- flatten
- linear 6272->512
- Relu
- linear 512->20

## runs
