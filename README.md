# Face-Mask-Detection-with-Pytorch
The spread of the COVID-19 Pandemic Disease has created the world's most significant global health crisis that has had a profound impact on the way we perceive our world and our daily lives. Many countries have created their own rules regarding face masks to control the spread of the virus pandemic, but many people refuse to abide by the government rules. The police are struggling to catch these people and they can't find each and every one of them. Face detection and object detection technology will help detect people not wearing masks and help police check them. Besides mask detection, there are various applications for object and face detection models in different real-time areas such as driverless car, crime detection, license plate detection. In this study, we will be implementing the Mask Detection Method project with Convolutional Neural Networks (CNN), which is frequently used in Deep Learning algorithms.
## Our Goal
Our goal in this project is to accurately identify masked and unmasked human images in the dataset.
## About Dataset
Images used in the dataset include images of masked and unmasked people. The dataset consists of a total of 4606 images belonging to two classes:
2863 masked
1743 unmasked
You can visit https://github.com/cabani/MaskedFace-Net for the dataset
<img width="350" alt="131366700-8b671931-148f-4c2d-bd49-ce4760bf0a82" src="https://user-images.githubusercontent.com/45899874/152232000-89e3ba1c-f7a0-4f13-adff-a0406f6cf7f2.png">
## 1) Import Operations and Loading of Required Libraries
## 2) Loading the Data Set
## 2.1) Determining Image Size and Label Label Values
## 3) Showing Examples from the Data Set
## 4) Dataset Creation and Visualization
## 4.1) Performing the Train - Validation Split Process
## 5) Adjusting Optimization and Evaluation Metrics
## 6) Modelling
Sequential: Sequential model is the neural network construction method that Pytorch has presented to us. Sequential literally means sequential, and thanks to this structure, it can easily create a neural network consisting of sequential layers.

Evrişim Katmanı (Convolution Layer, Conv2D): Bu katmanda giriş görüntüsü üzerinden öznitelik (kenar bulma, köşe bulma, görüntü üzerinde nesne bulma) çıkarımı yapılmaktadır. Bu çıkarım filtreler aracılığı ile gerçekleştirilmektedir.

Convolution Layer (Convolution Layer, Conv2D): In this layer, features (edge finding, corner finding, object finding on the image) are extracted from the input image. This inference is carried out by means of filters.

Conv2D:
in_channels : Number of channels in the input image
out_channels: Number of channels produced by convolution
kernel_size : Size information of the filter to be scrolled
padding: Padding with zeros equally to the left/right or up/down of the input

Activation Layer: Activation functions are needed to introduce nonlinear real-world properties to artificial neural networks. Activation functions such as relu, tanh, sigmoid are widely used. In the last layer, the activation function may vary depending on the problem. If there is a multiple classification problem, then softmax can be used as the activation function in the output layer. If there is a one-class classification problem, then sigmoid can be used as the activation function in the output layer.

ReLU: Rectified linear unit (RELU) is a nonlinear function. The ReLU function takes the value 0 for negative inputs, while x takes the value x for positive inputs.

Adding Pixels (Padding): Since the size of the input matrix and the size of the output matrix are not equal after the convolution process, in order to avoid this problem, pixel padding is performed to eliminate the size difference between the input matrix size and the output matrix size.

Pooling Layer: Basically, the process of reducing the size of the image is carried out without losing its properties. It can also appear as down sampling. There is no learning process in this layer. It is a layer used after the convolution layer.

Maximum Pooling: The output matrix is formed by taking the maximum value of the points that the filter has traveled on the output matrix formed after the convolution process.

MaxPool2d:

kernel_size : Size information of the filter to be scrolled
stride : The stride value is a value that can be changed as a parameter in CNN models. This value determines how many pixels the filter will slide over the main image.
dilation: Controls spacing between core points.

Flattening Layer: Neural networks take input data from a one-dimensional array. The Flattening layer performs the conversion of the matrices from the Convolutional and Pooling layers into a one-dimensional array.

Linear Layer: A linear transformation is applied to the incoming data in this layer.
in_features=80000: Size of each input instance
out_features=1024: Size of each output instance
## 6.1) Making Graphics Card Settings
## 6.2) Transferring the Data Set to the Graphics Card
## 7) Model Success Evaluation
## 7.1) Confusion Matrix
## 8) Testing the Model
## 9) Output
<img width="520" alt="Screen Shot 2022-02-02 at 23 21 12" src="https://user-images.githubusercontent.com/45899874/152232566-b33f67f9-350e-4603-8a91-b5429d584380.png">
