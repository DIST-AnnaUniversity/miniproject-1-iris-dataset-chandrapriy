# miniproject-1-iris-dataset-chandrapriy
miniproject-1-iris-dataset-chandrapriy created by GitHub Classroom
                    
                                                   MULTILAYER PERCEPTRON NEURAL NETWORK

*This uses multilayer perceptrons (Neural Network) to predict the species of the Iris dataset.

*Neural network is a machine learning algorithm which is inspired by a neuron.

*A neuron consists of a dendrite and an axon which are responsible for collecting and sending signals. 

*For this artificial neural network, the concept works similar in which a lot of neurons are connected to each layer with its own corresponding weight and biases.
Although there are currently architecture of neural network, multilayer perceptron is being used as the architecture to prevent the process from ,overfitting(training accuracy=good but test accuracy=bad) to the Iris Species due to less feature.


1)Visualisation of the dataset:
   The coding below shows the visualisation of the dataset in order to understand the data more.It can be seen that every species of the Iris can be segregated into different regions to be predicted.

2)Coding is then converted to the species into each respective category to be feed into the neural network

3)Converting data to numpy array in order for processing is done 

4)Normalization:
   The feature of the first dataset has 6cm in Sepal Length, 3.4cm in Sepal Width, 4.5cm in Petal Length and 1.6cm in Petal Width. 
  
   However, the range of the dataset may be different. Therefore, in order to maintain a good accuracy, the feature of each dataset must be normalized to a range 
   of 0-1 for processing.

5)Creating train,test and validation data

6)Change the label to one hot vector.

7)Finally,
   An accuracy of 100% is achieved in this dataset.It can be asserted that for each epoch, the neural network is trying to learn from its existing feature and predict it by its weights and biases.For each epoch, the weights and biases and changed by subtracting its rate to get a better accuracy each time.
