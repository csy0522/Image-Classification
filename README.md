# Image Classification

This project recognizes and identifies a cloth image from ten different categories.

## Getting Started

Use any python IDE to open the project. I personally use Spyder from Anaconda.You can download both Anaconda or Spyder from the following links:
* [Anaconda](https://www.anaconda.com/distribution/) - The Data Science Platform for Python/R
* [Spyder](https://www.spyder-ide.org/) - The Scientific Python Development Environment

### Installation

Before running the program, type the following command to install the libraries that the project depends on

```
pip install numpy, matplotlib, tensorflow
```
Or simply type the following:

```
pip install -r requirements.txt
```

## Running the tests

The description of each function is located on top of them. Please read them before running for clarity.<br/>
This project classifies images using three different models, which are:<br/>
1. Single Hidden Layer Neural Network<br/>
2. Multi Hidden Layer Neural Network<br/>
3. Convolutional Neural Network<br/>

The data consists of **70,000** images of clothes and accessories; **48,000** are used for the training, **12,000** are used for validation, and the remaining **10,000** are used for testing.
Ths following is the output from the Multi Hidden Layer Neural Network with random hyperparameters indicated on top of the graph: 

![Multi Hidden Layer Neural Network](/data/MLANN.png)

The graph represents the training progress; the x axis is the number of epochs, and the y axis is the accuracy of prediction.<br/>

For more details, go to **main.ipynb** and test run the all three models using various hyperparameters.

## Deployment

Instead of fashion dataset from mnist, you can also use different dataset from online to test the usability of the models.<br/>
Download any image dataset for classification from online (Ex: Kaggle) and insert the data to the model in order to test its accuracy.
* [Kaggle](https://www.kaggle.com/) - The Machine Learning and Data Science Community

## Built With

* [Python](https://www.python.org/) - The Programming Language
* [Tensorflow](https://www.tensorflow.org/) - The end-to-end open source machine learning platform

## Author

* **CSY** - [csy0522](https://github.com/csy0522)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
