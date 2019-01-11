# Flower Identification Web Application using Flask

### List of contents

- [Introduction](#introduction)
- [Working](#working)
- [Installation](#installation)
- [Running](#running)


## Introduction
---
[(Back to top)](#list-of-contents)

This Web application implements the GUI for flower classification on the dataset from http://www.robots.ox.ac.uk/~vgg/data/flowers/102/ coded in Pytorch. It involves classification of flowers into 102 categories occuring mostly in United Kingdom. It is done as a part of Pytorch Deep Learning scholarship challenge lab project.

### Following is an image collage showing the images present in the datasets.
 
 ![img](https://imgur.com/I3t3kKn.jpg)
 
## Working
---
[(Back to top)](#list-of-contents)

The step-by-step procedure of the Project:

+ Collection of dataset from the link mentioned at the top;
+ Data preprocessing: Augmentation being applied to train set;
+ Training the classifier part of the model Densenet121 pretrained on Imagenet;
+ The model built scores 98.3% on the validation set;
+ Saving the checkpoint containing the models parameteres;
+ Building a Flask Application using the inference from pretrained model;

NOTE : The whole Machine Learning pipeline is implemented in the jupyter notebook provided in the repository.
 

## Installation
---
[(Back to top)](#list-of-contents)

These instructions assume you have `git` installed for working with Github from command window.

1. Clone the repository, and navigate to the downloaded folder. Follow below commands.
```
git clone https://github.com/pswaldia/Flower_identification
cd Flower_identification

```

2. Creating python virtual environment using virtualenv package using following lines of code.

NOTE: For this step make sure you have virtualenv package installed.

```
virtualenv venv
source venv/bin/activate

```

3. Install few required pip packages, which are specified in the requirements.txt file .
```
pip3 install -r requirements.txt

```

## Running
---
[(Back to top)](#list-of-contents)

Run the following code:
```shell
flask run
```
Now copy the URL of the local host that will appear on your terminal and run it in browser.

### Home Page:

![img](https://imgur.com/GFy7ZHX.jpg)

### Prediction Page

![img6](https://imgur.com/96ndwbd.jpg)

 
