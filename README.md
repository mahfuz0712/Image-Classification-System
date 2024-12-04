# Image Classification System
=====================Project Directory Struncture=====================
#### Step-1: Organize your project directory like this:
```
image_classification_system/
│
├── datasets/
│   ├── test_data/
│   │   ├── sick/
│   │   │   ├── sick_child1.jpg
│   │   │   ├── sick_child2.jpg
│   │   │   └── ...
│   │   └── healthy/
│   │       ├── healthy_child1.jpg
│   │       ├── healthy_child2.jpg
│   │       └── ...
│   ├── augmented_data/
│   │   ├── sick/
│   │   │   ├── sick_child1.jpg
│   │   │   ├── sick_child2.jpg
│   │   │   └── ...
│   │   └── healthy/
│   │       ├── healthy_child1.jpg
│   │       ├── healthy_child2.jpg
│   │       └── ...
│   ├── validation/
│       ├── sick/
│       │   ├── sick_child1.jpg
│       │   ├── sick_child2.jpg
│       │   └── ...
│       |── healthy/
|       |   ├── healthy_child1.jpg
│       |   ├── healthy_child2.jpg
│       |   └── ...
│
├── models/
│   └── best.model.h5 // this will be generated after you run train.py
│
├── scripts/
│   ├── augment.py
│   ├── train.py
│   ├── evaluate.py
│   └── xai.py
│
├── requirements.txt
├── README.md
└── .gitignore
```
#### Step-2: Set Up Your Virtual Environment
##### 1.  Create a virtual environment (optional but recommended):
```
python -m venv venv
```
###### Activate the environment:

* Windows: .\venv\Scripts\activate
* Mac/Linux: source venv/bin/activate

##### 2. Install the required packages:
```
pip install -r requirements.txt
```

#### Step-3: augment the test_data to create augmented_data
##### 1. Run the augment.py script to create augmented_data
```
cd scripts
python augment.py
```
This script will train the model using the training data and save the model to the models directory.

#### Step-4: train the Model
##### 1. Run the train.py script to evaluate the model:
```
python train.py
```
This script will train the model using the augmented_data and validation data and generate  the best.mode.h5 with accuracy of 90% and above 

#### Step-5: Generate Classification Report
##### 1. Run the evaluate.py script to see the report
```
python evaluate.py
```
This script will load the trained model and make classification repornt on the on the images present at the validation path.

#### Step-6" AI Explainabilty
##### 1. Run the xai.py script to see the report

```
python xai.py
```
This script will load the trained model and make AI Explainabilty report on the on the images present
## Author
[Mohammad Mahfuz Rahman](https://github.com/mahfuz0712)

## License

This project is not open-source and is only available under a paid License.

---

**Happy Coding!**