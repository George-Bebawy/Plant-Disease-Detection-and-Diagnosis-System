# PLANT DISEASE RECOGNITION SYSTEM

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Dataset](#dataset)
- [Future Work](#future-work)
- [Team Members](#team-members) 
---

## Project Overview
The **Plant Disease Recognition System** is an innovative application designed to assist users in identifying plant diseases through advanced image classification techniques. With the increasing importance of agriculture and gardening in our daily lives, this project aims to empower gardeners, farmers, and plant enthusiasts by providing them with a tool that quickly and accurately diagnoses plant health issues. By leveraging machine learning and computer vision, this system not only saves time but also enhances the understanding of plant care and disease management.

---

## Features
- **Image Upload**: Users can upload images of their plants to receive a diagnosis of potential diseases. This feature is designed to be user-friendly, allowing anyone to easily utilize the application.
- **Disease Classification**: The model employs deep learning techniques to analyze uploaded images and classify them into specific disease categories. This classification helps users understand what issues their plants may be facing.
- **Disease Details**: For each identified disease, the application provides detailed descriptions, including symptoms, causes, and recommended treatments. This feature serves as a valuable resource for users seeking to care for their plants effectively.
- **User-Friendly Interface**: The application is built with an intuitive interface, making it accessible to users of all technical backgrounds. Clear instructions guide users through the image upload and diagnosis process.
- **Continuous Improvement**: The system is designed to be continuously updated with new data, allowing the model to learn from new plant diseases and improve its accuracy over time.

---

## Installation
To set up the **Plant Disease Recognition System** on your local machine, follow these steps:

1. **Clone the Repository**: Start by cloning the repository to your local machine using Git. Open your terminal and execute the following command:

    ```bash
    git clone https://github.com/yourusername/plant-disease-recognition.git
    ```

2. **Navigate to the Project Directory**: Change your current directory to the newly created project folder:

    ```bash
    cd plant-disease-recognition
    ```

3. **Install Required Packages**: The project requires specific Python libraries. To install all necessary dependencies, run:

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Application**: Finally, start the Streamlit application by executing:

    ```bash
    streamlit run app.py
    ```

5. **Access the Application**: Once the application is running, it will provide you with a local URL (typically http://localhost:8501) to access the interface via your web browser.

---

## Usage
1. **Open the Application**: Navigate to the URL provided in the terminal output to open the **Plant Disease Recognition System** in your web browser.
2. **Upload a Photo**: Click the "Upload" button to select an image of the plant you wish to diagnose. Ensure the image is clear and focused for optimal results.
3. **View Results**: After the image is processed, the system will display the predicted disease, along with a confidence score indicating the certainty of the diagnosis.
4. **Access Disease Details**: Below the classification results, detailed information about the identified disease will be displayed, including symptoms, causes, and recommended treatments. This information is designed to assist users in taking appropriate action to care for their plants.

---

## Model Architecture
The **Plant Disease Recognition System** utilizes a **Convolutional Neural Network (CNN)**, which is a class of deep neural networks particularly effective for image classification tasks. The architecture includes several layers designed to extract features from images and make predictions:

- **Input Layer**: Accepts the uploaded images in a specified format.
- **Convolutional Layers**: These layers apply various filters to the input images to detect patterns and features, such as edges and textures.
- **Pooling Layers**: Responsible for down-sampling the feature maps, reducing dimensionality, and retaining the most critical features while minimizing noise.
- **Fully Connected Layers**: After flattening the pooled features, these layers make the final predictions by interpreting the features extracted from the previous layers.
- **Output Layer**: Outputs the classification results, identifying the most likely disease category for the given input.

The model has been trained on a substantial dataset of labeled plant images, allowing it to generalize well to new, unseen data.

---

## Training
Training the model involved several critical steps to ensure high accuracy and reliability in predictions:

1. **Dataset Preparation**: The dataset was divided into training, validation, and testing sets to evaluate the model's performance during and after training.
2. **Data Augmentation**: Techniques such as rotation, flipping, and scaling were applied to the training images to artificially expand the dataset. This helps the model generalize better by learning from varied representations of the same class.
3. **Hyperparameter Tuning**: Various hyperparameters, such as learning rate, batch size, and the number of epochs, were optimized to achieve the best results. This process often involves experimentation and evaluation of different combinations.
4. **Model Evaluation**: The model's performance was assessed using metrics such as accuracy, precision, recall, and F1-score. Confusion matrices were also generated to visualize performance across different classes.
5. **Continual Learning**: The model is designed to be retrained with new data periodically to enhance its performance and accuracy as more plant disease images become available.

---

## Results
The **Plant Disease Recognition System** achieved a high accuracy rate on the validation dataset, demonstrating its effectiveness in classifying various plant diseases. The results indicate:

- **Overall Accuracy**: [Insert accuracy percentage here]
- **Precision and Recall Metrics**: [Insert precision and recall for each class if applicable]
- **Confusion Matrix**: A confusion matrix is available in the results folder, providing insights into how the model performs across different classes.

These results underscore the model's ability to assist users in accurately identifying plant diseases and inform them about proper care methods.

---

## Dataset
The model was trained using a comprehensive dataset containing images of various plant diseases, sourced from reputable repositories. This dataset includes a diverse range of plant species and associated diseases, ensuring that the model can generalize effectively.

You can access the dataset using the following link:
- [Kaggle Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

---

## Future Work
The **Plant Disease Recognition System** has several avenues for future development, including:

- **Integration of User Feedback**: Implementing a feedback mechanism to allow users to report inaccuracies, thereby improving the model through continual learning.
- **Expansion of Disease Categories**: Including additional plant species and diseases to broaden the scope of the application.
- **Mobile Application**: Developing a mobile version of the application to increase accessibility and convenience for users on the go.
- **Enhanced User Interface**: Continuously improving the user experience by incorporating features such as tutorials, FAQs, and a community forum for plant care discussions.

---

## Team Members
1. Ahmed Hamdy
2. Youssef Ahmed
3. George Bebawy
