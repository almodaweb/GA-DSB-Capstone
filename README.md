# DermAI 

DermAI is an end-to-end AI-powered dermatology assistant that combines deep learning for image classification with large language models for explainability, enabling users to upload skin images, receive condition predictions, and access structured, clinically-informed insights in real time.

## Problem Statement & Motivation

Skin conditions are often difficult for individuals without medical training to accurately identify. Many people delay seeking care due to limited access to dermatologists, low awareness of early warning signs, and the fact that symptoms can present differently across skin tones, age groups, and anatomical regions. These challenges contribute to delayed diagnosis, misinterpretation of symptoms, and preventable worsening of conditions.

This project aims to address this gap by developing an AI-powered system that can classify skin conditions from images and provide structured, user-friendly explanations. By combining computer vision with natural language generation, DermAI seeks to improve early awareness, support self-monitoring, and demonstrate how scalable artificial intelligence solutions can enhance accessibility and education in healthcare settings.

## The Dataset

The model is trained using the [SCIN (Skin Condition Image Network) Dataset](https://console.cloud.google.com/storage/browser/dx-scin-public-data), a publicly available dermatology dataset hosted on Google Cloud. The SCIN dataset contains 5,000+ volunteer contributions (10,000+ images) of common dermatology conditions. Contributions include Images, self-reported demographic, history, and symptom information, and self-reported Fitzpatrick skin type (sFST). In addition, dermatologist labels of the skin condition and estimated Fitzpatrick skin type (eFST) and layperson estimated Monk Skin tone (eMST) labels are provided for each contribution.

## Model Architecture & Training Pipeline

The classification model used in this project is on a pretrained DenseNet model fine-tuned on the SCIN dataset. This architecture is particularly well-suited for medical imaging tasks, where subtle visual patterns are important for distinguishing between conditions. This approach allows the model to capture both low-level visual features (e.g., texture, color) and high-level patterns (e.g., lesion structure), which are critical for dermatological image classification. The final classification head was customized to match the number of skin condition classes, and dropout was applied to reduce overfitting.

Training was performed using:
- Input: 224 × 224 RGB images  
- DenseNet121 (pretrained)
- Global Average Pooling  
- Batch Normalization  
- Dense layer (128 units, ReLU activation)  
- Dropout (0.5) for regularization  
- Output layer with softmax activation (multi-class classification) 

During initial training, most layers were frozen, with only the last ~30 layers fine-tuned to adapt higher-level features to the dataset. The model was trained using a two-stage fine-tuning approach and  learning rate scheduler was used to dynamically reduce the learning rate when validation loss plateaued, improving convergence and stability.

## Evaluation of Results & Baseline Comparison

The model achieved an overall accuracy of 41% on the test set, outperforming a random baseline (~16–17% for six classes). Performance varied across conditions, with stronger results for more common and visually consistent classes like Eczema, and weaker performance for conditions that are less represented or visually similar to others.

These results show that the model is able to learn meaningful patterns from the data, but performance is still limited by class imbalance, variability in image quality, and differences in how conditions appear across individuals. While not suitable for clinical use, the model demonstrates the potential of AI for supporting early awareness and education in dermatology.

## How to Run the Project

1. Clone the Repository
2. Install Dependencies
```
pip install -r requirements.txt
```
4. Authenticate (for SCIN Dataset Access)
```
from google.colab import auth
auth.authenticate_user()
```
4.  Run Data Pipeline & Training
5.  Launch the Streamlit App
```
streamlit run app.py
```
7.  Use the Application (Upload a skin image, run the scan, view predicted condition and explanation)

## File Directory
```
├── README.md
├── reqirements.txt

├── Code
│   ├── SCIN_model.ipynb
│   └── app.py

├── Figures
│   ├── class_distribution.png
│   └── confusion_matrix.png

├── Presentation
│   └── GA DS Capstone.pdf

├── Model
│   ├── SCIN_labels_DenseNet.json
│   └── SCIN_model_DenseNet.keras
```

## Limitations & Future Work

This project is limited by class imbalance and high variability in the dataset, including differences in image quality, lighting, and framing. Some skin conditions are underrepresented, which impacts model performance and leads to lower accuracy for certain classes. Additionally, multiple images per case required careful handling to avoid data leakage. Hardware constraints also limited the ability to train on larger image sizes or more complex models. The system is not clinically validated and is intended for educational use only.

Future work will focus on improving model performance through advanced transfer learning techniques, better class balancing strategies, and expanding the dataset to include greater diversity. This includes integrating more subclasses by factors such as age and skin tone to ensure each condition is adequately represented. Additional improvements may include incorporating longitudinal tracking of skin conditions, enhancing the user interface, and validating the model with clinical experts to support real-world applications.

## Project Demo

A short walkthrough of DermAI, including the problem, model, and live application demo can be found [here](https://youtu.be/UZiGqZhheig).

The application can be accessed [here](https://ga-dsb-capstone.streamlit.app). 

## License & acknowledgments

This project is for educational and research purposes only. Not intended for commercial use or clinical diagnosis.

This project was completed as part of the Data Science Bootcamp Capstone Project requirement at General Assembly.

Special thanks to:
- General Assembly for providing the structure, mentorship, and learning experience throughout the Data Science Bootcamp
- Google Research for providing access to the SCIN dataset
