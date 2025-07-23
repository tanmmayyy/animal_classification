# 🐾 Animal Classification AI Project

A comprehensive machine learning project that classifies 15 different animal species using deep learning techniques. The project includes both CNN from scratch and Transfer Learning approaches, along with a modern web application for real-time predictions.

## 📊 Project Overview

This project implements an image classification system capable of identifying 15 different animal species:
- 🐻 Bear
- 🐦 Bird  
- 🐱 Cat
- 🐄 Cow
- 🦌 Deer
- 🐕 Dog
- 🐬 Dolphin
- 🐘 Elephant
- 🦒 Giraffe
- 🐴 Horse
- 🦘 Kangaroo
- 🦁 Lion
- 🐼 Panda
- 🐯 Tiger
- 🦓 Zebra

## 📈 Dataset Statistics

- **Total Images**: 1,944
- **Number of Classes**: 15
- **Average Images per Class**: 129
- **Image Dimensions**: 224 x 224 x 3
- **Training Split**: 80% (1,561 images)
- **Validation Split**: 20% (383 images)

### Class Distribution:
| Animal | Image Count |
|--------|-------------|
| Bear | 125 |
| Bird | 137 |
| Cat | 123 |
| Cow | 131 |
| Deer | 127 |
| Dog | 122 |
| Dolphin | 129 |
| Elephant | 133 |
| Giraffe | 129 |
| Horse | 130 |
| Kangaroo | 126 |
| Lion | 131 |
| Panda | 135 |
| Tiger | 129 |
| Zebra | 137 |

## 🏗️ Model Architecture

### 1. CNN from Scratch
- **Architecture**: 4 Convolutional blocks + 2 Dense layers
- **Parameters**: 19,270,991 (73.51 MB)
- **Validation Accuracy**: 47.78%
- **Training Time**: ~20 minutes (20 epochs)

```
Layer Structure:
├── Conv2D (32 filters, 3x3) + MaxPooling2D
├── Conv2D (64 filters, 3x3) + MaxPooling2D  
├── Conv2D (128 filters, 3x3) + MaxPooling2D
├── Conv2D (256 filters, 3x3) + MaxPooling2D
├── Flatten + Dropout(0.5)
├── Dense(512) + Dropout(0.3)
└── Dense(15, softmax)
```

### 2. Transfer Learning (VGG16)
- **Base Model**: VGG16 pre-trained on ImageNet
- **Total Parameters**: 14,880,847
- **Trainable Parameters**: 166,159
- **Validation Accuracy**: 38.64%
- **Training Time**: ~60 minutes (14 epochs)

```
Architecture:
├── VGG16 Base (frozen)
├── GlobalAveragePooling2D
├── Dense(256, relu) + Dropout(0.5)
├── Dense(128, relu) + Dropout(0.3)
└── Dense(15, softmax)
```

## 🎯 Model Performance

### Best Model: CNN from Scratch (47.78% accuracy)

#### Per-Class Performance:
| Animal | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| Dolphin | 0.83 | 1.00 | 0.91 |
| Zebra | 0.87 | 0.96 | 0.91 |
| Panda | 0.68 | 0.96 | 0.80 |
| Giraffe | 0.78 | 0.56 | 0.65 |
| Lion | 0.53 | 0.62 | 0.57 |
| Cat | 0.48 | 0.58 | 0.53 |
| Bird | 0.57 | 0.44 | 0.50 |
| Tiger | 0.50 | 0.36 | 0.42 |
| Dog | 0.37 | 0.46 | 0.41 |
| Bear | 0.32 | 0.24 | 0.27 |
| Kangaroo | 0.29 | 0.24 | 0.26 |
| Elephant | 0.29 | 0.19 | 0.23 |
| Horse | 0.22 | 0.19 | 0.20 |
| Cow | 0.16 | 0.23 | 0.19 |
| Deer | 0.09 | 0.08 | 0.09 |

**Overall Metrics:**
- **Accuracy**: 47.78%
- **Macro Average**: Precision: 0.47, Recall: 0.47, F1: 0.46
- **Weighted Average**: Precision: 0.47, Recall: 0.48, F1: 0.47

## 🚀 Quick Start

### Prerequisites
```bash
pip install tensorflow matplotlib seaborn scikit-learn pandas numpy flask flask-cors pillow
```

### 1. Training the Model
```bash
# Clone or download the project
# Update the dataset path in the code
python animal_classifier.py
```

### 2. Running the Web Application
```bash
# Start the Flask backend
python app.py

# Open browser and go to:
# http://localhost:5000
```

## 📁 Project Structure
```
animal_classification/
├── animal_classifier.py         # Main ML training script
├── app.py                      # Flask backend API
├── templates/
│   └── index.html             # Web application frontend
├── dataset/                   # Your animal images dataset
│   ├── Bear/
│   ├── Bird/
│   ├── Cat/
│   └── ... (15 animal folders)
├── uploads/                   # Temporary upload folder
├── best_animal_classifier.h5  # Trained model (generated)
└── README.md                  # This file
```

## 🌐 Web Application Features

### Frontend
- **Modern UI/UX**: Responsive design with drag & drop functionality
- **Real-time Preview**: Image preview before prediction
- **Progress Indicators**: Loading animations during processing
- **Result Visualization**: Top 3 predictions with confidence bars
- **Error Handling**: User-friendly error messages
- **Mobile Support**: Fully responsive design

### Backend API
- **REST API**: Multiple endpoints for different use cases
- **File Validation**: Type and size validation
- **Image Processing**: Automatic preprocessing pipeline
- **Model Integration**: Seamless model inference
- **Error Handling**: Comprehensive error responses

### API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Main web application |
| GET | `/health` | Backend health check |
| POST | `/predict` | File upload prediction |
| POST | `/predict_base64` | Base64 image prediction |

## 🔧 Usage Examples

### Web Interface
1. Open `http://localhost:5000`
2. Upload an animal image (drag & drop or click)
3. Click "Classify Animal"
4. View results with confidence scores

### API Usage
```python
import requests

# Upload file
with open('animal_image.jpg', 'rb') as f:
    response = requests.post('http://localhost:5000/predict', 
                           files={'file': f})
    result = response.json()
    print(f"Predicted: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2f}")
```

## 📊 Training Insights

### Data Augmentation Techniques
- **Rotation**: ±20 degrees
- **Width/Height Shift**: ±20%
- **Horizontal Flip**: Random
- **Zoom**: ±20%
- **Shear**: ±20%

### Training Configuration
- **Optimizer**: Adam
- **Learning Rate**: 0.001 (CNN), 0.0001 (Transfer Learning)
- **Batch Size**: 32
- **Loss Function**: Categorical Crossentropy
- **Callbacks**: EarlyStopping, ReduceLROnPlateau

### Performance Analysis
- **Best Performing Classes**: Dolphin, Zebra, Panda (>80% F1-score)
- **Challenging Classes**: Deer, Cow, Horse (<25% F1-score)
- **Reason for CNN Superiority**: Better feature learning for this specific dataset

## ⚡ Future Improvements

### Model Enhancements
- [ ] **Data Collection**: Increase dataset size (aim for 500+ images per class)
- [ ] **Advanced Architectures**: Try ResNet, EfficientNet, Vision Transformers
- [ ] **Ensemble Methods**: Combine multiple models for better accuracy
- [ ] **Fine-tuning**: Unfreeze more layers in transfer learning
- [ ] **Data Balancing**: Address class imbalance issues

### Application Features
- [ ] **Batch Processing**: Multiple image classification
- [ ] **Model Comparison**: Side-by-side model performance
- [ ] **Confidence Threshold**: Adjustable prediction confidence
- [ ] **Export Results**: Download prediction reports
- [ ] **Model Statistics**: Real-time performance metrics

### Technical Improvements
- [ ] **Model Optimization**: TensorFlow Lite conversion
- [ ] **Caching**: Redis for faster predictions
- [ ] **Authentication**: User management system
- [ ] **Database**: Store prediction history
- [ ] **Deployment**: Docker containerization

## 🐛 Known Issues & Solutions

### Common Problems

1. **Low Accuracy on Some Classes**
   - **Issue**: Deer, Cow, Horse have low F1-scores
   - **Solution**: Collect more diverse training data for these classes

2. **Transfer Learning Underperforming**
   - **Issue**: VGG16 achieved lower accuracy than CNN
   - **Solution**: Try different pre-trained models or fine-tuning strategies

3. **Model File Size**
   - **Issue**: 73MB model file
   - **Solution**: Use model quantization or pruning techniques

### Troubleshooting

**Backend Not Starting:**
```bash
# Check if all dependencies are installed
pip install -r requirements.txt

# Verify model file exists
ls -la best_animal_classifier.h5
```

**Prediction Errors:**
- Ensure image is in supported format (JPG, PNG, GIF, BMP)
- Check image file size (max 16MB)
- Verify model file is not corrupted

## 👥 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Dataset**: Animal image classification dataset
- **Frameworks**: TensorFlow/Keras for deep learning
- **UI Inspiration**: Modern web design principles
- **Pre-trained Models**: ImageNet pre-trained weights

## 📞 Contact

For questions, suggestions, or collaborations:
- **Project Repository**: github.com/tanmmayyy
- **Email**: jaintanmay543@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/tanmay-jain-a706a428a/

---

**Made with ❤️ for animal lovers and AI enthusiasts!**
