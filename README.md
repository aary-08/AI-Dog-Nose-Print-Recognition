# Dog Nose Print Matching Project

This project demonstrates a machine learning pipeline to match a dog's nose print image with pre-stored nose print features. It uses a pre-trained ResNet50 model for feature extraction, enhances image quality, and identifies matching dogs based on cosine similarity.

# Features
-Nose Print Feature Extraction: Uses ResNet50 to extract features from uploaded nose print images.
- Image Enhancement: Improves image quality using sharpening techniques.
- Similarity Matching: Finds the best match from stored dog nose print features using cosine similarity.
- Automated Quality Check: Ensures uploaded images meet quality standards before processing.

---

## Setup Instructions

### Prerequisites
- Python 3.6+
- Libraries: TensorFlow, OpenCV, scikit-learn, pandas
- Google Colab (Optional but recommended for execution)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/dog-nose-print-matching.git
   cd dog-nose-print-matching
   ```

2. Install required libraries:
   ```bash
   pip install tensorflow opencv-python scikit-learn pandas
   ```

3. Mount Google Drive in Colab (if using Google Colab):
   -Folders are already given .
   -You can upload them in your drive and then you can mount the drive.
   -Link to download the train folder :
   - https://drive.google.com/drive/folders/1Sw0vNWYI84LTsZG22MdG2_CuyS7ZHwd6?usp=sharing
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

---

## Project Workflow

### 1. Load Training Data
- The CSV file (`train_data.csv`) contains columns for dog IDs and corresponding nose print image filenames.
- Images are stored in a specified folder, and their paths are generated dynamically.

```python
train_data = pd.read_csv("/content/drive/MyDrive/DOG/train_data.csv")
train_folder = '/content/drive/MyDrive/DOG/train/'
train_data['ImagePath'] = train_data['nose print image'].apply(lambda x: os.path.join(train_folder, x))
```

### 2. Feature Extraction
- Uses ResNet50 (pre-trained on ImageNet) to extract high-dimensional feature vectors from nose print images.

```python
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
features = model.predict(preprocessed_image)
```

### 3. Image Enhancement
- Low-quality images are enhanced using a sharpening kernel.

```python
def enhance_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    enhanced_image = cv2.filter2D(image, -1, kernel)
    return enhanced_image
```

### 4. Quality Check
- Uses Laplacian variance to ensure uploaded images are sharp enough for feature extraction.

```python
def is_image_sharp(image, threshold=100.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance > threshold
```

### 5. Matching
- Compares extracted features of the uploaded image with stored features using cosine similarity.

```python
similarity = cosine_similarity([new_image_features], [stored_features])[0][0]
```
- Outputs the best match if the similarity score exceeds the threshold (default: `0.8`).

---

## File Structure
```
project/
├── train_data.csv          # Metadata for training images
├── train/                 # Folder containing training nose print images
├── dog_features.npy       # Precomputed feature file
├── script.py              # Main Python script
└── README.md             # Project documentation
```

---

## Usage

1. **Train**: Generate and save features for all training images.
   ```bash
   python script.py --mode train
   ```

2. **Test**: Upload a new nose print image and find the best matching dog.
   ```bash
   python script.py --mode test
   ```

3. **Run in Colab**: Copy the code to a Google Colab notebook for seamless execution.

---

## Future Enhancements
- Use more advanced models like EfficientNet for feature extraction.
- Implement a web-based interface for easier uploads and matches.
- Add more robust data augmentation for training.

---

## License
This project is licensed under the MIT License.

---

## Acknowledgments
- Pre-trained ResNet50 model by TensorFlow/Keras.
- Laplacian variance for sharpness evaluation.
- OpenCV for image processing.
