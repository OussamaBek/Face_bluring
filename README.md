
# 🕵️‍♂️ Face Blurring with AI

This project is a Dockerized Python application that automatically detects and blurs faces in images using a pre-trained deep learning model. It's designed for privacy-focused applications, such as anonymizing faces in photos before sharing or publishing.

---

## 📦 Features

- 🤖 **Automatic Face Detection**: Utilizes a pre-trained TensorFlow model (`face.pb`) to detect faces in images.
- 🖼️ **Image Blurring**: Applies a blur effect to detected faces, preserving anonymity.
- 🐳 **Dockerized Setup**: Easily build and run the application in a containerized environment.
- 🧪 **Sample Images**: Includes example images (`R0010121.JPG.jpg`, `nicholas-green-nPz8akkUmDI-unsplash.jpg`) and their blurred counterparts for testing.

---

## 🛠️ Technologies Used

- **Python 3.x**
- **TensorFlow**: For loading and running the pre-trained face detection model.
- **OpenCV**: For image processing and applying blur effects.
- **Docker**: To containerize the application for easy deployment.

---

## 📁 Project Structure

```bash
Face_bluring/
├── .vscode/                     # VSCode configuration files
├── Dockerfile                   # Docker image definition
├── docker-compose.yml           # Docker Compose configuration
├── docker-compose.debug.yml     # Docker Compose configuration for debugging
├── auto_blur_image.py           # Main script to detect and blur faces
├── face.pb                      # Pre-trained TensorFlow face detection model
├── requirements.txt             # Python dependencies
├── R0010121.JPG.jpg             # Sample input image
├── R0010121_BLUR.jpg            # Blurred output image
├── nicholas-green-nPz8akkUmDI-unsplash.jpg         # Another sample input image
├── nicholas-green-nPz8akkUmDI-unsplash_BLUR.jpg    # Blurred output image
```

---

## 🚀 Getting Started

### Prerequisites

- [Docker](https://www.docker.com/get-started) installed on your machine.

### Build and Run with Docker

1. **Clone the repository**:

   ```bash
   git clone https://github.com/OussamaBek/Face_bluring.git
   cd Face_bluring
   ```

2. **Build the Docker image**:

   ```bash
   docker-compose build
   ```

3. **Run the application**:

   ```bash
   docker-compose up
   ```

   This will execute the `auto_blur_image.py` script inside the Docker container, processing the sample images and generating blurred versions.

---

## 🧪 Usage

To process your own images:

1. Place your image file in the project directory.

2. Modify the `auto_blur_image.py` script to specify your image file:

   ```python
   image_path = 'your_image.jpg'
   ```

3. Rebuild and run the Docker container:

   ```bash
   docker-compose build
   docker-compose up
   ```

   The script will process your image and output a blurred version in the same directory.



---

## 👨‍💻 Author

**Oussama Bek**  
[GitHub Profile](https://github.com/OussamaBek)
