import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def phash(image):
    # Convert the image to grayscale and resize it to 256x256 pixels
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (256, 256))

    # Apply a Discrete Cosine Transform (DCT) to the image
    dct = cv2.dct(np.float32(resized))

    # Compute the mean of the DCT coefficients and use it as the hash value
    mean = np.mean(dct)
    hash_value = np.uint8(dct > mean)

    # Convert the hash value to a single float number
    float_hash = np.packbits(hash_value.flatten())
    float_hash = np.frombuffer(float_hash, dtype=np.float32)[0]

    return float_hash


class ImageSimilarityMatrix:
    def __init__(self, directory_path):
        self.directory_path = directory_path
        self.image_paths = []
        self.images = []
        self.hashed_images = []
        
        self.similarity_matrix = None

    def load_images(self):
        for filename in tqdm(os.listdir(self.directory_path), 'Loading images into memory...'):
            if filename.endswith(".jpg"):
                image_path = os.path.join(self.directory_path, filename)
                image = cv2.imread(image_path)
                
                if image is not None:
                    self.image_paths.append(filename)
                    self.images.append(image)
                    self.hashed_images.append(phash(image))
                    
        self.hashed_images = np.array(checker.hashed_images)

    def calculate_similarity_matrix(self):
        num_images = len(self.images)
        similarity_matrix = np.zeros((num_images, num_images), dtype=np.int8)

        for i in range(num_images):
                similarity_matrix[i] = (checker.hashed_images[i] == checker.hashed_images).astype(np.int8)

        self.similarity_matrix = similarity_matrix
        
    def get_duplicated(self):
        return np.where(self.similarity_matrix.sum(axis=0) > 1.0)[0]
    
    
if __name__ == "__main__":
    checker = ImageSimilarityMatrix('path/to/images')
    checker.load_images()
    checker.calculate_similarity_matrix()
    duplicated = checker.get_duplicated()
    
    sim_df = pd.DataFrame(
        data=checker.similarity_matrix,
        columns=checker.image_paths,
        index=checker.image_paths
    )
    
    sim_df.to_csv('savefile_name.csv', index=False)
    