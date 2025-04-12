import os
import numpy as np
from PIL import Image
import glob
import zipfile
import io
import tarfile
from io import BytesIO
from tqdm import tqdm
from sklearn.datasets import fetch_openml
from scipy.io import loadmat
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import MDS
from scipy.optimize import linear_sum_assignment

class COIL20Loader:
    def load(self, data_path, normalize=True, resize=(32, 32), verbose=False):
        """
        Load COIL-20 dataset: 1440 images, 20 objects, 72 views per object.
        Images resized to 32x32 to match the 1024 features.

        Args:
            data_path (str): Path to directory containing coil20.zip
            normalize (bool): Apply L2 normalization (default=True)
            resize (tuple): Target image size (default=(32,32))
            verbose (bool): Print detailed messages (default=False)

        Returns:
            X (np.ndarray): Image data (n_samples, height*width)
            y (np.ndarray): Class labels (n_samples,)
        """
        if verbose:
            print("Loading COIL-20 dataset...")

        # Path to the zip file
        zip_path = os.path.join(data_path, 'coil20.zip')
        extract_path = os.path.join(data_path, 'coil20_extracted')

        # Extract if not already extracted
        if not os.path.exists(extract_path):
            os.makedirs(extract_path, exist_ok=True)
            self._extract_zip(zip_path, extract_path, verbose)

        # Find all PNG images
        image_files = sorted(glob.glob(os.path.join(extract_path, 'coil-20', '*.png')))
        if not image_files:
            raise ValueError(f"No PNG images found in {extract_path}/coil-20")

        # Initialize data structures
        n_samples = len(image_files)
        n_features = resize[0] * resize[1]
        X = np.zeros((n_samples, n_features))
        y = np.zeros(n_samples, dtype=int)

        # Load and process images
        if verbose:
            print(f"Processing {n_samples} images, resizing to {resize}...")
        for i, file_path in enumerate(image_files):
            # Extract label from filename (objXX__YY.png -> XX)
            y[i] = int(os.path.basename(file_path)[3:].split('__')[0]) - 1  # 0-based

            # Load and process image
            with Image.open(file_path) as img:
                X[i] = np.array(
                    img.convert('L').resize(resize)  # Convert to grayscale and resize
                ).flatten()

            # Progress reporting
            if verbose and (i + 1) % 144 == 0:
                print(f"Processed {i + 1}/{n_samples} images")

        # Normalize if requested
        if normalize:
            X = self._normalize(X)

        return X, y

    def _extract_zip(self, zip_path, extract_path, verbose):
        """Extract zip file using Python's zipfile module."""
        if verbose:
            print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    def _normalize(self, X):
        """Apply L2 normalization."""
        norms = np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
        norms[norms == 0] = 1.0
        return X / norms

class JAFFELoader():
    def load(self, data_path, normalize=True, resize=(26, 26), verbose=False):
        """
        Load JAFFE dataset: 213 facial expression images from 10 Japanese female models.

        Args:
            data_path (str): Path to directory containing jaffe.zip.
            normalize (bool): Apply L2 normalization (default=True).
            resize (tuple): Target image size (default=(26,26)).
            verbose (bool): Print detailed messages (default=False).

        Returns:
            X (np.ndarray): Image data (n_samples, height*width).
            y (np.ndarray): Subject IDs (n_samples,).
        """
        if verbose:
            print("Loading JAFFE dataset...")

        # Path configuration
        zip_path = os.path.join(data_path, 'jaffe.zip')
        extract_path = os.path.join(data_path, 'jaffe_extracted')

        # Extract dataset if needed
        if not os.path.exists(extract_path):
            os.makedirs(extract_path, exist_ok=True)
            self._extract_zip(zip_path, extract_path)

        # Locate TIFF images
        image_files = sorted(glob.glob(os.path.join(extract_path, 'jaffe', '*.tiff')))
        if not image_files:
            raise ValueError(f"No TIFF images found in {extract_path}/jaffe")

        # Initialize data structures
        n_samples = len(image_files)
        X = np.zeros((n_samples, resize[0] * resize[1]))
        y = np.zeros(n_samples, dtype=int)
        subject_map = {}  # Maps subject codes to numeric IDs

        # Process images
        if verbose:
            print(f"Processing {n_samples} images, resizing to {resize}...")
        for i, file_path in enumerate(image_files):
            # Extract and encode subject ID (filename format: YM.DI3.66.tiff)
            subject_code = os.path.basename(file_path).split('.')[0]
            y[i] = self._get_subject_id(subject_code, subject_map)

            # Load and process image
            with Image.open(file_path) as img:
                X[i] = np.array(
                    img.convert('L').resize(resize)  # Grayscale conversion and resizing
                ).flatten()

            # Progress reporting
            if verbose and ((i + 1) % 20 == 0 or (i + 1) == n_samples):
                print(f"Processed {i + 1}/{n_samples} images")

        # Normalize if requested
        if normalize:
            X = self._normalize(X)
        return X, y

    def _get_subject_id(self, subject_code, subject_map):
        """Map subject codes to sequential numeric IDs."""
        if subject_code not in subject_map:
            subject_map[subject_code] = len(subject_map)
        return subject_map[subject_code]

    def _extract_zip(self, zip_path, extract_path):
        """Extract zip file using Python's zipfile module."""
        import zipfile
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    def _normalize(self, X):
        """Apply L2 normalization."""
        norms = np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
        norms[norms == 0] = 1.0
        return X / norms

class Pointing04Loader():
    def load(self, data_path, normalize=True, resize=(40, 28), verbose=False):
        """
        Load Pointing04 dataset: 2790 face images across 15 people.
        Each image is resized to 40x28 (1120 features).

        Args:
            data_path (str): Path to directory containing pointing04.zip.
            normalize (bool): Apply Min-Max normalization (default=True).
            resize (tuple): Target image size (default=(40,28)).
            verbose (bool): Print detailed messages (default=False).

        Returns:
            X (np.ndarray): Image data (n_samples, height*width).
            y (np.ndarray): Person IDs (0-based).
        """
        if verbose:
            print("Loading Pointing04 dataset...")

        # Path to the zip file
        zip_path = os.path.join(data_path, 'pointing04.zip')
        
        # Initialize feature matrix and labels
        X_list = []
        y_list = []
        
        try:
            # Open the zip file directly
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # List all files in the zip
                all_files = zip_ref.namelist()
                if verbose:
                    print(f"Files in zip: {len(all_files)} total files")
                    print(f"Sample files: {all_files[:5]}")
                
                # Filter for .tar.gz files with "Person" in the name
                tar_files = [f for f in all_files if f.endswith('.tar.gz') and 'Person' in f]
                if verbose:
                    print(f"Found {len(tar_files)} person tar.gz files")
                
                if len(tar_files) == 0:
                    raise ValueError("No person tar.gz files found in pointing04.zip")
                    
                # Process each tar.gz file directly from the zip
                for tar_file_name in tar_files:
                    if verbose:
                        print(f"Processing {tar_file_name}...")
                        
                    try:
                        # Extract the person number from the filename
                        file_name = os.path.basename(tar_file_name)
                        try:
                            person_num = int(file_name.split('Person')[1].split('-')[0])
                        except (IndexError, ValueError) as e:
                            if verbose:
                                print(f"Skipping {file_name} - could not extract person number: {e}")
                            continue
                        
                        # Extract the tar.gz file to memory
                        tar_data = BytesIO(zip_ref.read(tar_file_name))
                        
                        # Open the tar file directly
                        with tarfile.open(fileobj=tar_data, mode='r:gz') as tar:
                            # Process images in the tar
                            image_members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(('.jpg', '.jpeg', '.png'))]
                            if verbose:
                                print(f"Found {len(image_members)} images in {file_name}")
                            
                            for member in image_members:
                                f = tar.extractfile(member)
                                if f is not None:
                                    image_data = f.read()
                                    
                                    try:
                                        img = Image.open(BytesIO(image_data)).convert('L')
                                        img = img.resize(resize)
                                        X_list.append(np.array(img).flatten())
                                        y_list.append(person_num - 1)  # Zero-based indexing
                                    except Exception as e:
                                        if verbose:
                                            print(f"Error processing image {member.name}: {e}")
                    except Exception as e:
                        if verbose:
                            print(f"Error processing {tar_file_name}: {e}")
        except zipfile.BadZipFile:
            raise ValueError(f"Could not open {zip_path}. File may be corrupted.")
        except FileNotFoundError:
            raise ValueError(f"Zip file not found at {zip_path}.")
        
        if len(X_list) == 0:
            raise ValueError("No images were successfully processed from Pointing04 dataset")
            
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Apply normalization
        if normalize:
            min_vals = np.min(X, axis=1, keepdims=True)
            max_vals = np.max(X, axis=1, keepdims=True)
            norm = np.where(max_vals == min_vals, 1, max_vals - min_vals)
            X = (X - min_vals) / norm
            
        return X, y
    
class UMISTLoader:
    def load(self, data_path, normalize=True, resize=(28, 23), verbose=False):
        """
        Load UMIST face dataset: 575 multiview face images of 20 people.
        Images resized to 28x23 pixels (644 features).

        Args:
            data_path (str): Path to directory containing umist.zip.
            normalize (bool): Apply L2 normalization (default=True).
            resize (tuple): Target image size (default=(28,23)).
            verbose (bool): Print detailed messages (default=False).

        Returns:
            X (np.ndarray): Image data (n_samples, height*width).
            y (np.ndarray): Person IDs (0-based).
        """
        
        if verbose:
            print("Loading UMIST dataset...")

        # Path to the zip file
        zip_path = os.path.join(data_path, 'umist.zip')
        
        # Initialize
        X_list = []
        y_list = []
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # List the contents
                if verbose:
                    all_files = zip_ref.namelist()
                    print(f"Found {len(all_files)} total files in the zip")
                    print(f"Sample paths: {all_files[:5]}")
                
                # Process each file
                for file_path in zip_ref.namelist():
                    # Check if it's a PGM file in a person folder
                    if file_path.endswith('.pgm'):
                        # Extract person ID from path
                        path_parts = file_path.split('/')
                        if len(path_parts) >= 2:
                            try:
                                # The second-to-last directory should be the person ID
                                person_str = path_parts[-2]
                                if person_str.isdigit():
                                    person_id = int(person_str)
                                    
                                    # Read the image data
                                    with zip_ref.open(file_path) as img_file:
                                        img_data = img_file.read()
                                        img = Image.open(BytesIO(img_data)).convert('L')
                                        img = img.resize(resize)
                                        
                                        # Add to our lists
                                        X_list.append(np.array(img).flatten())
                                        y_list.append(person_id - 1)  # Zero-based indexing
                            except (ValueError, IndexError) as e:
                                if verbose:
                                    print(f"Skipping {file_path}: {e}")
                
                # Check how many images we found
                if verbose:
                    print(f"Processed {len(X_list)} images across {len(set(y_list))} people")
        
        except zipfile.BadZipFile:
            raise ValueError(f"Could not open {zip_path}. File may be corrupted.")
        except FileNotFoundError:
            raise ValueError(f"Zip file not found at {zip_path}.")
        
        # Check if we found any images
        if len(X_list) == 0:
            raise ValueError("No images were found in the UMIST dataset")
        
        # Convert lists to arrays
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Apply L2 normalization
        if normalize:
            norms = np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
            norms[norms == 0] = 1.0
            X = X / norms
            
        return X, y
    
class YaleBLoader:
    def load(self, data_path, normalize=True, verbose=False):
        """
        Load Yale-B dataset from .mat file.

        Args:
            data_path (str): Path to directory containing YaleB_32x32.mat.
            normalize (bool): Apply Min-Max normalization (default=True).
            verbose (bool): Print detailed messages (default=False).

        Returns:
            X (np.ndarray): Image data (n_samples, 1024).
            y (np.ndarray): Person IDs (0-based).
        """
        if verbose:
            print("Loading Yale-B dataset...")

        mat_path = os.path.join(data_path, 'YaleB_32x32.mat')
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"MATLAB file not found at {mat_path}")

        try:
            mat_data = loadmat(mat_path)
            X = mat_data['fea'].astype(np.float64)
            y = mat_data['gnd'].squeeze().astype(np.int32)

            # Convert to 0-based indexing if needed
            if np.min(y) > 0:
                y = y - np.min(y)

            # Reshape images to 32x32 and rotate 90 degrees
            X = self._reshape_and_rotate(X)

            # Apply L2 normalization
            if normalize:
                norms = np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
                norms[norms == 0] = 1.0  # Avoid division by zero
                X = X / norms
            return X, y

        except Exception as e:
            raise RuntimeError(f"Error loading MATLAB file: {str(e)}")


    def _reshape_and_rotate(self, X):
        """Reshape images to 32x32 and rotate 90 degrees counterclockwise."""
        n_samples = X.shape[0]
        img_size = (32, 32)  # Target size is 32x32
        reshaped_images = np.zeros((n_samples, img_size[0], img_size[1]))  # (n_samples, height, width)

        for i in range(n_samples):
            img = X[i].reshape(img_size)  # Reshape to 32x32
            reshaped_images[i] = np.rot90(img, k=3)  # Rotate 90 degrees counterclockwise

        return reshaped_images.reshape(n_samples, -1)  # Flatten if needed

class USPSLoader:
    def load(self, normalize=True, resize=None, verbose=False):
        """Load USPS dataset using fetch_openml.

        Args:
            normalize (bool): Apply L2 normalization (default=True).
            resize (tuple): Target image size (default=None).
            verbose (bool): Print detailed messages (default=False).

        Returns:
            X (np.ndarray): Image data (n_samples, n_features).
            y (np.ndarray): Class labels (n_samples,).
        """
        if verbose:
            print("Loading USPS dataset using fetch_openml...")

        # Fetch the dataset
        data = fetch_openml(name='usps', version=2, as_frame=False)
        X = data.data
        y = data.target.astype(int)

        # Apply resizing if requested
        if resize and resize != (16, 16):
            if verbose:
                print(f"Resizing USPS images to {resize}...")
            X_resized = []
            for img in tqdm(X, desc="Resizing USPS images"):
                img_reshaped = img.reshape(16, 16)
                img_pil = Image.fromarray((img_reshaped * 255).astype(np.uint8))
                img_pil = img_pil.resize(resize)
                X_resized.append(np.array(img_pil).flatten() / 255.0)  # Normalize to [0, 1]
            X = np.array(X_resized)

        # Apply L2 normalization
        if normalize:
            norms = np.sqrt(np.sum(X**2, axis=1, keepdims=True))
            norms[norms == 0] = 1.0  # Avoid division by zero
            X = X / norms

        return X, y

class MNISTTLoader:
    def load(self, normalize=True, verbose=False):
        """
        Load MNIST-T dataset (first part of test set, 500 images per class).

        Args:
            normalize (bool): Apply L2 normalization (default=True).
            verbose (bool): Print detailed messages (default=False).

        Returns:
            X (np.ndarray): Image data (n_samples, n_features).
            y (np.ndarray): Class labels (n_samples,).
        """
        if verbose:
            print("Loading MNIST-T dataset...")

        # Get full MNIST dataset
        X_full, y_full = self.load_mnist_data(verbose)

        # Get test set (last 10000 samples)
        test_start_idx = len(X_full) - 10000
        X_test = X_full[test_start_idx:]
        y_test = y_full[test_start_idx:]

        # Extract first 5000 samples (500 per class)
        n_classes = 10
        samples_per_class = 500
        n_samples = n_classes * samples_per_class

        X = np.zeros((n_samples, X_test.shape[1]))
        y = np.zeros(n_samples, dtype=int)

        # Fill arrays with balanced samples
        class_counts = {i: 0 for i in range(10)}
        sample_idx = 0

        for idx in range(len(y_test)):
            digit = y_test[idx]
            if class_counts[digit] < samples_per_class:
                X[sample_idx] = X_test[idx]
                y[sample_idx] = digit
                class_counts[digit] += 1
                sample_idx += 1

            if all(count >= samples_per_class for count in class_counts.values()):
                break

        # Apply L2 normalization
        if normalize:
            norms = np.sqrt(np.sum(X**2, axis=1, keepdims=True))
            norms[norms == 0] = 1.0
            X = X / norms

        return X, y

    @staticmethod
    def load_mnist_data(verbose=False):
        """Load full MNIST dataset using fetch_openml."""

        if verbose:
            print("Fetching MNIST dataset from OpenML...")

        data = fetch_openml(name='mnist_784', version=1, as_frame=False)
        X = data.data / 255.0  # Normalize to [0, 1]
        y = data.target.astype(int)

        return X, y

class MNISTSLoader:
    def load(self, normalize=True, verbose=False):
        """
        Load MNIST-S dataset (1/10 sampling of MNIST).
        Sample one image per ten images, resulting in exactly 6996 images.

        Args:
            normalize (bool): Apply L2 normalization (default=True).
            verbose (bool): Print detailed messages (default=False).

        Returns:
            X (np.ndarray): Image data (n_samples, n_features).
            y (np.ndarray): Class labels (n_samples,).
        """
        if verbose:
            print("Loading MNIST-S dataset...")

        # Get full MNIST dataset
        X_full, y_full = self.load_mnist_data(verbose)

        # Sample every 10th image to get exactly 6996 samples
        indices = np.arange(0, len(X_full), 10)[:6996]

        X = X_full[indices]
        y = y_full[indices]

        # Check class balance
        unique, counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        if verbose:
            print(f"Class distribution: {class_distribution}")

        # Apply L2 normalization
        if normalize:
            norms = np.sqrt(np.sum(X**2, axis=1, keepdims=True))
            norms[norms == 0] = 1.0
            X = X / norms

        return X, y

    @staticmethod
    def load_mnist_data(verbose=False):
        """Load full MNIST dataset using fetch_openml."""
        if verbose:
            print("Fetching MNIST dataset from OpenML...")

        data = fetch_openml(name='mnist_784', version=1, as_frame=False)
        X = data.data / 255.0  # Normalize to [0, 1]
        y = data.target.astype(int)

        return X, y

class UMISTGaborLoader:
    def __init__(self, ksize=31, sigma=4.0, scales=None, gamma=0.5, psi=0, normalize=True, resize=(28, 23)):
        if scales is None:
            scales = [16, 12, 8, 6, 4]
        self.ksize = ksize
        self.sigma = sigma
        self.scales = scales
        self.gamma = gamma
        self.psi = psi
        self.normalize = normalize
        self.resize = resize
        self.gabor_kernels = self.create_gabor_kernels()

    def create_gabor_kernels(self):
        """Generate Gabor kernels based on the specified parameters."""
        theta_range = np.arange(0, np.pi, np.pi / 8)
        gabor_kernels = [
            cv2.getGaborKernel((self.ksize, self.ksize), self.sigma, theta, scale, self.gamma, self.psi, ktype=cv2.CV_32F)
            for scale in self.scales for theta in theta_range
        ]
        return gabor_kernels

    def load_dataset(self, data_path, verbose=False):
        """Load the UMIST dataset directly from zip file."""
        if verbose:
            print("Loading UMIST dataset...")

        # Path to the zip file
        zip_path = os.path.join(data_path, 'umist.zip')
        
        # Initialize data collection
        X_list = []
        y_list = []
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # List the contents to debug
                if verbose:
                    all_files = zip_ref.namelist()
                    print(f"Found {len(all_files)} total files in the zip")
                    print(f"Sample paths: {all_files[:5]}")
                
                # Process each file
                for file_path in zip_ref.namelist():
                    # Check if it's a PGM file in a person folder
                    if file_path.endswith('.pgm'):
                        # Extract person ID from path
                        path_parts = file_path.split('/')
                        if len(path_parts) >= 2:
                            try:
                                # The second-to-last directory should be the person ID
                                person_str = path_parts[-2]
                                if person_str.isdigit():
                                    person_id = int(person_str)
                                    
                                    # Read the image data
                                    with zip_ref.open(file_path) as img_file:
                                        img_data = img_file.read()
                                        img = Image.open(BytesIO(img_data)).convert('L')
                                        img = img.resize(self.resize)
                                        
                                        # Add to our lists
                                        X_list.append(np.array(img).flatten())
                                        y_list.append(person_id - 1)  # Zero-based indexing
                            except (ValueError, IndexError) as e:
                                if verbose:
                                    print(f"Skipping {file_path}: {e}")
                
                # Check how many images we found
                if verbose:
                    print(f"Processed {len(X_list)} images across {len(set(y_list))} people")
        
        except zipfile.BadZipFile:
            raise ValueError(f"Could not open {zip_path}. File may be corrupted.")
        except FileNotFoundError:
            raise ValueError(f"Zip file not found at {zip_path}.")
        
        # Check if we found any images
        if len(X_list) == 0:
            raise ValueError("No images were found in the UMIST dataset")
        
        # Convert lists to arrays
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Apply L2 normalization
        if self.normalize:
            norms = np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
            norms[norms == 0] = 1.0
            X = X / norms
            
        return X, y

    def apply_gabor_filters(self, X, verbose=False):
        """Apply Gabor filters to the dataset."""
        if verbose:
            print("Applying Gabor filters...")

        n_samples = X.shape[0]
        H, W = self.resize  # Image dimensions
        n_filters = len(self.gabor_kernels)
        X_Gabor = np.zeros((n_samples, H * W * n_filters))

        for i in range(n_samples):
            image = X[i].reshape((H, W))
            for j, kernel in enumerate(self.gabor_kernels):
                filtered_image = cv2.filter2D(image, cv2.CV_32F, kernel)
                X_Gabor[i, j * (H * W):(j + 1) * (H * W)] = filtered_image.flatten()

        if verbose:
            print("Gabor filters applied successfully.")
        return X_Gabor

    def load(self, data_path, verbose=False):
        """Load the dataset, apply Gabor filters, and save the results."""
        X, y = self.load_dataset(data_path, verbose)
        X_Gabor = self.apply_gabor_filters(X, verbose)
        return X_Gabor, y  # Return Gabor features and labels
    
class MPEG7Loader:
    @staticmethod
    def extract_shape_boundaries(image_np):
        """Extract shape boundaries from an image."""
        if len(image_np.shape) > 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        _, binary_image = cv2.threshold(image_np, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    @staticmethod
    def sample_points_from_contour(contour, n_points):
        """Sample n points from a contour with proper handling for small contours."""
        contour = np.vstack(contour).squeeze()

        if len(contour) >= n_points:
            indices = np.random.choice(len(contour), size=n_points, replace=False)
            return contour[indices]
        else:
            remaining = n_points - len(contour)
            additional_indices = np.random.choice(len(contour), size=remaining, replace=True)
            all_indices = np.concatenate([np.arange(len(contour)), additional_indices])
            return contour[all_indices]

    @staticmethod
    def compute_shape_context(points, n_bins_r=5, n_bins_theta=12):
        """Compute the shape context (log-polar histogram)."""
        n_points = len(points)
        shape_contexts = np.zeros((n_points, n_bins_r, n_bins_theta))

        for i in range(n_points):
            reference_point = points[i]
            relative_coords = points - reference_point
            r = np.sqrt(np.sum(relative_coords**2, axis=1))
            theta = np.arctan2(relative_coords[:, 1], relative_coords[:, 0])

            log_r = np.log10(r + 1e-10)
            r_bins = np.logspace(log_r.min(), log_r.max(), n_bins_r + 1)
            theta_bins = np.linspace(-np.pi, np.pi, n_bins_theta + 1)

            hist, _, _ = np.histogram2d(log_r, theta, bins=[r_bins, theta_bins])
            if np.sum(hist) > 0:
                hist /= np.sum(hist)

            shape_contexts[i] = hist

        return shape_contexts

    @staticmethod
    def chi_square_distance(hist1, hist2):
        """Compute the Chi-Square distance between two histograms."""
        return 0.5 * np.sum((hist1 - hist2)**2 / (hist1 + hist2 + 1e-10))

    @staticmethod
    def compute_distance_matrix(shape_contexts1, shape_contexts2):
        """Compute the distance matrix between two sets of shape contexts."""
        n1 = len(shape_contexts1)
        n2 = len(shape_contexts2)
        distance_matrix = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                distance_matrix[i, j] = MPEG7Loader.chi_square_distance(shape_contexts1[i], shape_contexts2[j])

        return distance_matrix

    @staticmethod
    def hungarian_matching(distance_matrix):
        """Find the optimal matching using the Hungarian algorithm."""
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        return distance_matrix[row_ind, col_ind].sum()

    def load(self, data_path, normalize=True, verbose=False):
        """Load MPEG7 dataset from a ZIP file."""
        n_points, n_bins_r, n_bins_theta, n_components = 70, 5, 12, 200
        contours_dict = {}
        zip_path = os.path.join(data_path, 'mpeg7.zip')

        if verbose:
            print("Opening ZIP file for MPEG7 dataset...")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            image_files = [f for f in file_list if f.endswith('.png')][:600]

            for image_file in image_files:
                with zip_ref.open(image_file) as file:
                    image = Image.open(io.BytesIO(file.read()))
                    image_np = np.array(image)
                    contours = self.extract_shape_boundaries(image_np)

                    fname = os.path.basename(image_file).replace(".png", "")
                    if contours:
                        contours_dict[fname] = contours[0]
                    else:
                        if verbose:
                            print(f"No contours found: {image_file}")

        n_samples = len(contours_dict)
        distance_matrices = np.zeros((n_samples, n_samples))

        if verbose:
            print(f"Processing {n_samples} samples...")

        for i, (label_i, contours_i) in enumerate(tqdm(contours_dict.items(), desc="Processing")):
            for j, (label_j, contours_j) in enumerate(contours_dict.items()):
                if j > i:
                    points_i = self.sample_points_from_contour(contours_i, n_points)
                    points_j = self.sample_points_from_contour(contours_j, n_points)

                    shape_contexts_i = self.compute_shape_context(points_i, n_bins_r, n_bins_theta)
                    shape_contexts_j = self.compute_shape_context(points_j, n_bins_r, n_bins_theta)

                    hungarian_dist = self.compute_distance_matrix(shape_contexts_i, shape_contexts_j)
                    dist = self.hungarian_matching(hungarian_dist)

                    distance_matrices[i, j] = dist
                    distance_matrices[j, i] = dist

        if verbose:
            print("Computing MDS transformation...")

        mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42)
        X = mds.fit_transform(distance_matrices)

        labels = list(contours_dict.keys())
        extracted_labels = [label.split('-')[0] for label in labels]
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(extracted_labels)

        if normalize:
            norms = np.sqrt(np.sum(X**2, axis=1, keepdims=True))
            norms[norms == 0] = 1.0
            X = X / norms

        if verbose:
            print("Loading complete. Returning data.")

        return X, y

class MPEG7GrayLoader:

    def load(self, data_path, verbose=False):
        num_images = 600
        threshold = 128
        X = []
        y = []
        zip_path = os.path.join(data_path, 'mpeg7.zip')

        if verbose:
            print("Opening ZIP file for MPEG7 Gray dataset...")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            image_files = [f for f in file_list if f.endswith('.png')]

            if verbose:
                print(f"Found {len(image_files)} images. Loading up to {num_images} images...")

            for image_file in image_files[:num_images]:
                with zip_ref.open(image_file) as file:
                    # Open the image and convert to grayscale
                    with Image.open(io.BytesIO(file.read())) as img:
                        img = img.convert('L')  # Convert to grayscale
                        img = img.resize((60, 100))

                        image_binary = img.point(lambda p: 255 if p > threshold else 0)
                        image_np = np.array(image_binary).flatten()
                        X.append(image_np)

                        fname = os.path.basename(image_file).replace(".png", "")
                        label = fname.split('-')[0]
                        y.append(label)

                        if verbose:
                            print(f"Processed image: {fname}")

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        return np.array(X), y