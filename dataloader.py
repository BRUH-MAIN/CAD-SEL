import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import random


class NETDataset(Dataset):
    def __init__(self, 
                 images_root_dir,
                 labels_root_dir,
                 img_size=640,
                 augment=False,
                 patient_split=None,
                 merge_classes=False):
        """
        Dataset for NET-WL YOLO object detection.
        
        Args:
            images_root_dir (str): Root directory for images, e.g., 'data/Images/Internal/NET-WL/'
            labels_root_dir (str): Root directory for labels, e.g., 'data/Labels/Internal/NET-WL/'
            img_size (int): Size to resize images to
            augment (bool): Whether to apply data augmentation
            patient_split (dict, optional): Dict specifying which patients to include in this dataset
                                          e.g., {'WL-G1': ['XXX-G1', 'YYY-G1'], ...} or 'all'
            merge_classes (bool): Whether to map labels into NET=0 and non-NET=1
        """
        self.images_root_dir = images_root_dir
        self.labels_root_dir = labels_root_dir
        self.img_size = img_size
        self.augment = augment
        self.merge_classes = merge_classes
        
        # Find all image files and their corresponding labels
        self.image_files = []
        self.label_files = []
        self.image_categories = []
        self.classes = []
        
        # Get all category directories
        category_dirs = sorted([d for d in os.listdir(images_root_dir) 
                        if os.path.isdir(os.path.join(images_root_dir, d))])
        self.classes = category_dirs
        
        for category in category_dirs:
            category_dir = os.path.join(images_root_dir, category)
            
            # Get all patient directories 
            patient_dirs = sorted([d for d in os.listdir(category_dir) 
                           if os.path.isdir(os.path.join(category_dir, d))])
            
            # Filter patient directories if patient_split is specified
            if patient_split is not None and patient_split != 'all':
                if category in patient_split:
                    patient_dirs = [d for d in patient_dirs if d in patient_split[category]]
                else:
                    # Skip this category if it's not in the split
                    continue
            
            for patient in patient_dirs:
                patient_img_dir = os.path.join(category_dir, patient)
                patient_label_dir = os.path.join(labels_root_dir, category, patient)
                
                # Skip if label directory doesn't exist
                if not os.path.exists(patient_label_dir):
                    continue
                
                # Get all image files
                img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
                image_paths = []
                for ext in img_extensions:
                    image_paths.extend(glob.glob(os.path.join(patient_img_dir, ext)))
                
                for img_path in sorted(image_paths):
                    # Get the corresponding label file path
                    img_name = os.path.basename(img_path)
                    label_name = os.path.splitext(img_name)[0] + '.txt'
                    label_path = os.path.join(patient_label_dir, label_name)
                    
                    # Only add the pair if the label file exists
                    if os.path.exists(label_path):
                        self.image_files.append(img_path)
                        self.label_files.append(label_path)
                        self.image_categories.append(category)
        
        print(f"Loaded {len(self.image_files)} images from {len(category_dirs)} categories")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.label_files[idx]
        
        category = self.image_categories[idx]
        
        # Load image as numpy array
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]  # original height, width
        
        # Load labels (YOLO format: class x_center y_center width height)
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    values = line.strip().split()
                    if len(values) == 5:
                        class_id = int(values[0])
                        if self.merge_classes and class_id > 0:
                            class_id = 1
                        x_center = float(values[1])
                        y_center = float(values[2])
                        w = float(values[3])
                        h = float(values[4])
                        
                        labels.append([class_id, x_center, y_center, w, h])
        
        labels = np.array(labels)
        
        # Apply augmentations with correct label transformation
        if self.augment:
            # Random horizontal flip with 50% probability
            if random.random() < 0.5:
                img = np.fliplr(img)
                if len(labels):
                    # Flip x coordinates: new_x = 1 - x
                    labels[:, 1] = 1 - labels[:, 1]
            
            # Random color augmentation
            img = self.augment_color(img)
            
            # Apply random rotation (small angles only) with careful box rotation
            if random.random() < 0.3:  # 30% probability of rotation
                angle = random.uniform(-15, 15)  # Random angle between -15 and 15 degrees
                img, labels = self.rotate_image_and_boxes(img, labels, angle)
        
        # Resize image to target size
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Convert to tensor and normalize
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        img /= 255.0  # Normalize to [0, 1]
        
        # Apply ImageNet normalization
        normalize = transforms.Normalize(mean=[0.5491333788465744, 0.3259111685958548, 0.2525661486929927],
                std=[0.24582985533315432, 0.19453490875333898, 0.15729866757757052])
        img = normalize(img)
        
        # Convert labels to tensor
        if len(labels) == 0:
            labels = torch.zeros((0, 5), dtype=torch.float32)
        else:
            labels = torch.from_numpy(labels).float()
        
        return {
            'image': img,
            'labels': labels,
            'image_path': img_path,
            'category': category
        }
    
    def augment_color(self, img):
        """Apply color augmentation to image"""
        # Random brightness, contrast, saturation, hue adjustments
        hgain = random.uniform(0.8, 1.2)  # Hue gain
        sgain = random.uniform(0.8, 1.2)  # Saturation gain
        vgain = random.uniform(0.8, 1.2)  # Value gain
        
        # Convert to HSV and apply changes
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # Apply gains
        h = np.clip(h * hgain, 0, 255).astype(np.uint8)
        s = np.clip(s * sgain, 0, 255).astype(np.uint8)
        v = np.clip(v * vgain, 0, 255).astype(np.uint8)
        
        # Combine channels and convert back to RGB
        hsv = cv2.merge([h, s, v])
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return img
    
    def rotate_image_and_boxes(self, img, boxes, angle):
        """
        Rotate image and transform bounding boxes accordingly
        
        Args:
            img: Original image
            boxes: Normalized bounding boxes [class, x_center, y_center, width, height]
            angle: Rotation angle in degrees
            
        Returns:
            Rotated image and transformed boxes
        """
        # Get image dimensions
        h, w = img.shape[:2]
        
        # Calculate the rotation matrix
        center = (w/2, h/2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image dimensions after rotation
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust the rotation matrix to take into account the translation
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Perform the actual rotation
        rotated_img = cv2.warpAffine(img, M, (new_w, new_h), borderValue=(114, 114, 114))
        
        # If no boxes, return early
        if len(boxes) == 0:
            return rotated_img, boxes
        
        # Create a copy of boxes to modify
        new_boxes = boxes.copy()
        
        # Convert normalized coordinates to absolute
        new_boxes[:, 1] = boxes[:, 1] * w  # x center
        new_boxes[:, 2] = boxes[:, 2] * h  # y center
        new_boxes[:, 3] = boxes[:, 3] * w  # width
        new_boxes[:, 4] = boxes[:, 4] * h  # height
        
        # Convert center coordinates to corners (x1, y1, x2, y2)
        x1 = new_boxes[:, 1] - new_boxes[:, 3] / 2
        y1 = new_boxes[:, 2] - new_boxes[:, 4] / 2
        x2 = new_boxes[:, 1] + new_boxes[:, 3] / 2
        y2 = new_boxes[:, 2] + new_boxes[:, 4] / 2
        
        # Rotate the corner points
        corners = np.zeros((len(boxes) * 4, 2))
        corners[0::4, 0] = x1  # Top-left x
        corners[0::4, 1] = y1  # Top-left y
        corners[1::4, 0] = x2  # Top-right x
        corners[1::4, 1] = y1  # Top-right y
        corners[2::4, 0] = x2  # Bottom-right x
        corners[2::4, 1] = y2  # Bottom-right y
        corners[3::4, 0] = x1  # Bottom-left x
        corners[3::4, 1] = y2  # Bottom-left y
        
        # Apply the rotation matrix to all corners
        corners = corners.dot(M[:2, :2].T) + M[:2, 2]
        
        # Reshape back to get the rotated corners for each box
        corners = corners.reshape(-1, 8)
        
        # Calculate the new bounding boxes from the rotated corners
        x_corners = corners[:, [0, 2, 4, 6]]
        y_corners = corners[:, [1, 3, 5, 7]]
        
        # Get min/max to create new bounding box
        x_min = np.min(x_corners, axis=1)
        y_min = np.min(y_corners, axis=1)
        x_max = np.max(x_corners, axis=1)
        y_max = np.max(y_corners, axis=1)
        
        # Convert back to center format
        new_boxes[:, 1] = (x_min + x_max) / 2  # x center
        new_boxes[:, 2] = (y_min + y_max) / 2  # y center
        new_boxes[:, 3] = x_max - x_min  # width
        new_boxes[:, 4] = y_max - y_min  # height
        
        # Normalize coordinates to [0, 1]
        new_boxes[:, 1] /= new_w
        new_boxes[:, 2] /= new_h
        new_boxes[:, 3] /= new_w
        new_boxes[:, 4] /= new_h
        
        # Clip the boxes to ensure they are within image bounds
        new_boxes[:, 1:] = np.clip(new_boxes[:, 1:], 0, 1)
        
        # Filter out boxes that are too small after rotation
        valid_indices = (new_boxes[:, 3] > 0.01) & (new_boxes[:, 4] > 0.01)
        
        return rotated_img, new_boxes[valid_indices]


def create_train_val_splits(images_root_dir, train_ratio=0.8, seed=42):
    """
    Create train/validation splits based on patients (to avoid data leakage).
    
    Args:
        images_root_dir (str): Root directory for images
        train_ratio (float): Ratio of patients to use for training
        seed (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary containing train and val patient splits
    """
    random.seed(seed)
    
    train_split = {}
    val_split = {}
    
    # Get all category directories
    category_dirs = sorted([d for d in os.listdir(images_root_dir) 
                    if os.path.isdir(os.path.join(images_root_dir, d))])
    
    for category in category_dirs:
        category_dir = os.path.join(images_root_dir, category)
        
        # Get all patient directories and sort them
        patient_dirs = sorted([d for d in os.listdir(category_dir) 
                       if os.path.isdir(os.path.join(category_dir, d))])
        
        # Shuffle and split
        random.shuffle(patient_dirs)
        split_idx = int(len(patient_dirs) * train_ratio)
        
        train_patients = patient_dirs[:split_idx]
        val_patients = patient_dirs[split_idx:]
        
        train_split[category] = train_patients
        val_split[category] = val_patients
    
    return {
        'train': train_split,
        'val': val_split
    }


def build_dataloaders(images_root_dir, 
                     labels_root_dir,
                     img_size=640,
                     batch_size=16,
                     train_ratio=0.8,
                     num_workers=4,
                     seed=42,
                     is_external=False,
                     merge_classes=False):
    """
    Build train and validation dataloaders.
    
    Args:
        images_root_dir (str): Root directory for images
        labels_root_dir (str): Root directory for labels
        img_size (int): Size to resize images to
        batch_size (int): Batch size
        train_ratio (float): Ratio of patients to use for training
        num_workers (int): Number of workers for data loading
        seed (int): Random seed for reproducibility
        is_external (bool): Whether this is external dataset
        merge_classes (bool): Whether to map labels into NET=0 and non-NET=1
        
    Returns:
        dict: Dictionary containing train and val dataloaders
    """
    if is_external:
        # For External dataset, no splitting, all data used for validation
        dataset = NETDataset(
            images_root_dir=images_root_dir,
            labels_root_dir=labels_root_dir,
            img_size=img_size,
            augment=False,  # No data augmentation needed for validation
            patient_split='all',  # Use all data
            merge_classes=merge_classes
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        return {
            'train': None,  # External dataset doesn't need training set
            'val': dataloader,
            'test': dataloader,  # Can be used for test as well
            'num_classes': len(dataset.classes)
        }
    else:
        # For Internal dataset, maintain original splitting logic
        splits = create_train_val_splits(images_root_dir, train_ratio, seed)
        
        train_dataset = NETDataset(
            images_root_dir=images_root_dir,
            labels_root_dir=labels_root_dir,
            img_size=img_size,
            augment=True,
            patient_split=splits['train'],
            merge_classes=merge_classes
        )
        
        val_dataset = NETDataset(
            images_root_dir=images_root_dir,
            labels_root_dir=labels_root_dir,
            img_size=img_size,
            augment=False,
            patient_split=splits['val'],
            merge_classes=merge_classes
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        return {
            'train': train_dataloader,
            'val': val_dataloader,
            'num_classes': len(train_dataset.classes)
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable length labels.
    """
    images = torch.stack([item['image'] for item in batch])
    image_paths = [item['image_path'] for item in batch]
    categories = [item['category'] for item in batch]
    
    # Handle variable length label tensors
    max_labels = max([item['labels'].shape[0] for item in batch], default=0)
    
    if max_labels > 0:
        labels_batch = torch.zeros((len(batch), max_labels, 5))
        
        for i, item in enumerate(batch):
            if item['labels'].shape[0] > 0:
                labels_batch[i, :item['labels'].shape[0], :] = item['labels']
    else:
        labels_batch = torch.zeros((len(batch), 1, 5))
    
    return {
        'images': images,
        'labels': labels_batch,
        'image_paths': image_paths,
        'categories': categories
    }


# Example usage
if __name__ == "__main__":
    dataloaders = build_dataloaders(
        images_root_dir='data/Images/Internal/NET-WL/',
        labels_root_dir='data/Labels/Internal/NET-WL/',
        img_size=640,
        batch_size=8
    )
    
    # Print dataset information
    print(f"Number of classes: {dataloaders['num_classes']}")
    print(f"Training batches: {len(dataloaders['train'])}")
    print(f"Validation batches: {len(dataloaders['val'])}")
    
    # Get a batch from the training dataloader
    batch = next(iter(dataloaders['train']))
    images = batch['images']
    labels = batch['labels']
    
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Show first image with denormalization
    img = images[0].permute(1, 2, 0).cpu().numpy()
    img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    img = img.astype(np.uint8)
    
    # Draw bounding boxes
    img_with_boxes = img.copy()
    h, w = img.shape[:2]
    
    for box in labels[0]:
        if box.sum() == 0:  # Skip padded boxes
            continue
        
        class_id, x_center, y_center, width, height = box.tolist()
        
        # YOLO format to pixel coordinates
        x1 = int((x_center - width/2) * w)
        y1 = int((y_center - height/2) * h)
        x2 = int((x_center + width/2) * w)
        y2 = int((y_center + height/2) * h)
        
        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        
        # Draw rectangle if the box is valid
        if x2 > x1 and y2 > y1:
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_with_boxes, f"Class {int(class_id)}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save the image with boxes (for visualization)
    cv2.imwrite("sample_with_boxes.jpg", cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
    print("Sample image with bounding boxes saved to 'sample_with_boxes.jpg'")
