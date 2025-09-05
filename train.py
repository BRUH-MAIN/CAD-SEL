import os
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import argparse
from collections import defaultdict

from models.model import build_detection_model, convert_batch_yolo_to_rcnn
from dataloader import build_dataloaders


def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    # box format: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate areas of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union area
    union = box1_area + box2_area - intersection
    
    # Return IoU
    return intersection / union if union > 0 else 0


def calculate_map(all_detections, all_targets, iou_threshold=0.5, num_classes=None):
    """Calculate mAP (mean Average Precision)"""
    # Initialize variables
    precisions = defaultdict(list)
    recalls = defaultdict(list)
    average_precisions = {}
    
    # Calculate AP for each class
    for class_id in range(num_classes):
        # Collect all detections for this class
        detections = []
        for img_detections in all_detections:
            for det in img_detections:
                if det[5] == class_id:
                    # [x, y, w, h, score, class_id] -> [x1, y1, x2, y2, score]
                    x, y, w, h, score, _ = det
                    x1, y1 = x - w/2, y - h/2
                    x2, y2 = x + w/2, y + h/2
                    detections.append([x1, y1, x2, y2, score])
        
        # Collect all ground truth annotations for this class
        gt_boxes = []
        for img_targets in all_targets:
            for target in img_targets:
                if target[0] == class_id:
                    # [class_id, x, y, w, h] -> [x1, y1, x2, y2]
                    _, x, y, w, h = target
                    x1, y1 = x - w/2, y - h/2
                    x2, y2 = x + w/2, y + h/2
                    gt_boxes.append([x1, y1, x2, y2])
                    
        # Sort detections by confidence score
        detections.sort(key=lambda x: x[4], reverse=True)
        
        # If no ground truth or detections, AP is 0
        if len(gt_boxes) == 0 or len(detections) == 0:
            average_precisions[class_id] = 0
            continue
        
        # Create matching records
        gt_matched = [False] * len(gt_boxes)
        
        # Calculate TP and FP
        tp = np.zeros(len(detections))
        fp = np.zeros(len(detections))
        
        # Iterate through all detections
        for i, detection in enumerate(detections):
            # Find best matching ground truth
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt in enumerate(gt_boxes):
                if gt_matched[j]:
                    continue
                    
                iou = calculate_iou(detection[:4], gt)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            # Check if match is successful
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                gt_matched[best_gt_idx] = True
                tp[i] = 1
            else:
                fp[i] = 1
        
        # Calculate cumulative values
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # Calculate precision and recall
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / len(gt_boxes)
        
        # Add starting points
        precision = np.concatenate(([1], precision))
        recall = np.concatenate(([0], recall))
        
        # Calculate AP (using interpolation method)
        for i in range(len(precision) - 1, 0, -1):
            precision[i - 1] = max(precision[i - 1], precision[i])
            
        # Find recall change points
        indices = np.where(recall[1:] != recall[:-1])[0]
        
        # Calculate AP
        ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
        average_precisions[class_id] = ap
        
        # Save precision and recall curve data
        precisions[class_id] = precision
        recalls[class_id] = recall
        
    # Calculate mAP
    mean_ap = sum(average_precisions.values()) / len(average_precisions) if average_precisions else 0
    
    return mean_ap, average_precisions, precisions, recalls


def visualize_detections(image, detections, targets, num_classes, threshold=0.5, max_images=5):
    """Visualize detection results"""
    # Convert image from tensor to NumPy array
    img = image.permute(1, 2, 0).cpu().numpy()
    img = (img * np.array([0.24582985533315432, 0.19453490875333898, 0.15729866757757052]) + 
           np.array([0.5491333788465744, 0.3259111685958548, 0.2525661486929927])) * 255
    img = img.astype(np.uint8)
    
    # Create image copy for drawing
    img_with_dets = img.copy()
    h, w = img.shape[:2]
    
    # Draw detection results
    for det in detections:
        # [x, y, w, h, score, class_id]
        x_center, y_center, width, height, score, class_id = det
        
        if score < threshold:
            continue
            
        # Convert to pixel coordinates
        x1 = int((x_center - width/2) * w)
        y1 = int((y_center - height/2) * h)
        x2 = int((x_center + width/2) * w)
        y2 = int((y_center + height/2) * h)
        
        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        
        # Draw bounding box
        cv2.rectangle(img_with_dets, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add class and confidence labels
        label = f"Class {int(class_id)}: {score:.2f}"
        cv2.putText(img_with_dets, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw ground truth annotations (in red)
    for target in targets:
        # [class_id, x, y, w, h]
        class_id, x_center, y_center, width, height = target
        
        # Convert to pixel coordinates
        x1 = int((x_center - width/2) * w)
        y1 = int((y_center - height/2) * h)
        x2 = int((x_center + width/2) * w)
        y2 = int((y_center + height/2) * h)
        
        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        
        # Draw bounding box
        cv2.rectangle(img_with_dets, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Add class label
        label = f"Class {int(class_id)}"
        cv2.putText(img_with_dets, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return img_with_dets


def plot_precision_recall_curve(precisions, recalls, num_classes, output_dir):
    """Plot Precision-Recall curves"""
    plt.figure(figsize=(10, 8))
    
    for class_id, precision in precisions.items():
        recall = recalls[class_id]
        plt.plot(recall, precision, label=f"Class {class_id}")
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    plt.close()


def get_actual_num_classes(dataloader):
    """Get actual number of classes from label files"""
    unique_classes = set()
    
    # Iterate through all batches
    for batch in dataloader:
        labels = batch['labels']
        # Iterate through labels for each image
        for img_labels in labels:
            # Extract class IDs from all non-zero labels (first column)
            for label in img_labels:
                if label.sum() > 0:  # Ignore padded zero labels
                    class_id = int(label[0].item())
                    unique_classes.add(class_id)
    
    # Return max class ID + 1 (classes start from 0)
    return max(unique_classes) + 1 if unique_classes else 0


def evaluate_model(model, dataloader, device, num_classes, output_dir, epoch, class_weights=None, iou_threshold=0.5, conf_threshold=0.5, max_images=5):
    """Evaluate model performance and visualize results"""
    model.eval()
    all_detections = []
    all_targets = []
    all_losses = []
    
    # Create directory for saving visualization results
    vis_dir = os.path.join(output_dir, f'vis_epoch_{epoch}')
    os.makedirs(vis_dir, exist_ok=True)
    
    vis_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Evaluation')):
            images = batch['images'].to(device)
            yolo_targets = batch['labels']
            
            # Convert YOLO format labels to RCNN format
            rcnn_targets = convert_batch_yolo_to_rcnn(images, yolo_targets)
            rcnn_targets = [{k: v.to(device) for k, v in t.items()} for t in rcnn_targets]
            
            # Get prediction results
            detections = model.model(images)
            
            # Calculate loss (using training mode)
            model.train()
            loss_dict = model(images, rcnn_targets, class_weights=class_weights)
            model.eval()
            
            loss = sum(loss for loss in loss_dict.values())
            all_losses.append(loss.item())
            
            # Process each image
            for i in range(len(images)):
                # Get prediction results for current image
                boxes = detections[i]['boxes'].cpu()
                scores = detections[i]['scores'].cpu()
                labels = detections[i]['labels'].cpu() - 1  # Convert to YOLO class index (subtract 1)
                
                # Filter high confidence predictions
                keep = scores > conf_threshold
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
                
                # Convert to YOLO format [x, y, w, h, score, class_id]
                h, w = images[i].shape[1:]
                image_dets = []
                
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box.tolist()
                    
                    # Convert to YOLO format
                    x_center = (x1 + x2) / 2 / w
                    y_center = (y1 + y2) / 2 / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    
                    image_dets.append([x_center, y_center, width, height, score.item(), label.item()])
                
                all_detections.append(image_dets)
                
                # Get ground truth labels for current image
                img_targets = []
                for tgt in yolo_targets[i]:
                    if tgt.sum() > 0:  # Ignore padded labels
                        img_targets.append(tgt.tolist())
                
                all_targets.append(img_targets)
                
                # Visualize some images
                if vis_count < max_images:
                    vis_img = visualize_detections(
                        images[i], 
                        image_dets, 
                        img_targets, 
                        num_classes, 
                        threshold=conf_threshold
                    )
                    cv2.imwrite(os.path.join(vis_dir, f'vis_{batch_idx}_{i}.jpg'), 
                               cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
                    vis_count += 1
    
    # Calculate mAP
    mean_ap, average_precisions, precisions, recalls = calculate_map(
        all_detections, all_targets, iou_threshold, num_classes
    )
    
    # Plot Precision-Recall curves
    plot_precision_recall_curve(precisions, recalls, num_classes, output_dir)
    
    # Calculate average loss
    avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0
    
    # Calculate AP for each class
    class_ap = {f"Class {class_id}": ap for class_id, ap in average_precisions.items()}
    
    return {
        'loss': avg_loss,
        'mAP': mean_ap,
        'AP_per_class': class_ap
    }


def calculate_class_weights(class_distribution):
    """Calculate weights based on class distribution"""
    # Class distribution should be a list or dict containing sample counts for each class
    
    if isinstance(class_distribution, dict):
        counts = np.array(list(class_distribution.values()))
    else:
        counts = np.array(class_distribution)
    
    # Calculate weights: inverse of sample counts, then normalize
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(weights)  # Normalize so sum equals number of classes
    
    return weights


def train(args):
    # Configure parameters
    config = {
        'images_dir': args.images_dir,
        'labels_dir': args.labels_dir,
        'img_size': args.img_size,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'model_type': args.model_type,
        'lr': args.learning_rate,
        'output_dir': args.output_dir,
        'iou_threshold': args.iou_threshold,
        'conf_threshold': args.conf_threshold,
    }
    
    # Set up GPU
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Load data
    print("Loading data...")
    dataloaders = build_dataloaders(
        images_root_dir=config['images_dir'],
        labels_root_dir=config['labels_dir'],
        img_size=config['img_size'],
        batch_size=config['batch_size']
    )
    
    # Get actual number of classes from label files
    print("Determining actual number of classes from label files...")
    actual_num_classes = get_actual_num_classes(dataloaders['train'])
    print(f"Found {actual_num_classes} unique classes in label files")
    
    # Initialize model
    print(f"Building {config['model_type']} model...")
    num_classes = actual_num_classes + 1  # +1 for background class
    
    model = build_detection_model(
        num_classes=num_classes, 
        model_type=config['model_type']
    ).to(device)
    
    # Count samples for each class
    class_counts = [0] * actual_num_classes
    for batch in dataloaders['train']:
        labels = batch['labels']
        for img_labels in labels:
            for label in img_labels:
                if label.sum() > 0:
                    class_id = int(label[0].item())
                    if class_id < actual_num_classes:  # Prevent index out of bounds
                        class_counts[class_id] += 1
    
    print(f"Class distribution: {class_counts}")
    
    # Calculate weights based on actual class distribution
    class_weights = calculate_class_weights(class_counts)
    class_weights = torch.tensor(class_weights, device=device)
    print(f"Class weights: {class_weights}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Record training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'mAP': [],
        'lr': []
    }
    
    # Training loop
    print("Starting training...")
    best_map = 0
    
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch in tqdm(dataloaders['train'], desc=f'Epoch {epoch+1}/{config["epochs"]}'):
            images = batch['images'].to(device)
            targets = convert_batch_yolo_to_rcnn(images, batch['labels'])
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Calculate loss using class weights
            loss_dict = model(images, targets, class_weights=class_weights)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += losses.item()
        
        train_loss /= len(dataloaders['train'])
        history['train_loss'].append(train_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Evaluation phase
        print("Evaluating model...")
        eval_results = evaluate_model(
            model, 
            dataloaders['val'], 
            device, 
            actual_num_classes,  # Use actual number of classes
            config['output_dir'],
            epoch,
            class_weights=class_weights,
            iou_threshold=config['iou_threshold'],
            conf_threshold=config['conf_threshold']
        )
        
        val_loss = eval_results['loss']
        mAP = eval_results['mAP']
        
        history['val_loss'].append(val_loss)
        history['mAP'].append(mAP)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print evaluation results
        print(f"Epoch {epoch+1}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  mAP@{config['iou_threshold']}: {mAP:.4f}")
        print("  AP per class:")
        for class_name, ap in eval_results['AP_per_class'].items():
            print(f"    {class_name}: {ap:.4f}")
        
        # Save best model
        if mAP > best_map:
            best_map = mAP
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mAP': mAP,
                'val_loss': val_loss,
                'class_weights': class_weights.cpu(),
                'num_classes': actual_num_classes,
                'model_type': config['model_type']
            }, os.path.join(config['output_dir'], f'best_model_{config["model_type"]}.pth'))
            print(f"  Saved best model with mAP: {mAP:.4f}")
        
        # Save training history
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['mAP'], label='mAP')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config['output_dir'], f'training_history_{config["model_type"]}.png'))
        plt.close()


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train an object detection model')
    
    # Required parameters
    parser.add_argument('--model_type', type=str, default='fasterrcnn_resnet', 
                       choices=["fasterrcnn_resnet", "fasterrcnn_resnet50_fpn_v2", 
                               "fasterrcnn_mobilenet", "fasterrcnn_mobilenet_320",
                               "fasterrcnn_mobilenet_320_v2",
                               "ssd", "ssdlite_mobilenet_large", "ssdlite_mobilenet_small",
                               "fcos_resnet",
                               "retinanet", "retinanet_v2"],
                       help='Detection model type: fasterrcnn_resnet, fasterrcnn_resnet50_fpn_v2, fasterrcnn_mobilenet, fasterrcnn_mobilenet_320, fasterrcnn_mobilenet_320_v2, ssd, ssdlite_mobilenet_large, ssdlite_mobilenet_small, fcos_resnet, retinanet, or retinanet_v2')
    
    # Dataset parameters
    parser.add_argument('--images_dir', type=str, default='data/Images/Internal/NET-WL/',
                        help='Directory containing training images')
    parser.add_argument('--labels_dir', type=str, default='data/Labels/Internal/NET-WL/',
                        help='Directory containing label files')
    parser.add_argument('--img_size', type=int, default=640,
                        help='Image size for training')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    
    # Evaluation parameters
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='IoU threshold for mAP calculation')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                        help='Confidence threshold for detections')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save outputs')
    
    # GPU parameters
    parser.add_argument('--gpu', type=str, default=None,
                        help='GPU ID(s) to use (e.g., "0" or "0,1,2")')
    
    args = parser.parse_args()
    
    train(args)