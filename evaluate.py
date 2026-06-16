import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
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


def calculate_iou_metrics(all_detections, all_targets, num_classes=None):
    """Calculate IoU metrics"""
    # Initialize variables
    class_ious = defaultdict(list)
    overall_ious = []
    
    # Calculate IoU for each class
    for class_id in range(num_classes):
        # Collect all detections for this class
        detections = []
        for img_detections in all_detections:
            for det in img_detections:
                if det[5] == class_id:
                    # [x, y, w, h, score, class_id] -> [x1, y1, x2, y2]
                    x, y, w, h, score, _ = det
                    x1, y1 = x - w/2, y - h/2
                    x2, y2 = x + w/2, y + h/2
                    detections.append([x1, y1, x2, y2])
        
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
        
        # Skip if no ground truth or detections
        if len(gt_boxes) == 0 or len(detections) == 0:
            continue
        
        # Create matching records
        gt_matched = [False] * len(gt_boxes)
        
        # Iterate through all detections
        for detection in detections:
            # Find best matching ground truth
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt in enumerate(gt_boxes):
                if gt_matched[j]:
                    continue
                    
                iou = calculate_iou(detection, gt)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            # If match found, record IoU
            if best_gt_idx >= 0:
                gt_matched[best_gt_idx] = True
                class_ious[class_id].append(best_iou)
                overall_ious.append(best_iou)
    
    # Calculate mean IoU for each class
    class_mean_ious = {}
    for class_id, ious in class_ious.items():
        class_mean_ious[class_id] = sum(ious) / len(ious) if ious else 0
    
    # Calculate overall mean IoU
    overall_mean_iou = sum(overall_ious) / len(overall_ious) if overall_ious else 0
    
    return overall_mean_iou, class_mean_ious


def visualize_detections(image, detections, targets, num_classes, threshold=0.5):
    """Visualize detection results - display labels and predictions side by side"""
    # Convert image from tensor to NumPy array
    img = image.permute(1, 2, 0).cpu().numpy()
    
    # Denormalize using same mean and std as in NETDataset class
    img = (img * np.array([0.24582985533315432, 0.19453490875333898, 0.15729866757757052]) + 
           np.array([0.5491333788465744, 0.3259111685958548, 0.2525661486929927])) * 255
    img = img.astype(np.uint8)
    
    # Create two image copies for drawing labels and predictions separately
    img_with_targets = img.copy()
    img_with_dets = img.copy()
    h, w = img.shape[:2]
    
    # Define dark red color
    dark_red = (189, 0, 0)  # Dark red (RGB: 189, 0, 0)
    
    # Create custom class name mapping
    custom_class_names = {}
    if num_classes == 2:  # Merged class mode
        custom_class_names = {
            0: "NET",
            1: "Non-NET"
        }
    else:  # Normal mode
        custom_class_names = {
            0: "NET",
            1: "Leiomyoma",
            2: "Lipoma",
            3: "Non-tumor"
        }
    
    # Draw detection results (predictions)
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
        
        # Draw bounding box (using dark red)
        cv2.rectangle(img_with_dets, (x1, y1), (x2, y2), dark_red, 2)
        
        # Add class and confidence labels, using custom class names
        class_name = custom_class_names.get(int(class_id), f"Class {int(class_id)}")
        label = f"{class_name}: {score:.2f}"
        cv2.putText(img_with_dets, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, dark_red, 2)
    
    # Draw ground truth annotations
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
        
        # Draw bounding box (using dark red)
        cv2.rectangle(img_with_targets, (x1, y1), (x2, y2), dark_red, 2)
        
        # Add class label, using custom class names
        class_name = custom_class_names.get(int(class_id), f"Class {int(class_id)}")
        label = f"{class_name}"
        cv2.putText(img_with_targets, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, dark_red, 2)
    
    # Create a side-by-side image containing both images
    combined_img = np.zeros((h, w*2, 3), dtype=np.uint8)
    
    # Add titles
    # cv2.putText(img_with_targets, "Ground Truth", (10, 30), 
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # cv2.putText(img_with_dets, "Predictions", (10, 30), 
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Place two images side by side
    combined_img[:, :w] = img_with_targets
    combined_img[:, w:] = img_with_dets
    
    return combined_img


def plot_precision_recall_curve(precisions, recalls, class_names, output_dir, average_precisions=None):
    """Plot Precision-Recall curves"""
    plt.figure(figsize=(12, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(precisions)))
    
    for i, (class_id, precision) in enumerate(precisions.items()):
        recall = recalls[class_id]
        class_name = class_names.get(class_id, f"Class {class_id}")
        
        # If AP values are provided, display them in the label
        if average_precisions and class_id in average_precisions:
            label = f"{class_name} (AP: {average_precisions[class_id]:.4f})"
        else:
            label = f"{class_name}"
            
        plt.plot(recall, precision, label=label, 
                 color=colors[i % len(colors)], linewidth=2)
    
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curves by Class', fontsize=16)
    plt.legend(loc='lower left', fontsize=12)
    plt.grid(True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_ap_by_class(average_precisions, class_names, output_dir):
    """Plot AP bar chart for each class"""
    plt.figure(figsize=(14, 8))
    
    # Sorted class IDs
    sorted_class_ids = sorted(average_precisions.keys(), 
                             key=lambda k: average_precisions[k], 
                             reverse=True)
    
    # Get corresponding AP values and class names
    aps = [average_precisions[class_id] for class_id in sorted_class_ids]
    names = [class_names.get(class_id, f"Class {class_id}") for class_id in sorted_class_ids]
    
    # Set color mapping
    colors = plt.cm.viridis(np.array(aps) / max(aps))
    
    # Draw bar chart
    bars = plt.bar(range(len(aps)), aps, color=colors)
    
    # Set axis labels and title
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Average Precision (AP)', fontsize=14)
    plt.title('AP by Class (mAP: {:.4f})'.format(sum(aps) / len(aps)), fontsize=16)
    
    # Set x-axis ticks
    plt.xticks(range(len(aps)), names, rotation=45, ha='right')
    
    # Add value labels
    for bar, ap in zip(bars, aps):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{ap:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ap_by_class.png'), dpi=300, bbox_inches='tight')
    plt.close()


def calculate_precision_recall_f1(all_detections, all_targets, iou_threshold=0.5, num_classes=None):
    """Calculate precision/recall/F1 metrics"""
    # Initialize variables
    class_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    overall_metrics = {'tp': 0, 'fp': 0, 'fn': 0}
    
    # Calculate metrics for each class
    for class_id in range(num_classes):
        # Collect all detections for this class
        detections = []
        for img_detections in all_detections:
            for det in img_detections:
                if det[5] == class_id:
                    # [x, y, w, h, score, class_id] -> [x1, y1, x2, y2]
                    x, y, w, h, score, _ = det
                    x1, y1 = x - w/2, y - h/2
                    x2, y2 = x + w/2, y + h/2
                    detections.append([x1, y1, x2, y2])
        
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
        
        # Skip if no ground truth or detections
        if len(gt_boxes) == 0 and len(detections) == 0:
            continue
        
        # Create matching records
        gt_matched = [False] * len(gt_boxes)
        
        # Iterate through all detections
        for detection in detections:
            # Find best matching ground truth
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt in enumerate(gt_boxes):
                if gt_matched[j]:
                    continue
                    
                iou = calculate_iou(detection, gt)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            # Check if match is successful
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                gt_matched[best_gt_idx] = True
                class_metrics[class_id]['tp'] += 1
                overall_metrics['tp'] += 1
            else:
                class_metrics[class_id]['fp'] += 1
                overall_metrics['fp'] += 1
        
        # Calculate false negatives
        class_metrics[class_id]['fn'] = len(gt_boxes) - sum(gt_matched)
        overall_metrics['fn'] += class_metrics[class_id]['fn']
    
    # Calculate metrics for each class
    class_results = {}
    for class_id, metrics in class_metrics.items():
        tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
        
        # Calculate precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Calculate recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Calculate F1-score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_results[class_id] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # Calculate overall metrics
    tp, fp, fn = overall_metrics['tp'], overall_metrics['fp'], overall_metrics['fn']
    
    # Calculate overall precision
    overall_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Calculate overall recall
    overall_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Calculate overall F1-score
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    overall_results = {
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1
    }
    
    return class_results, overall_results


def evaluate_model(model, dataloader, device, num_classes, class_names, output_dir, conf_threshold=0.5, max_images=300, merge_classes=False):
    """Evaluate model performance and visualize results"""
    model.eval()
    all_detections = []
    all_targets = []
    all_losses = []
    
    # Create directory for saving visualization results
    vis_dir = os.path.join(output_dir, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Initialize visualization counter for each class
    class_vis_counts = {class_id: 0 for class_id in range(num_classes)}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Evaluation')):
            images = batch['images'].to(device)
            yolo_targets = batch['labels']
            
            # Convert YOLO format labels to RCNN format
            rcnn_targets = convert_batch_yolo_to_rcnn(images, yolo_targets)
            rcnn_targets = [{k: v.to(device) for k, v in t.items()} for t in rcnn_targets]
            
            # Get prediction results
            detections = model.model(images)
            
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
                    
                    # If merging classes, merge classes 1,2,3 into class 1
                    if merge_classes and label.item() > 0:
                        label_value = 1
                    else:
                        label_value = label.item()
                    
                    image_dets.append([x_center, y_center, width, height, score.item(), label_value])
                
                all_detections.append(image_dets)
                
                # Get ground truth labels for current image
                img_targets = []
                for tgt in yolo_targets[i]:
                    if tgt.sum() > 0:  # Ignore padded labels
                        # Get label data
                        tgt_data = tgt.tolist()
                        
                        # If merging classes, merge classes 1,2,3 into class 1
                        if merge_classes and tgt_data[0] > 0:
                            tgt_data[0] = 1
                        
                        img_targets.append(tgt_data)
                
                all_targets.append(img_targets)
                
                # Check if classes in current image need visualization
                should_visualize = False
                for target in img_targets:
                    class_id = target[0]
                    if class_vis_counts[class_id] < max_images:
                        should_visualize = True
                        break
                
                # If visualization is needed, save image and update counter
                if should_visualize:
                    vis_img = visualize_detections(
                        images[i], 
                        image_dets, 
                        img_targets, 
                        num_classes=2 if merge_classes else num_classes,  
                        threshold=conf_threshold
                    )
                    
                    # Get classes in detection results
                    det_classes = set()
                    for det in image_dets:
                        det_classes.add(int(det[5]))  # det[5] is class id
                    
                    # Get classes in ground truth labels
                    target_classes = set()
                    for target in img_targets:
                        target_classes.add(int(target[0]))  # target[0] is class id
                    
                    # Only save image if detection results and ground truth have same classes
                    if det_classes & target_classes:  # Use set intersection operation
                        cv2.imwrite(os.path.join(vis_dir, f'vis_{batch_idx}_{i}.jpg'), 
                                   cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
                    
                    # Update counter for each class (regardless of whether image is saved)
                    for target in img_targets:
                        class_id = target[0]
                        if class_vis_counts[class_id] < max_images:
                            class_vis_counts[class_id] += 1
    
    # Calculate mAP and metrics at IoU=0.5
    actual_num_classes = 2 if merge_classes else num_classes
    mean_ap_50, average_precisions_50, precisions_50, recalls_50 = calculate_map(
        all_detections, all_targets, iou_threshold=0.5, num_classes=actual_num_classes
    )
    
    # Calculate mAP and metrics at IoU=0.75
    mean_ap_75, average_precisions_75, precisions_75, recalls_75 = calculate_map(
        all_detections, all_targets, iou_threshold=0.75, num_classes=actual_num_classes
    )
    
    # Calculate IoU metrics
    overall_mean_iou, class_mean_ious = calculate_iou_metrics(
        all_detections, all_targets, num_classes=actual_num_classes
    )
    
    # Calculate precision/recall/F1 at IoU=0.5
    class_prf1_50, overall_prf1_50 = calculate_precision_recall_f1(
        all_detections, all_targets, iou_threshold=0.5, num_classes=actual_num_classes
    )
    
    # Calculate precision/recall/F1 at IoU=0.75
    class_prf1_75, overall_prf1_75 = calculate_precision_recall_f1(
        all_detections, all_targets, iou_threshold=0.75, num_classes=actual_num_classes
    )
    
    # Create custom class name mapping
    custom_class_names = {}
    if merge_classes:
        custom_class_names = {
            0: "NET",      # Class 0 remains as NET
            1: "Non-NET"   # Merge classes 1,2,3 into Non-NET
        }
    else:
        custom_class_names = {
            0: "NET",           # Class 0 -> NET
            1: "Leiomyoma",     # Class 1 -> Leiomyoma 
            2: "Lipoma",        # Class 2 -> Lipoma
            3: "Non-tumor"      # Class 3 -> Non-tumor
        }
    
    # Plot Precision-Recall curves (IoU=0.5)
    plot_precision_recall_curve(precisions_50, recalls_50, custom_class_names, output_dir, average_precisions_50)
    
    # Plot AP bar chart for each class (IoU=0.5)
    plot_ap_by_class(average_precisions_50, custom_class_names, output_dir)
    
    # Calculate metrics for each class, using custom class names
    class_ap_50 = {custom_class_names.get(class_id, f"Class {class_id}"): ap 
                  for class_id, ap in average_precisions_50.items()}
    class_ap_75 = {custom_class_names.get(class_id, f"Class {class_id}"): ap 
                  for class_id, ap in average_precisions_75.items()}
    class_iou = {custom_class_names.get(class_id, f"Class {class_id}"): iou 
                for class_id, iou in class_mean_ious.items()}
    class_prf1_50_dict = {custom_class_names.get(class_id, f"Class {class_id}"): metrics 
                         for class_id, metrics in class_prf1_50.items()}
    class_prf1_75_dict = {custom_class_names.get(class_id, f"Class {class_id}"): metrics 
                         for class_id, metrics in class_prf1_75.items()}
    
    return {
        'mAP@0.5': mean_ap_50,
        'mAP@0.75': mean_ap_75,
        'AP_per_class@0.5': class_ap_50,
        'AP_per_class@0.75': class_ap_75,
        'Overall_mIoU': overall_mean_iou,
        'IoU_per_class': class_iou,
        'PRF1_per_class@0.5': class_prf1_50_dict,
        'PRF1_per_class@0.75': class_prf1_75_dict,
        'Overall_PRF1@0.5': overall_prf1_50,
        'Overall_PRF1@0.75': overall_prf1_75,
        'average_precisions_50': average_precisions_50,
        'average_precisions_75': average_precisions_75,
        'precisions_50': precisions_50,
        'recalls_50': recalls_50,
        'precisions_75': precisions_75,
        'recalls_75': recalls_75
    }


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate object detection model')
    parser.add_argument('--model_path', type=str, required=True, help='Model weight file path')
    parser.add_argument('--images_dir', type=str, required=True, help='Image folder path')
    parser.add_argument('--labels_dir', type=str, required=True, help='Label folder path')
    parser.add_argument('--output_dir', type=str, default='evaluation_output', help='Output folder path')
    parser.add_argument('--img_size', type=int, default=640, help='Image size')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--conf_threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--dataset', type=str, default='val', choices=['train', 'val', 'test'], help='Evaluation dataset')
    parser.add_argument('--model_type', type=str, default='fasterrcnn_resnet', help='Model type')
    parser.add_argument('--merge_classes', action='store_true', help='Merge all classes into one class for evaluation')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create custom class name mapping
    class_names = {}
    if args.merge_classes:
        class_names = {
            0: "NET",
            1: "Non-NET"
        }
    else:
        class_names = {
            0: "NET",
            1: "Leiomyoma",
            2: "Lipoma",
            3: "Non-tumor"
        }
    
    # Load data
    print(f"Loading {args.dataset} dataset...")
    # Check if it's External dataset
    is_external = "External" in args.images_dir
    
    dataloaders = build_dataloaders(
        images_root_dir=args.images_dir,
        labels_root_dir=args.labels_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        is_external=is_external,  # Add this parameter
        merge_classes=args.merge_classes
    )
    
    # If it's External dataset, force use 'val' as evaluation set
    if is_external:
        args.dataset = 'val'
    
    # Load model weights
    print(f"Loading model from {args.model_path}")
    # Set weights_only=False for compatibility with PyTorch 2.6+ changes
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    # Get model parameters
    num_classes = checkpoint.get('num_classes', 80) + 1  # +1 for background
    class_weights = checkpoint.get('class_weights', None)
    
    print(f"Building {args.model_type} model with {num_classes - 1} classes...")
    model = build_detection_model(
        num_classes=num_classes,
        model_type=args.model_type
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Print loaded model information
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"mAP on validation: {checkpoint.get('mAP', 'unknown')}")
    
    # Evaluate model
    print(f"Evaluating model on {args.dataset} dataset...")
    eval_results = evaluate_model(
        model=model,
        dataloader=dataloaders[args.dataset],
        device=device,
        num_classes=num_classes - 1,  # Subtract background class
        class_names=class_names,
        output_dir=args.output_dir,
        conf_threshold=args.conf_threshold,
        merge_classes=args.merge_classes
    )
    
    # Output evaluation results
    print("\nEvaluation Results:")
    print(f"mAP@0.5: {eval_results['mAP@0.5']:.4f}")
    print(f"mAP@0.75: {eval_results['mAP@0.75']:.4f}")
    print(f"Overall mIoU: {eval_results['Overall_mIoU']:.4f}")
    
    print("\nAP per class@0.5:")
    # Sort and output by AP value
    sorted_classes_50 = sorted(
        eval_results['AP_per_class@0.5'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    for class_name, ap in sorted_classes_50:
        print(f"  {class_name}: {ap:.4f}")
    
    print("\nAP per class@0.75:")
    # Sort and output by AP value
    sorted_classes_75 = sorted(
        eval_results['AP_per_class@0.75'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    for class_name, ap in sorted_classes_75:
        print(f"  {class_name}: {ap:.4f}")
    
    print("\nIoU per class:")
    # Sort and output by IoU value
    sorted_classes_iou = sorted(
        eval_results['IoU_per_class'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    for class_name, iou in sorted_classes_iou:
        print(f"  {class_name}: {iou:.4f}")
    
    print("\nPrecision/Recall/F1 per class@0.5:")
    for class_name, metrics in eval_results['PRF1_per_class@0.5'].items():
        print(f"  {class_name}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1-score: {metrics['f1']:.4f}")
    
    print("\nPrecision/Recall/F1 per class@0.75:")
    for class_name, metrics in eval_results['PRF1_per_class@0.75'].items():
        print(f"  {class_name}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1-score: {metrics['f1']:.4f}")
    
    print("\nOverall Precision/Recall/F1@0.5:")
    print(f"  Precision: {eval_results['Overall_PRF1@0.5']['precision']:.4f}")
    print(f"  Recall: {eval_results['Overall_PRF1@0.5']['recall']:.4f}")
    print(f"  F1-score: {eval_results['Overall_PRF1@0.5']['f1']:.4f}")
    
    print("\nOverall Precision/Recall/F1@0.75:")
    print(f"  Precision: {eval_results['Overall_PRF1@0.75']['precision']:.4f}")
    print(f"  Recall: {eval_results['Overall_PRF1@0.75']['recall']:.4f}")
    print(f"  F1-score: {eval_results['Overall_PRF1@0.75']['f1']:.4f}")
    
    # Save evaluation results to text file
    with open(os.path.join(args.output_dir, 'evaluation_results.txt'), 'w') as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Confidence threshold: {args.conf_threshold}\n\n")
        
        f.write(f"mAP@0.5: {eval_results['mAP@0.5']:.4f}\n")
        f.write(f"mAP@0.75: {eval_results['mAP@0.75']:.4f}\n")
        f.write(f"Overall mIoU: {eval_results['Overall_mIoU']:.4f}\n\n")
        
        f.write("AP per class@0.5:\n")
        for class_name, ap in sorted_classes_50:
            f.write(f"  {class_name}: {ap:.4f}\n")
        
        f.write("\nAP per class@0.75:\n")
        for class_name, ap in sorted_classes_75:
            f.write(f"  {class_name}: {ap:.4f}\n")
            
        f.write("\nIoU per class:\n")
        for class_name, iou in sorted_classes_iou:
            f.write(f"  {class_name}: {iou:.4f}\n")
        
        f.write("\nPrecision/Recall/F1 per class@0.5:\n")
        for class_name, metrics in eval_results['PRF1_per_class@0.5'].items():
            f.write(f"  {class_name}:\n")
            f.write(f"    Precision: {metrics['precision']:.4f}\n")
            f.write(f"    Recall: {metrics['recall']:.4f}\n")
            f.write(f"    F1-score: {metrics['f1']:.4f}\n")
        
        f.write("\nPrecision/Recall/F1 per class@0.75:\n")
        for class_name, metrics in eval_results['PRF1_per_class@0.75'].items():
            f.write(f"  {class_name}:\n")
            f.write(f"    Precision: {metrics['precision']:.4f}\n")
            f.write(f"    Recall: {metrics['recall']:.4f}\n")
            f.write(f"    F1-score: {metrics['f1']:.4f}\n")
        
        f.write("\nOverall Precision/Recall/F1@0.5:\n")
        f.write(f"  Precision: {eval_results['Overall_PRF1@0.5']['precision']:.4f}\n")
        f.write(f"  Recall: {eval_results['Overall_PRF1@0.5']['recall']:.4f}\n")
        f.write(f"  F1-score: {eval_results['Overall_PRF1@0.5']['f1']:.4f}\n")
        
        f.write("\nOverall Precision/Recall/F1@0.75:\n")
        f.write(f"  Precision: {eval_results['Overall_PRF1@0.75']['precision']:.4f}\n")
        f.write(f"  Recall: {eval_results['Overall_PRF1@0.75']['recall']:.4f}\n")
        f.write(f"  F1-score: {eval_results['Overall_PRF1@0.75']['f1']:.4f}\n")
    
    print(f"\nResults and visualizations saved to {args.output_dir}")


if __name__ == '__main__':
    main()
