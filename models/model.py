import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,
    ssd300_vgg16, SSD300_VGG16_Weights,
    retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights,
    retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights,
    fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights,
    ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSDHead


class DetectionModel(nn.Module):
    def __init__(self, num_classes, model_type='fasterrcnn_resnet', pretrained=True, **kwargs):
        """
        Object detection model wrapper
        
        Args:
            num_classes (int): Number of classes (including background class)
            model_type (str): Model type, options:
                Faster R-CNN series:
                - 'fasterrcnn_resnet' (ResNet50-FPN)
                - 'fasterrcnn_resnet50_fpn_v2' (ResNet50-FPN V2)
                - 'fasterrcnn_mobilenet' (MobileNetV3-Large-FPN)
                - 'fasterrcnn_mobilenet_320' (MobileNetV3-Large-320-FPN)
                - 'fasterrcnn_mobilenet_320_v2' (MobileNetV3-Large-320-FPN V2)
                
                SSD series:
                - 'ssd' (SSD300-VGG16)
                - 'ssdlite_mobilenet_large' (SSDLite320-MobileNetV3-Large)
                
                FCOS series:
                - 'fcos_resnet' (FCOS ResNet50-FPN)
                
                Others:
                - 'retinanet' (RetinaNet-ResNet50-FPN)
                - 'retinanet_v2' (RetinaNet-ResNet50-FPN V2)
            pretrained (bool): Whether to use pretrained weights
            **kwargs: Other parameters passed to base model
        """
        super(DetectionModel, self).__init__()
        
        self.num_classes = num_classes
        self.model_type = model_type
        
        # Initialize detection model
        if model_type == 'fasterrcnn_resnet':
            if pretrained:
                weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
                self.model = fasterrcnn_resnet50_fpn(weights=weights, **kwargs)
            else:
                self.model = fasterrcnn_resnet50_fpn(weights=None, **kwargs)
            
            # Replace classification head to match number of classes
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            
        elif model_type == 'fasterrcnn_resnet50_fpn_v2':
            if pretrained:
                weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
                self.model = fasterrcnn_resnet50_fpn_v2(weights=weights, **kwargs)
            else:
                self.model = fasterrcnn_resnet50_fpn_v2(weights=None, **kwargs)
            
            # Replace classification head to match number of classes
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            
        elif model_type == 'fasterrcnn_mobilenet':
            if pretrained:
                weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
                self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights, **kwargs)
            else:
                self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, **kwargs)
            
            # Replace classification head to match number of classes
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            
        elif model_type == 'fasterrcnn_mobilenet_320':
            if pretrained:
                weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
                self.model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights, **kwargs)
            else:
                self.model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, **kwargs)
            
            # Replace classification head to match number of classes
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            
        elif model_type == 'fasterrcnn_mobilenet_320_v2':
            # Since V2 version is not available, we use V1 version as substitute
            if pretrained:
                weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
                self.model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights, **kwargs)
            else:
                self.model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, **kwargs)
            
            # Replace classification head to match number of classes
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            
        elif model_type == 'ssd':
            if pretrained:
                weights = SSD300_VGG16_Weights.DEFAULT
                self.model = ssd300_vgg16(weights=weights, **kwargs)
            else:
                self.model = ssd300_vgg16(weights=None, **kwargs)
            
            # Fix: Construct SSD head directly
            if num_classes != 21:  # SSD pretrained model has 21 classes (20 + background)
                # Get correct configuration
                num_anchors = [4, 6, 6, 6, 4, 4]  # Default anchor count for SSD300
                
                # Get correct input channel count from model
                in_channels = [512, 1024, 512, 256, 256, 256]  # Fixed channel count for SSD300
                
                # Create new SSD head
                self.model.head = SSDHead(
                    in_channels=in_channels,
                    num_anchors=num_anchors,
                    num_classes=num_classes
                )
                
        elif model_type == 'ssdlite_mobilenet_large':
            if pretrained:
                weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
                self.model = ssdlite320_mobilenet_v3_large(weights=weights, **kwargs)
            else:
                self.model = ssdlite320_mobilenet_v3_large(weights=None, **kwargs)
            
        elif model_type == 'retinanet':
            if pretrained:
                weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
                self.model = retinanet_resnet50_fpn(weights=weights, **kwargs)
            else:
                self.model = retinanet_resnet50_fpn(weights=None, **kwargs)
            
            # RetinaNet doesn't need to replace classification head as it uses different prediction method
            
        elif model_type == 'retinanet_v2':
            if pretrained:
                weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
                self.model = retinanet_resnet50_fpn_v2(weights=weights, **kwargs)
            else:
                self.model = retinanet_resnet50_fpn_v2(weights=None, **kwargs)
            
            # RetinaNet doesn't need to replace classification head as it uses different prediction method
            
        elif model_type == 'fcos_resnet':
            if pretrained:
                weights = FCOS_ResNet50_FPN_Weights.DEFAULT
                self.model = fcos_resnet50_fpn(weights=weights, **kwargs)
            else:
                self.model = fcos_resnet50_fpn(weights=None, **kwargs)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def forward(self, images, targets=None, class_weights=None):
        """
        Forward pass
        
        Args:
            images (List[Tensor]): Input images
            targets (List[Dict[str, Tensor]], optional): Training targets
                Each dictionary contains:
                - boxes (FloatTensor[N, 4]): Bounding box coordinates (x1, y1, x2, y2)
                - labels (Int64Tensor[N]): Class labels
                - image_id (Int64Tensor[1]): Image ID
                - area (Tensor[N]): Bounding box area
                - iscrowd (UInt8Tensor[N]): Whether it's a crowd
            class_weights (Tensor, optional): Weight tensor for each class
        
        Returns:
            Training mode: Loss dictionary
            Inference mode: Prediction result list
        """
        if self.training and targets is not None:
            # Use original model to calculate loss dictionary
            loss_dict = self.model(images, targets)
            
            # If class weights are provided, apply to classification loss
            if class_weights is not None:
                if self.model_type in ['fasterrcnn_resnet', 'fasterrcnn_mobilenet', 'fasterrcnn_resnet50_fpn_v2', 
                                     'fasterrcnn_mobilenet_320', 'fasterrcnn_mobilenet_320_v2'] and 'loss_classifier' in loss_dict:
                    weighted_cls_loss = 0.0
                    total_weight = 0.0
                    
                    # Iterate through all target labels and apply weights
                    for target in targets:
                        if len(target['labels']) > 0:
                            # Get class indices for targets (subtract 1 because FasterRCNN labels start from 1, while weights start from 0)
                            # Note: Background class (0) is specially handled in FasterRCNN, we only focus on foreground classes
                            cls_indices = target['labels'] - 1
                            
                            # Filter valid foreground classes (prevent index out of bounds)
                            valid_mask = (cls_indices >= 0) & (cls_indices < len(class_weights))
                            if valid_mask.sum() > 0:
                                valid_indices = cls_indices[valid_mask]
                                weight_sum = class_weights[valid_indices].sum().item()
                                total_weight += weight_sum
                    
                    # If there are valid weights, rescale classification loss
                    if total_weight > 0:
                        scale_factor = len(class_weights) / total_weight  # Normalization factor
                        loss_dict['loss_classifier'] *= scale_factor
                
                elif self.model_type in ['ssd', 'ssdlite_mobilenet_large'] and 'classification' in loss_dict:
                    # SSD class weighted loss handling
                    batch_size = len(targets)
                    all_cls_indices = []
                    
                    # Collect class indices for all targets
                    for target in targets:
                        if len(target['labels']) > 0:
                            # SSD labels start from 0, use directly
                            cls_indices = target['labels']  
                            all_cls_indices.append(cls_indices)
                    
                    if all_cls_indices:
                        # Merge class indices from all batches
                        all_cls_indices = torch.cat(all_cls_indices)
                        
                        # Calculate weight sum for each class
                        valid_mask = (all_cls_indices >= 0) & (all_cls_indices < len(class_weights))
                        if valid_mask.sum() > 0:
                            valid_indices = all_cls_indices[valid_mask]
                            unique_classes, counts = torch.unique(valid_indices, return_counts=True)
                            
                            # Calculate class weight sum
                            class_weight_sum = sum([class_weights[cls.item()] * count.item() 
                                                   for cls, count in zip(unique_classes, counts)])
                            
                            # Apply weight scaling
                            if class_weight_sum > 0:
                                scale_factor = valid_mask.sum().float() / class_weight_sum
                                loss_dict['classification'] *= scale_factor
                
                elif self.model_type in ['retinanet', 'retinanet_v2'] and 'classification' in loss_dict:
                    # RetinaNet class weighted loss handling (similar to SSD)
                    batch_size = len(targets)
                    all_cls_indices = []
                    
                    # Collect class indices for all targets
                    for target in targets:
                        if len(target['labels']) > 0:
                            # RetinaNet labels start from 0 for foreground classes, we subtract 1 to correctly index weights
                            cls_indices = target['labels'] - 1  
                            all_cls_indices.append(cls_indices)
                    
                    if all_cls_indices:
                        # Merge class indices from all batches
                        all_cls_indices = torch.cat(all_cls_indices)
                        
                        # Calculate weight sum for each class
                        valid_mask = (all_cls_indices >= 0) & (all_cls_indices < len(class_weights))
                        if valid_mask.sum() > 0:
                            valid_indices = all_cls_indices[valid_mask]
                            unique_classes, counts = torch.unique(valid_indices, return_counts=True)
                            
                            # Calculate class weight sum
                            class_weight_sum = sum([class_weights[cls.item()] * count.item() 
                                                   for cls, count in zip(unique_classes, counts)])
                            
                            # Apply weight scaling
                            if class_weight_sum > 0:
                                scale_factor = valid_mask.sum().float() / class_weight_sum
                                loss_dict['classification'] *= scale_factor
            
            return loss_dict
        else:
            # Inference mode, directly return original model output
            return self.model(images)


def build_detection_model(num_classes, model_type='fasterrcnn_resnet', pretrained=True, **kwargs):
    """
    Helper function to build detection model
    
    Args:
        num_classes (int): Number of classes (including background class)
        model_type (str): Model type, options:
            - 'fasterrcnn_resnet' (ResNet50-FPN)
            - 'fasterrcnn_resnet50_fpn_v2' (ResNet50-FPN V2)
            - 'fasterrcnn_mobilenet' (MobileNetV3-Large-FPN)
            - 'fasterrcnn_mobilenet_320' (MobileNetV3-Large-320-FPN)
            - 'fasterrcnn_mobilenet_320_v2' (MobileNetV3-Large-320-FPN V2)
            - 'ssd' (SSD300-VGG16)
            - 'ssdlite_mobilenet_large' (SSDLite320-MobileNetV3-Large)
            - 'fcos_resnet' (FCOS ResNet50-FPN)
            - 'retinanet' (RetinaNet-ResNet50-FPN)
            - 'retinanet_v2' (RetinaNet-ResNet50-FPN V2)
        pretrained (bool): Whether to use pretrained weights
        **kwargs: Other parameters
        
    Returns:
        DetectionModel: Detection model instance
    """
    return DetectionModel(num_classes, model_type, pretrained, **kwargs)


def convert_yolo_to_rcnn_target(yolo_targets, image_size):
    """
    Convert YOLO format targets to Faster R-CNN format
    
    Args:
        yolo_targets (Tensor): YOLO format targets [N, 5] (class_id, x_center, y_center, width, height)
        image_size (tuple): Image size (height, width)
    
    Returns:
        Dict[str, Tensor]: Faster R-CNN format targets
    """
    height, width = image_size
    boxes = []
    labels = []
    
    for target in yolo_targets:
        if target.sum() == 0:  # Skip padded targets
            continue
            
        class_id, x_center, y_center, w, h = target.tolist()
        
        # Convert center point, width/height format to top-left, bottom-right coordinates
        x1 = (x_center - w / 2) * width
        y1 = (y_center - h / 2) * height
        x2 = (x_center + w / 2) * width
        y2 = (y_center + h / 2) * height
        
        boxes.append([x1, y1, x2, y2])
        labels.append(int(class_id) + 1)  # +1 because torchvision models use 0 as background class
    
    if len(boxes) > 0:
        return {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([0]),  # Virtual image ID
            'area': (torch.tensor(boxes)[:, 2] - torch.tensor(boxes)[:, 0]) * 
                    (torch.tensor(boxes)[:, 3] - torch.tensor(boxes)[:, 1]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.uint8)
        }
    else:
        # If no targets, return empty target dictionary
        return {
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.zeros((0,), dtype=torch.int64),
            'image_id': torch.tensor([0]),
            'area': torch.zeros((0,), dtype=torch.float32),
            'iscrowd': torch.zeros((0,), dtype=torch.uint8)
        }


def convert_batch_yolo_to_rcnn(images, yolo_batch_targets):
    """
    Convert batch YOLO targets to Faster R-CNN format
    
    Args:
        images (Tensor): Image batch [B, C, H, W]
        yolo_batch_targets (Tensor): YOLO format target batch [B, max_objects, 5]
        
    Returns:
        List[Dict[str, Tensor]]: List of Faster R-CNN format targets
    """
    rcnn_targets = []
    batch_size = images.shape[0]
    height, width = images.shape[2:]
    
    for i in range(batch_size):
        rcnn_target = convert_yolo_to_rcnn_target(yolo_batch_targets[i], (height, width))
        rcnn_targets.append(rcnn_target)
    
    return rcnn_targets


def convert_rcnn_to_yolo_pred(rcnn_predictions, image_size):
    """
    Convert Faster R-CNN prediction results to YOLO format
    
    Args:
        rcnn_predictions (List[Dict]): Faster R-CNN prediction results
        image_size (tuple): Image size (height, width)
    
    Returns:
        List[Tensor]: YOLO format prediction results [N, 6] (x_center, y_center, width, height, confidence, class_id)
    """
    yolo_predictions = []
    height, width = image_size
    
    for pred in rcnn_predictions:
        boxes = pred['boxes'].cpu()
        scores = pred['scores'].cpu()
        labels = pred['labels'].cpu()
        
        # Check if there are detection results
        if len(boxes) == 0:
            continue
            
        yolo_pred = torch.zeros((len(boxes), 6))
        
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            x1, y1, x2, y2 = box.tolist()
            
            # Convert to YOLO format
            x_center = (x1 + x2) / 2 / width
            y_center = (y1 + y2) / 2 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            
            yolo_pred[i] = torch.tensor([x_center, y_center, w, h, score, label - 1])  # -1 because YOLO doesn't use background class
        
        yolo_predictions.append(yolo_pred)
    
    return yolo_predictions