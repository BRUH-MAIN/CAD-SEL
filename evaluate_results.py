import os
import glob
import pandas as pd
import re

def extract_metrics_from_file(file_path):
    """Extract metrics from evaluation_results.txt file"""
    metrics = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Extract model name and dataset name
        folder_name = os.path.dirname(file_path)
        # New regex pattern to match evaluation_results_<model_name>_<DATASET>_<DATA>
        folder_pattern = r'evaluation_results_(.+?)_(Internal|External|Full)_(WL|EUS)$'
        folder_match = re.search(folder_pattern, folder_name)
        
        if folder_match:
            model_name = folder_match.group(1)
            dataset_type = folder_match.group(2)  # Internal or External or Full
            data_type = folder_match.group(3)     # WL or EUS
        else:
            print(f"Warning: Unable to parse information from folder name: {folder_name}")
            return None
        
        metrics['model'] = model_name
        metrics['dataset'] = dataset_type
        metrics['data_type'] = data_type
        
        # Extract main metrics
        for i, line in enumerate(lines):
            try:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                if line.startswith('mAP@0.5:'):
                    value = line.split(': ')[1]
                    metrics['mAP@0.5'] = float(value)
                elif line.startswith('mAP@0.75:'):
                    value = line.split(': ')[1]
                    metrics['mAP@0.75'] = float(value)
                elif line.startswith('Overall mIoU:'):
                    value = line.split(': ')[1]
                    metrics['Overall_mIoU'] = float(value)
                elif line.startswith('Overall Precision/Recall/F1@0.5:'):
                    if i + 3 < len(lines):
                        metrics['Overall_Precision@0.5'] = float(lines[i+1].strip().split(': ')[1])
                        metrics['Overall_Recall@0.5'] = float(lines[i+2].strip().split(': ')[1])
                        metrics['Overall_F1@0.5'] = float(lines[i+3].strip().split(': ')[1])
                elif line.startswith('Overall Precision/Recall/F1@0.75:'):
                    if i + 3 < len(lines):
                        metrics['Overall_Precision@0.75'] = float(lines[i+1].strip().split(': ')[1])
                        metrics['Overall_Recall@0.75'] = float(lines[i+2].strip().split(': ')[1])
                        metrics['Overall_F1@0.75'] = float(lines[i+3].strip().split(': ')[1])
                        
                # Extract AP for each class
                elif line == 'AP per class@0.5:':
                    j = i + 1
                    while j < len(lines) and lines[j].strip() and not lines[j].startswith('AP per class@0.75'):
                        class_line = lines[j].strip()
                        if ': ' in class_line:
                            class_name, ap = class_line.split(': ')
                            try:
                                metrics[f'AP@0.5_{class_name.strip()}'] = float(ap)
                            except ValueError:
                                print(f"Warning: Unable to parse AP value: {ap} in file {file_path}")
                        j += 1
                        
                elif line == 'AP per class@0.75:':
                    j = i + 1
                    while j < len(lines) and lines[j].strip() and not lines[j].startswith('IoU per class'):
                        class_line = lines[j].strip()
                        if ': ' in class_line:
                            class_name, ap = class_line.split(': ')
                            try:
                                metrics[f'AP@0.75_{class_name.strip()}'] = float(ap)
                            except ValueError:
                                print(f"Warning: Unable to parse AP value: {ap} in file {file_path}")
                        j += 1
                        
            except Exception as e:
                print(f"Warning: Error processing line '{line}': {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error: Error processing file {file_path}: {str(e)}")
        return None
        
    # Verify that all necessary metrics exist
    required_metrics = ['mAP@0.5', 'mAP@0.75', 'Overall_mIoU']
    for metric in required_metrics:
        if metric not in metrics:
            print(f"Warning: File {file_path} is missing necessary metric {metric}")
            return None
            
    return metrics

# Get all evaluation_results folders
result_folders = glob.glob('evaluation_results_*')

# Collect all metrics
all_metrics = []
for folder in result_folders:
    result_file = os.path.join(folder, 'evaluation_results.txt')
    if os.path.exists(result_file):
        metrics = extract_metrics_from_file(result_file)
        if metrics:  # Only add successfully parsed results
            all_metrics.append(metrics)
            print(f"Successfully parsed file: {result_file}")
        else:
            print(f"Unable to parse file: {result_file}")

# If no results were successfully parsed, exit early
if not all_metrics:
    print("Error: No parseable result files found")
    exit(1)

print(f"Number of successfully parsed results: {len(all_metrics)}")

# Define fixed order for models
MODEL_ORDER = [
    "fasterrcnn_resnet",
    "fasterrcnn_resnet50_fpn_v2",
    "fasterrcnn_mobilenet",
    "fasterrcnn_mobilenet_320",
    "fasterrcnn_mobilenet_320_v2",
    "fcos_resnet",
    "ssd",
    "retinanet",
    "retinanet_v2"
]

# Create DataFrame
df = pd.DataFrame(all_metrics)

# Reorder columns
columns_order = ['model', 'dataset', 'data_type', 'mAP@0.5', 'mAP@0.75', 'Overall_mIoU',
                'Overall_Precision@0.5', 'Overall_Recall@0.5', 'Overall_F1@0.5',
                'Overall_Precision@0.75', 'Overall_Recall@0.75', 'Overall_F1@0.75']

# Add class-specific metrics
class_specific_columns = [col for col in df.columns if col.startswith('AP@')]
columns_order.extend(sorted(class_specific_columns))

# Create model order mapping and add model_order column
df['model_order'] = pd.Categorical(df['model'], categories=MODEL_ORDER, ordered=True).codes

# Create data type order mapping
data_type_order = ['WL', 'EUS']
df['data_type'] = pd.Categorical(df['data_type'], categories=data_type_order, ordered=True)

# Create dataset type order mapping
dataset_order = ['Full']
df['dataset'] = pd.Categorical(df['dataset'], categories=dataset_order, ordered=True)

# Sort by data type, dataset type, and predefined model order
df = df.sort_values(['data_type', 'dataset', 'model_order'])

# Remove auxiliary columns
df = df.drop('model_order', axis=1)

# Reorder columns and reset index
df = df[columns_order].reset_index()
df = df.rename(columns={'index': 'ID'})

# Save as CSV
df.to_csv('evaluation_results_summary.csv', index=False)
print("Results saved to evaluation_results_summary.csv")
print("\nData preview:")
print(df.head())

# Print number of results for each combination
print("\nResult statistics:")
print(df.groupby(['data_type', 'dataset', 'model']).size().unstack(fill_value=0))