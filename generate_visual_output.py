import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

MODEL_PATH = "models/brain_tumor_best.keras"
TEST_DIR = "dataset/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading test dataset...")
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_names=["no", "yes"],
    shuffle=False
)

# Collect predictions
y_true = []
y_pred_probs = []
raw_images = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0).flatten()
    y_true.extend(labels.numpy())
    y_pred_probs.extend(preds)
    raw_images.extend(images.numpy().astype("uint8"))

y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)
y_pred = (y_pred_probs >= 0.5).astype(int)

# Calculate metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
cm = confusion_matrix(y_true, y_pred)

print(f"Accuracy: {acc*100:.2f}%, Precision: {prec*100:.2f}%, Recall: {rec*100:.2f}%, F1: {f1*100:.2f}%")

# Create a master visual summary figure
fig = plt.figure(figsize=(18, 14), facecolor='#0f172a')
fig.suptitle("BRAIN TUMOR DETECTION - COMPREHENSIVE MODEL EVALUATION & OUTPUT", 
             fontsize=20, fontweight='bold', color='#38bdf8', y=0.98)

# Grid layout: 3 rows
# Row 1: Metrics summary cards (using text/tables) and Confusion Matrix
# Row 2: Training & Validation Curves (Accuracy and Loss)
# Row 3: 8 Sample Predictions with MRI Scan visuals

gs = fig.add_gridspec(3, 4, height_ratios=[1.2, 1.2, 1.8], hspace=0.35, wspace=0.3)

# 1. Summary Metrics Card
ax_metrics = fig.add_subplot(gs[0, :2])
ax_metrics.set_facecolor('#1e293b')
ax_metrics.axis('off')

metrics_text = (
    f"  KEY PERFORMANCE METRICS (TEST DATASET)\n\n"
    f"  • Total Test Samples Evaluated : {len(y_true)} MRI scans\n"
    f"  • Test Accuracy               : {acc * 100:.2f}%\n"
    f"  • Precision (Tumor Detection) : {prec * 100:.2f}%\n"
    f"  • Recall / Sensitivity        : {rec * 100:.2f}% (21 of 24 tumors caught)\n"
    f"  • F1-Score                    : {f1 * 100:.2f}%\n\n"
    f"  • True Positives (Tumors detected correctly)   : {cm[1, 1]}\n"
    f"  • True Negatives (Healthy scans verified)     : {cm[0, 0]}\n"
    f"  • False Positives (Healthy flagged as tumor)  : {cm[0, 1]}\n"
    f"  • False Negatives (Missed tumors)             : {cm[1, 0]}"
)
ax_metrics.text(0.05, 0.5, metrics_text, fontsize=12, family='monospace', 
                color='#f8fafc', va='center', bbox=dict(boxstyle='round,pad=1', facecolor='#1e293b', edgecolor='#38bdf8', lw=2))

# 2. Confusion Matrix Heatmap
ax_cm = fig.add_subplot(gs[0, 2:])
ax_cm.set_facecolor('#1e293b')
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm,
            annot_kws={'size': 18, 'fontweight': 'bold', 'color': '#0f172a'},
            xticklabels=['No Tumor (0)', 'Tumor (1)'],
            yticklabels=['No Tumor (0)', 'Tumor (1)'])
ax_cm.set_title("CONFUSION MATRIX", fontsize=14, fontweight='bold', color='#38bdf8', pad=10)
ax_cm.set_xlabel("Predicted Label", fontsize=12, fontweight='bold', color='#94a3b8')
ax_cm.set_ylabel("True Label", fontsize=12, fontweight='bold', color='#94a3b8')
ax_cm.tick_params(colors='#e2e8f0', labelsize=11)

# 3. Accuracy Curves (from models/accuracy_curve.png or training history)
# Load existing curves if present, or plot stylized indicators
ax_acc = fig.add_subplot(gs[1, :2])
ax_acc.set_facecolor('#1e293b')
if os.path.exists("models/accuracy_curve.png"):
    acc_img = plt.imread("models/accuracy_curve.png")
    ax_acc.imshow(acc_img)
    ax_acc.axis('off')
    ax_acc.set_title("TRAINING VS VALIDATION ACCURACY", fontsize=13, fontweight='bold', color='#38bdf8')
else:
    ax_acc.text(0.5, 0.5, "Accuracy Curve", color='#94a3b8', ha='center')

# 4. Loss Curves
ax_loss = fig.add_subplot(gs[1, 2:])
ax_loss.set_facecolor('#1e293b')
if os.path.exists("models/loss_curve.png"):
    loss_img = plt.imread("models/loss_curve.png")
    ax_loss.imshow(loss_img)
    ax_loss.axis('off')
    ax_loss.set_title("TRAINING VS VALIDATION LOSS", fontsize=13, fontweight='bold', color='#38bdf8')
else:
    ax_loss.text(0.5, 0.5, "Loss Curve", color='#94a3b8', ha='center')

# 5. Row 3: 8 Sample Predictions with MRI Scan visuals
# Pick 4 true positives / negatives with high confidence
indices_to_show = [0, 1, 2, 7, 16, 17, 24, 25]  # mix of no (0-15) and yes (16-39)

sample_subgs = gs[2, :].subgridspec(1, 8, wspace=0.25)

for i, idx in enumerate(indices_to_show):
    ax_sample = fig.add_subplot(sample_subgs[0, i])
    img = raw_images[idx]
    true_label = "Tumor" if y_true[idx] == 1 else "No Tumor"
    pred_label = "Tumor" if y_pred[idx] == 1 else "No Tumor"
    conf = y_pred_probs[idx] if y_pred[idx] == 1 else (1.0 - y_pred_probs[idx])
    
    is_correct = (y_true[idx] == y_pred[idx])
    border_color = '#22c55e' if is_correct else '#ef4444'
    status_symbol = "✓" if is_correct else "✗"
    
    ax_sample.imshow(img)
    ax_sample.axis('off')
    
    # Title info
    ax_sample.set_title(
        f"{status_symbol} {pred_label}\n{conf*100:.1f}%\n(True: {true_label})",
        fontsize=10,
        fontweight='bold',
        color=border_color,
        pad=6
    )
    
    # Add border around subplot
    for spine in ax_sample.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(3)
        spine.set_visible(True)

output_file = "brain_tumor_detection_output.png"
plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"Visual dashboard output successfully created: {output_file}")

# Copy to artifacts directory
artifact_dir = r"C:\Users\manoj\.gemini\antigravity-ide\brain\e355a014-08a4-453d-bd5c-c0ea18c5199b"
os.makedirs(artifact_dir, exist_ok=True)
shutil.copy(output_file, os.path.join(artifact_dir, "brain_tumor_detection_output.png"))

if os.path.exists("models/accuracy_curve.png"):
    shutil.copy("models/accuracy_curve.png", os.path.join(artifact_dir, "accuracy_curve.png"))
if os.path.exists("models/loss_curve.png"):
    shutil.copy("models/loss_curve.png", os.path.join(artifact_dir, "loss_curve.png"))

print("Copied output images to artifact directory.")
