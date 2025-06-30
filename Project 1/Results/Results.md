
## Findings and Results

Found 24991 files belonging to 2 classes.

Using 19993 files for training.
Using 4998 files for validation.

TensorFlow found 24,991 images split into 2 classes (cats and dogs).

80% (19,993 images) are used for training.
20% (4,998 images) are used for validation.

Corrupt JPEG data: ### extraneous bytes before marker 0xd9
Warning: unknown JFIF revision number 0.00

These are non-critical warnings.
They just mean some images have extra or unusual bytes but are still readable thanks to ImageFile.
LOAD_TRUNCATED_IMAGES = True.

Training continues normally.

# Training Progress

# Epoch 1/3

Training started and progressed over 625 steps (batches).

Initial accuracy: ~52% (model is just beginning to learn)

Final accuracy for Epoch 1: 0.5859

Loss: 0.6877

Validation Accuracy: 0.7635

Validation Loss: 0.4907

The model performs better on validation than training(not overfitting yet).

# Epoch 2/3

Training Accuracy: 0.7583

Training Loss: 0.4941

Validation Accuracy: 0.8039

Validation Loss: 0.4233

The model improved significantly. Validation accuracy is over 80%.

# Epoch 3/3

Training Accuracy: 0.8079

Training Loss: 0.4088

Validation Accuracy: 0.8171

Validation Loss: 0.4070

Slight further improvement. Both training and validation accuracy are over 80%. Loss is also lower.






# Sample Prediction Results

### 🔗 Source:  
![Prediction Results](https://github.com/Shishirise/ML-with-Python/blob/main/Project%201/Results/image.jpg?raw=true)  


---

##  Evaluation of Model Predictions

Each image shows:
- **True label**: the actual category (Cat or Dog)
- **Predicted label**: the model’s output

### ✅ Correct Predictions (7 out of 9)
- Correctly predicted **7 images** using the trained model.
- The model mostly succeeded at distinguishing between cats and dogs, especially when the images were sharp and clear.

### ❌ Misclassifications (2 out of 9)
- **True: Cat → Pred: Dog**  
- These were likely misclassified due to:
  - Blurry or dark lighting
  - Pet's pose or surroundings causing confusion
  - Limited training data for similar-looking cats

---

###  Final Thoughts

- **Sample prediction accuracy**: ~77.8%, which closely matches the validation accuracy of ~81%.
- With further training or improvements like data augmentation, I believe the model could reduce such misclassifications.


