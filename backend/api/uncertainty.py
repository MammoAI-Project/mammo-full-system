import numpy as np
import tensorflow as tf


def uncertainty_quantification(model, img, num_samples=10):
    model.trainable = True

    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)

    all_predictions = []
    for _ in range(num_samples):
        predictions = model.predict(img)
        all_predictions.append([p[0] for p in predictions])

    mean_preds = []
    std_preds = []

    for i in range(len(all_predictions[0])):
        branch_preds = np.array([pred[i] for pred in all_predictions])
        mean_preds.append(np.mean(branch_preds, axis=0))
        std_preds.append(np.std(branch_preds, axis=0))

    model.trainable = False

    return mean_preds, std_preds