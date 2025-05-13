import numpy as np
import tensorflow as tf
import cv2
import base64


def apply_gradcam(model, img_array, layer_name='hybrid_backbone', pred_index=None, branch='risk_output'):
    original_img = img_array.copy()

    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype(np.float32)
    branch_names = ['risk_output', 'detection_output', 'staging_output',
                    'factor_output', 'diagnosis_output']
    try:
        branch_index = branch_names.index(branch)
    except ValueError:
        branch_index = 0

    backbone = model.get_layer('hybrid_backbone')
    target_layer_output = backbone.output

    input_tensor = tf.convert_to_tensor(img_array)

    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        outputs = model(input_tensor)
        branch_output = outputs[branch_index]

        if pred_index is None:
            pred_index = tf.argmax(branch_output[0])

        class_score = branch_output[:, pred_index]

    grads = tape.gradient(class_score, input_tensor)
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
    pooled_grads = tf.expand_dims(tf.expand_dims(pooled_grads, axis=1), axis=1)
    weighted_input = tf.multiply(input_tensor, pooled_grads)
    heatmap = tf.reduce_mean(weighted_input, axis=-1)
    heatmap = tf.maximum(heatmap, 0)[0]
    heatmap = heatmap / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
    heatmap = heatmap.numpy()

    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    img_for_overlay = (original_img * 255).astype(np.uint8)

    if len(img_for_overlay.shape) < 3:
        img_for_overlay = cv2.cvtColor(img_for_overlay, cv2.COLOR_GRAY2BGR)
    elif img_for_overlay.shape[2] == 1:
        img_for_overlay = cv2.cvtColor(img_for_overlay, cv2.COLOR_GRAY2BGR)
    elif img_for_overlay.shape[2] == 3:
        img_for_overlay = cv2.cvtColor(img_for_overlay, cv2.COLOR_RGB2BGR)

    superimposed = cv2.addWeighted(img_for_overlay, 0.6, heatmap_colored, 0.4, 0)
    superimposed = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
    
    # Convert to base64 string for API response
    _, buffer = cv2.imencode('.jpg', superimposed)
    return base64.b64encode(buffer).decode('utf-8')
