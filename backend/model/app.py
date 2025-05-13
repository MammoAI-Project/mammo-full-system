import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile
import os
import cv2
from pathlib import Path


# Load model
try:
    model = load_model("best_breast_cancer_model.h5")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None
    
def is_valid_image_file(file):
    if file is None:
        return False
    filename = file.name.lower()
    return filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png')


def apply_clahe_enhancement(images):
    enhanced_images = []

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for img in images:
        lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)

        lab_planes = list(cv2.split(lab))
        lab_planes[0] = clahe.apply(lab_planes[0])

        lab = cv2.merge(lab_planes)
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) / 255.0
        enhanced_images.append(enhanced)

    return np.array(enhanced_images)

def apply_gradcam(model, img_array, layer_name='hybrid_backbone', pred_index=None, branch='risk_output'):
    original_img = img_array.copy()

    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype(np.float32)
    branch_names = ['risk_output', 'detection_output', 'staging_output',
                    'factor_output', 'diagnosis_output']
    try:
        branch_index = branch_names.index(branch)
        print(f"Using branch: {branch} (index {branch_index})")
    except ValueError:
        print(f"Branch {branch} not found, using risk_output")
        branch_index = 0

    backbone = model.get_layer('hybrid_backbone')
    print(f"Backbone found: {backbone.name}, output shape: {backbone.output_shape}")

    if hasattr(backbone, 'layers'):
        for i, layer in enumerate(backbone.layers):
            if i < 5 or i > len(backbone.layers) - 5:
                print(f"Backbone layer {i}: {layer.name}, type: {type(layer).__name__}")
                if hasattr(layer, 'output_shape'):
                    print(f"   Shape: {layer.output_shape}")

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

    import cv2

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

    return superimposed

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

def process_new_image(model, img_path, threshold=0.7):
    try:
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0

        enhanced_img = apply_clahe_enhancement(np.array([img_array]))[0]

        # For demonstration, use mock data if model is not available
        if model == "mock_model":
            # Create mock predictions for demonstration
            branches = ['Risk Assessment', 'Cancer Detection', 'Cancer Staging',
                        'Risk Factor Analysis', 'Differential Diagnosis']
            birads_classes = ['CL1', 'CL3', 'CL4', 'CL5']
            
            results = {}
            for branch in branches:
                pred_class = np.random.randint(0, 4)  # Random class for demo
                pred_prob = np.random.uniform(0.6, 0.95)  # Random probability
                uncertainty = np.random.uniform(0.01, 0.2)  # Random uncertainty
                
                results[branch] = {
                    'prediction': birads_classes[pred_class],
                    'confidence': float(pred_prob),
                    'uncertainty': float(uncertainty),
                    'needs_review': pred_prob < threshold
                }
            
            # Create a mock gradcam image
            gradcam_img = np.array(img)  # Just use original image for demo
            
            return results, gradcam_img, True
            
        else:
            # Use the actual model
            mean_preds, std_preds = uncertainty_quantification(model, enhanced_img)

            branches = ['Risk Assessment', 'Cancer Detection', 'Cancer Staging',
                    'Risk Factor Analysis', 'Differential Diagnosis']
            birads_classes = ['CL1', 'CL3', 'CL4', 'CL5']

            results = {}
            for i, branch in enumerate(branches):
                pred_class = np.argmax(mean_preds[i])
                pred_prob = mean_preds[i][pred_class]
                uncertainty = std_preds[i][pred_class]

                needs_review = pred_prob < threshold

                results[branch] = {
                    'prediction': birads_classes[pred_class],
                    'confidence': float(pred_prob),
                    'uncertainty': float(uncertainty),
                    'needs_review': needs_review
                }

            gradcam_img = apply_gradcam(model, enhanced_img, 'hybrid_backbone')

            return results, gradcam_img, True
            
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None, False

# Define BIRADS descriptions for the UI
BIRADS_DESCRIPTIONS = {
    'CL1': 'Normal mammogram, no significant abnormalities',
    'CL3': 'Probably benign finding, short-term follow-up suggested',
    'CL4': 'Suspicious abnormality, biopsy should be considered',
    'CL5': 'Highly suspicious of malignancy, appropriate action should be taken'
}

# Streamlit app
def main():
    st.set_page_config(page_title="Breast Cancer Analysis Tool", 
                       page_icon="🏥", 
                       layout="wide")
    
    # Header with medical styling
    st.markdown("""
    <div style="padding: 20px; background-color: #f8f9fa; border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: #2c3e50; margin-bottom: 10px; text-align:center;">MammoAI</h1>
        <h2 style="color: #2c3e50; margin-bottom: 10px; text-align:center;">Breast Cancer analysis tool</h2>
        <p style="color: #34495e; font-size: 1.1em; ">
            This application analyzes mammogram images using AI to assist in breast cancer detection and risk assessment.
            The tool evaluates images across multiple diagnostic branches including Risk Assessment, Cancer Detection, 
            Cancer Staging, Risk Factor Analysis, and Differential Diagnosis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add quick instruction
    st.info("👉 Upload a mammogram image below to begin analysis.")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Confidence threshold slider
    threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.5, 
        max_value=0.95, 
        value=0.7,
        help="Predictions with confidence below this threshold will be marked for review"
    )
    
    # Add additional options to sidebar
    st.sidebar.subheader("Display Options")
    show_descriptions = st.sidebar.checkbox("Show BI-RADS Descriptions", value=True)
    show_recommendations = st.sidebar.checkbox("Show Recommendations", value=True)
    
    # Add sidebar info box about BI-RADS categories
    st.sidebar.subheader("BI-RADS Categories")
    st.sidebar.info("""
    **CL 1:** Normal
    
    **CL 1 3:** Probably benign
    
    **CL 4:** Suspicious
    
    **CL 5:** Highly suspicious
    """)
    
    # Add disclaimer
    st.sidebar.markdown("---")
    st.sidebar.caption("This tool is for medical professional use only. AI analysis is meant to assist, not replace, clinical judgment.")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a mammogram image", type=None, accept_multiple_files=False)
    
    if uploaded_file is not None:
        if is_valid_image_file(uploaded_file):
            # Display the uploaded image
            col1, col2 = st.columns(2)
        
            with col1:
                st.subheader("Uploaded Image")
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Mammogram", use_container_width=True)
                except Exception as e:
                    st.error(f"Error opening image: {e}")
        else:
            st.error("Invalid file format. Please upload a JPG, JPEG, or PNG image.")
        
        # Process button
        if st.button("Analyze Image"):
            with st.spinner("Processing image..."):
                try:
                    # Save the uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        tmp.write(uploaded_file.getvalue())
                        temp_file_path = tmp.name
                    
                    # Process the image 
                    results, gradcam_img, success = process_new_image(model, temp_file_path, threshold)
                    
                    # Remove temporary file
                    os.unlink(temp_file_path)
                    
                    if success:
                        with col2:
                            st.subheader("Region of Interest")
                            st.image(gradcam_img, caption="Heatmap indicates regions influencing the model's decision", use_container_width=True)
                            st.markdown("<p><em>Areas highlighted in red have the strongest influence on the model's predictions</em></p>", unsafe_allow_html=True)
                        
                        # Display results 
                        st.subheader("Analysis Results")
                        
                        # Create result cards in a grid
                        for i in range(0, len(results), 2):
                            cols = st.columns(2)
                            
                            # Process current card
                            branch = list(results.keys())[i]
                            data = results[branch]
                            
                            with cols[0]:
                                # Style card based on confidence
                                if data['needs_review']:
                                    card_color = "#F44336"  
                                elif data['confidence'] > 0.8:
                                    card_color = "#4CAF50"  
                                else:
                                    card_color = "#FF9800"  
                                
                                # Create card with HTML/CSS styling
                                st.markdown(f"""
                                <div style="border: 1px solid #ddd; border-radius: 5px; padding: 15px; 
                                     margin: 10px 0; border-left: 5px solid {card_color};">
                                    <h3 style="margin-top: 0;">{branch}</h3>
                                    <table style="width: 100%;">
                                        <tr>
                                            <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Prediction:</th>
                                            <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;"><strong>{data['prediction']}</strong></td>
                                        </tr>
                                        <tr>
                                            <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Description:</th>
                                            <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">{BIRADS_DESCRIPTIONS[data['prediction']]}</td>
                                        </tr>
                                        <tr>
                                            <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Confidence:</th>
                                            <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">{data['confidence']:.2%}</td>
                                        </tr>
                                        <tr>
                                            <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Uncertainty:</th>
                                            <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">{data['uncertainty']:.3f}</td>
                                        </tr>
                                        <tr>
                                            <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Status:</th>
                                            <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">
                                                {"<span style='color:#F44336;'>Needs Review</span>" if data['needs_review'] 
                                                else "<span style='color:#4CAF50;'>Confident Prediction</span>"}
                                            </td>
                                        </tr>
                                    </table>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Add recommendation based on BIRADS
                                if show_recommendations:
                                    if data['prediction'] == 'CL1':
                                        st.success("👍 **Recommendation:** Routine screening as per guidelines")
                                    elif data['prediction'] == 'CL3':
                                        st.info("📋 **Recommendation:** Follow-up examination in 6 months")
                                    elif data['prediction'] == 'CL4':
                                        st.warning("⚠️ **Recommendation:** Biopsy should be considered")
                                    elif data['prediction'] == 'CL5':
                                        st.error("🚨 **Recommendation:** Immediate biopsy and consultation")
                            
                            # Process second card if it exists
                            if i + 1 < len(results):
                                branch = list(results.keys())[i + 1]
                                data = results[branch]
                                
                                with cols[1]:
                                    # Style card based on confidence
                                    if data['needs_review']:
                                        card_color = "#F44336"  
                                    elif data['confidence'] > 0.8:
                                        card_color = "#4CAF50"  
                                    else:
                                        card_color = "#FF9800"  
                                    
                                    # Create card with HTML/CSS styling
                                    st.markdown(f"""
                                    <div style="border: 1px solid #ddd; border-radius: 5px; padding: 15px; 
                                         margin: 10px 0; border-left: 5px solid {card_color};">
                                        <h3 style="margin-top: 0;">{branch}</h3>
                                        <table style="width: 100%;">
                                            <tr>
                                                <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Prediction:</th>
                                                <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;"><strong>{data['prediction']}</strong></td>
                                            </tr>
                                            <tr>
                                                <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Description:</th>
                                                <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">{BIRADS_DESCRIPTIONS[data['prediction']]}</td>
                                            </tr>
                                            <tr>
                                                <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Confidence:</th>
                                                <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">{data['confidence']:.2%}</td>
                                            </tr>
                                            <tr>
                                                <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Uncertainty:</th>
                                                <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">{data['uncertainty']:.3f}</td>
                                            </tr>
                                            <tr>
                                                <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Status:</th>
                                                <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">
                                                    {"<span style='color:#F44336;'>Needs Review</span>" if data['needs_review'] 
                                                    else "<span style='color:#4CAF50;'>Confident Prediction</span>"}
                                                </td>
                                            </tr>
                                        </table>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Add recommendation based on BIRADS
                                    if show_recommendations:
                                        if data['prediction'] == 'CL1':
                                            st.success("👍 **Recommendation:** Routine screening as per guidelines")
                                        elif data['prediction'] == 'CL3':
                                            st.info("📋 **Recommendation:** Follow-up examination in 6 months")
                                        elif data['prediction'] == 'CL4':
                                            st.warning("⚠️ **Recommendation:** Biopsy should be considered")
                                        elif data['prediction'] == 'CL5':
                                            st.error("🚨 **Recommendation:** Immediate biopsy and consultation")
                        
                        # Summary statistics
                        st.subheader("Summary Statistics")
                        
                        # Count results by category
                        birads_counts = {}
                        for birads in BIRADS_DESCRIPTIONS.keys():
                            count = sum(1 for branch, data in results.items() if data['prediction'] == birads)
                            birads_counts[birads] = count
                        
                        # Display metrics
                        metric_cols = st.columns(len(birads_counts))
                        for i, (birads, count) in enumerate(birads_counts.items()):
                            with metric_cols[i]:
                                color = "#4CAF50" if birads == "CL1" else \
                                       "#FF9800" if birads == "CL3" else \
                                       "#F44336"
                                st.markdown(f"""
                                <div style="padding: 10px; border-radius: 5px; background-color: {color}20; 
                                     border-left: 5px solid {color}; text-align: center;">
                                    <h3 style="margin: 0; color: {color};">{birads}</h3>
                                    <h2 style="margin: 5px 0;">{count}/{len(results)}</h2>
                                    <p style="margin: 0; font-size: 0.8em;">branches</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Additional overall assessment
                        st.subheader("Overall Assessment")
                        
                        # Determine if any branch needs review
                        needs_review = any(data['needs_review'] for data in results.values())
                        # Calculate average confidence
                        avg_confidence = sum(data['confidence'] for data in results.values()) / len(results)
                        
                        # Display overall status
                        if needs_review:
                            st.error("⚠️ This case needs review by a healthcare professional")
                        elif avg_confidence > 0.8:
                            st.success("✅ High confidence predictions across all branches")
                        else:
                            st.warning("⚠️ Moderate confidence predictions, consider secondary review")
                    else:
                        st.error("Failed to process image. Please try again with a different image.")
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
    
    # Footer with disclaimer styling from the HTML report
    st.markdown("---")
    st.markdown("""
    <div style="margin-top: 30px; font-size: 0.8em; color: #666; text-align: center; background-color: #f8f9fa; 
         padding: 15px; border-radius: 5px; border-top: 3px solid #e9ecef;">
        <p>This tool is generated by an AI system and should be reviewed by a qualified healthcare professional 
        before making any medical decisions. AI analysis is meant to assist, not replace, professional medical judgment.</p>
        <p>© Medical Imaging AI Tools - Generated on {}</p>
    </div>
    """.format(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)

if __name__ == "__main__":
    main()