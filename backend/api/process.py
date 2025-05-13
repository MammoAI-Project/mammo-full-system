from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import tensorflow as tf
import io
import base64
import uuid
from PIL import Image
import cv2
from .enhancement import apply_clahe_enhancement
from .gradcam import apply_gradcam
from .uncertainty import uncertainty_quantification
from .schema import AnalysisResponse, BranchResult


# Define BIRADS descriptions
BIRADS_DESCRIPTIONS = {
    'CL1': 'Normal mammogram, no significant abnormalities',
    'CL3': 'Probably benign finding, short-term follow-up suggested',
    'CL4': 'Suspicious abnormality, biopsy should be considered',
    'CL5': 'Highly suspicious of malignancy, appropriate action should be taken'
}

def process_image(model, image_data, threshold=0.7):
    try:
        img = Image.open(io.BytesIO(image_data))
        img = img.convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0

        enhanced_img = apply_clahe_enhancement(np.array([img_array]))[0]

        
        if model == "mock_model":
            # Create mock predictions for demonstration
            branches = ['Risk Assessment', 'Cancer Detection', 'Cancer Staging',
                        'Risk Factor Analysis', 'Differential Diagnosis']
            birads_classes = ['CL1', 'CL3', 'CL4', 'CL5']
            
            results = {}
            for branch in branches:
                pred_class = np.random.randint(0, 4)  
                pred_prob = np.random.uniform(0.6, 0.95) 
                uncertainty = np.random.uniform(0.01, 0.2) 
                
                results[branch] = {
                    'prediction': birads_classes[pred_class],
                    'confidence': float(pred_prob),
                    'uncertainty': float(uncertainty),
                    'needs_review': pred_prob < threshold,
                    'description': BIRADS_DESCRIPTIONS[birads_classes[pred_class]]
                }
            
            # Create a mock gradcam image
            _, buffer = cv2.imencode('.jpg', (img_array * 255).astype(np.uint8))
            gradcam_img = base64.b64encode(buffer).decode('utf-8')
            
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
                    'needs_review': needs_review,
                    'description': BIRADS_DESCRIPTIONS[birads_classes[pred_class]]
                }

            gradcam_img = apply_gradcam(model, enhanced_img, 'hybrid_backbone')

        # Calculate summary statistics
        birads_counts = {}
        for birads in BIRADS_DESCRIPTIONS.keys():
            count = sum(1 for branch, data in results.items() if data['prediction'] == birads)
            birads_counts[birads] = count
            
        # Determine if any branch needs review
        needs_review = any(data['needs_review'] for data in results.values())
        # Calculate average confidence
        avg_confidence = sum(data['confidence'] for data in results.values()) / len(results)
            
        return AnalysisResponse(
            analysis_id=str(uuid.uuid4()),
            results=results,
            gradcam_image=gradcam_img,
            summary=birads_counts,
            avg_confidence=float(avg_confidence),
            needs_review=needs_review
        )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
