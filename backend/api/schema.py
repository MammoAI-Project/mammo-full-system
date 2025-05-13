from pydantic import BaseModel
from typing import Dict, List, Optional

# Define response models
class BranchResult(BaseModel):
    prediction: str
    confidence: float
    uncertainty: float
    needs_review: bool
    description: str

class AnalysisResponse(BaseModel):
    analysis_id: str
    results: Dict[str, BranchResult]
    gradcam_image: str  
    summary: Dict[str, int]  
    avg_confidence: float
    needs_review: bool