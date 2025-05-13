import { useState } from 'react';
import { 
  Paper, Typography, Slider, Divider, 
  FormControlLabel, Switch, Box 
} from '@mui/material';

export default function SidebarConfig({ 
  threshold, setThreshold, 
  showDescriptions, setShowDescriptions,
  showRecommendations, setShowRecommendations 
}) {
  return (
    <>
      <Paper elevation={2} sx={{ p: 2, mb: 2 }}>
        <Typography variant="h6" gutterBottom>Configuration</Typography>
        <Typography id="threshold-slider" gutterBottom>
          Confidence Threshold: {threshold}
        </Typography>
        <Slider
          value={threshold}
          onChange={(e, newValue) => setThreshold(newValue)}
          min={0.5}
          max={0.95}
          step={0.01}
          aria-labelledby="threshold-slider"
        />
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          Predictions with confidence below this threshold will be marked for review
        </Typography>
        
        <Divider sx={{ my: 2 }} />
        
        <Typography variant="h6" gutterBottom>Display Options</Typography>
        <FormControlLabel
          control={
            <Switch
              checked={showDescriptions}
              onChange={(e) => setShowDescriptions(e.target.checked)}
            />
          }
          label="Show BI-RADS Descriptions"
        />
        <FormControlLabel
          control={
            <Switch
              checked={showRecommendations}
              onChange={(e) => setShowRecommendations(e.target.checked)}
            />
          }
          label="Show Recommendations"
        />
        
        <Divider sx={{ my: 2 }} />
        
        <Typography variant="h6" gutterBottom>BI-RADS Categories</Typography>
        <Box sx={{ mt: 1 }}>
          <Typography variant="body2"><strong>CL 1:</strong> Normal</Typography>
          <Typography variant="body2"><strong>CL 3:</strong> Probably benign</Typography>
          <Typography variant="body2"><strong>CL 4:</strong> Suspicious</Typography>
          <Typography variant="body2"><strong>CL 5:</strong> Highly suspicious</Typography>
        </Box>
      </Paper>
      
      <Paper elevation={2} sx={{ p: 2, mt: 2 }}>
        <Typography variant="caption" color="text.secondary">
          This tool is for medical professional use only. AI analysis is meant to assist, not replace, clinical judgment.
        </Typography>
      </Paper>
    </>
  );
}

