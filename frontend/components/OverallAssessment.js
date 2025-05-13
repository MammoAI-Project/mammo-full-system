import { Box, Alert, Typography } from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import WarningIcon from '@mui/icons-material/Warning';

export default function OverallAssessment({ needsReview, avgConfidence }) {
  return (
    <Box>
      {needsReview ? (
        <Alert severity="error" icon={<ErrorIcon />}>
          <Typography variant="subtitle1">
            ⚠️ This case needs review by a healthcare professional
          </Typography>
        </Alert>
      ) : avgConfidence > 0.8 ? (
        <Alert severity="success" icon={<CheckCircleIcon />}>
          <Typography variant="subtitle1">
            ✅ High confidence predictions across all branches
          </Typography>
        </Alert>
      ) : (
        <Alert severity="warning" icon={<WarningIcon />}>
          <Typography variant="subtitle1">
            ⚠️ Moderate confidence predictions, consider secondary review
          </Typography>
        </Alert>
      )}
    </Box>
  );
}

