import { useState } from 'react';
import {
  Card, CardContent, CardHeader, Table, TableBody,
  TableCell, TableRow, Typography, Box, Chip, Alert
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import InfoIcon from '@mui/icons-material/Info';
import WarningIcon from '@mui/icons-material/Warning';

export default function ResultCard({ branch, data, showDescriptions, showRecommendations }) {
  const getCardColor = (data) => {
    if (data.needs_review) return '#F44336';
    if (data.confidence > 0.8) return '#4CAF50';
    return '#FF9800';
  };
  
  const getBiradsColor = (birads) => {
    switch (birads) {
      case 'CL1': return '#4CAF50';
      case 'CL3': return '#FF9800';
      case 'CL4': return '#F44336';
      case 'CL5': return '#D32F2F';
      default: return '#757575';
    }
  };
  
  const getRecommendationByBirads = (birads) => {
    switch (birads) {
      case 'CL1':
        return { text: 'Routine screening as per guidelines', severity: 'success', icon: <CheckCircleIcon /> };
      case 'CL3':
        return { text: 'Follow-up examination in 6 months', severity: 'info', icon: <InfoIcon /> };
      case 'CL4':
        return { text: 'Biopsy should be considered', severity: 'warning', icon: <WarningIcon /> };
      case 'CL5':
        return { text: 'Immediate biopsy and consultation', severity: 'error', icon: <ErrorIcon /> };
      default:
        return { text: 'Consult healthcare professional', severity: 'info', icon: <InfoIcon /> };
    }
  };

  return (
    <Card 
      elevation={3} 
      sx={{ 
        borderLeft: `5px solid ${getCardColor(data)}`,
        height: '100%'
      }}
    >
      <CardHeader title={branch} />
      <CardContent>
        <Table size="small">
          <TableBody>
            <TableRow>
              <TableCell component="th" scope="row">Prediction:</TableCell>
              <TableCell>
                <Chip 
                  label={data.prediction} 
                  sx={{ 
                    bgcolor: getBiradsColor(data.prediction) + '20',
                    color: getBiradsColor(data.prediction),
                    fontWeight: 'bold'
                  }} 
                />
              </TableCell>
            </TableRow>
            
            {showDescriptions && (
              <TableRow>
                <TableCell component="th" scope="row">Description:</TableCell>
                <TableCell>{data.description}</TableCell>
              </TableRow>
            )}
            
            <TableRow>
              <TableCell component="th" scope="row">Confidence:</TableCell>
              <TableCell>{(data.confidence * 100).toFixed(1)}%</TableCell>
            </TableRow>
            
            <TableRow>
              <TableCell component="th" scope="row">Uncertainty:</TableCell>
              <TableCell>{data.uncertainty.toFixed(3)}</TableCell>
            </TableRow>
            
            <TableRow>
              <TableCell component="th" scope="row">Status:</TableCell>
              <TableCell>
                <Typography 
                  color={data.needs_review ? 'error' : 'success'}
                  fontWeight="medium"
                >
                  {data.needs_review ? 'Needs Review' : 'Confident Prediction'}
                </Typography>
              </TableCell>
            </TableRow>
          </TableBody>
        </Table>
        
        {showRecommendations && (
          <Box sx={{ mt: 2 }}>
            <Alert 
              severity={getRecommendationByBirads(data.prediction).severity}
              icon={getRecommendationByBirads(data.prediction).icon}
            >
              <strong>Recommendation:</strong> {getRecommendationByBirads(data.prediction).text}
            </Alert>
          </Box>
        )}
      </CardContent>
    </Card>
  );
}
