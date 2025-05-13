import { useState } from 'react';
import { 
  Paper, Button, Typography, 
  Alert, CircularProgress 
} from '@mui/material';

export default function ImageUploader({ 
  onFileChange, 
  onAnalyze, 
  file, 
  error, 
  loading 
}) {
  return (
    <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
      <input
        type="file"
        id="upload-input"
        accept="image/jpeg,image/jpg,image/png"
        onChange={onFileChange}
        style={{ display: 'none' }}
      />
      <label htmlFor="upload-input">
        <Button
          variant="contained"
          component="span"
          fullWidth
          sx={{ mb: 2 }}
        >
          Upload Mammogram Image
        </Button>
      </label>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      
      {file && (
        <Typography variant="body2" sx={{ mb: 2 }}>
          Selected file: {file.name}
        </Typography>
      )}
      
      <Button
        variant="contained"
        color="primary"
        onClick={onAnalyze}
        disabled={!file || loading}
        fullWidth
      >
        {loading ? <CircularProgress size={24} color="inherit" /> : 'Analyze Image'}
      </Button>
    </Paper>
  );
}