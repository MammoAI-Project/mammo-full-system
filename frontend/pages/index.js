import { useState, useEffect } from 'react';
import { Box, Grid, Typography, Paper, Alert } from '@mui/material';
import Layout from '../components/Layout';
import SidebarConfig from '../components/SidebarConfig';
import ImageUploader from '../components/ImageUploader';
import ResultCard from '../components/ResultCard';
import SummaryStatistics from '../components/SummaryStatistics';
import OverallAssessment from '../components/OverallAssessment';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Home() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [threshold, setThreshold] = useState(0.7);
  const [showDescriptions, setShowDescriptions] = useState(true);
  const [showRecommendations, setShowRecommendations] = useState(true);
  const [biradsDescriptions, setBiradsDescriptions] = useState({});
  const [error, setError] = useState(null);
  const [timestamp, setTimestamp] = useState('');

  useEffect(() => {
    const now = new Date().toLocaleString();
    setTimestamp(now);
  }, []);
  
  useEffect(() => {
    fetch(`${API_URL}/api/birads-descriptions`)
      .then(res => res.json())
      .then(data => setBiradsDescriptions(data))
      .catch(err => console.error('Error fetching BIRADS descriptions:', err));
  }, []);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
      if (!validTypes.includes(selectedFile.type)) {
        setError('Invalid file format. Please upload a JPG, JPEG, or PNG image.');
        setFile(null);
        setPreview(null);
        return;
      }
      
      setFile(selectedFile);
      setError(null);
      
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(selectedFile);
    }
  };

  const handleAnalyze = async () => {
    if (!file) {
      setError('Please upload an image first.');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('threshold', threshold);
      
      const response = await fetch(`${API_URL}/api/analyze`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to analyze image');
      }
      
      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Layout>
      {/* Header */}
      <Box sx={{ my: 4, textAlign: 'center', bgcolor: '#f8f9fa', borderRadius: 2, p: 3, width: '100%' }}>
        <Typography variant="h3" component="h1" gutterBottom sx={{ color: '#2c3e50' }}>
          MammoAI
        </Typography>
        <Typography variant="h5" component="h2" gutterBottom sx={{ color: '#2c3e50' }}>
          Breast Cancer Analysis Tool
        </Typography>
        <Typography variant="body1" sx={{ color: '#34495e', fontSize: '1.1rem' }}>
          This application analyzes mammogram images using AI to assist in breast cancer detection and risk assessment.
          The tool evaluates images across multiple diagnostic branches including Risk Assessment, Cancer Detection, 
          Cancer Staging, Risk Factor Analysis, and Differential Diagnosis.
        </Typography>
      </Box>

      <Alert severity="info" sx={{ mb: 4, width: '100%' }}>
        👉 Upload a mammogram image below to begin analysis.
      </Alert>
      
      {/* Main Content with Sidebar Layout */}
      <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 4 }}>
        {/* Left Fixed Sidebar */}
        <Box 
          sx={{ 
            width: { xs: '100%', md: '280px' },
            flexShrink: 0,
            position: { xs: 'static', md: 'sticky' },
            top: { xs: 0, md: '20px' },
            alignSelf: 'flex-start',
            height: 'fit-content'
          }}
        >
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>Configuration</Typography>
            <SidebarConfig 
              threshold={threshold}
              setThreshold={setThreshold}
              showDescriptions={showDescriptions}
              setShowDescriptions={setShowDescriptions}
              showRecommendations={showRecommendations}
              setShowRecommendations={setShowRecommendations}
            />
          </Paper>
        </Box>
        
        {/* Main Content Area */}
        <Box sx={{ flexGrow: 1 }}>
          {/* File Upload Area */}
          <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
            <Typography variant="h6" gutterBottom>Upload Mammogram</Typography>
            <ImageUploader 
              onFileChange={handleFileChange}
              onAnalyze={handleAnalyze}
              file={file}
              error={error}
              loading={loading}
            />
          </Paper>
          
          {/* Image Preview and Results */}
          {preview && (
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Paper elevation={2} sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom>Uploaded Image</Typography>
                  <Box sx={{ position: 'relative', width: '100%', height: 300 }}>
                    <img 
                      src={preview} 
                      alt="Preview" 
                      style={{ 
                        maxWidth: '100%', 
                        maxHeight: '100%',
                        objectFit: 'contain',
                        display: 'block',
                        margin: '0 auto'
                      }} 
                    />
                  </Box>
                </Paper>
              </Grid>
              
              {results && (
                <Grid item xs={12} md={6}>
                  <Paper elevation={2} sx={{ p: 2 }}>
                    <Typography variant="h6" gutterBottom>Region of Interest</Typography>
                    <Box sx={{ position: 'relative', width: '100%', height: 300 }}>
                      <img
                        src={`data:image/jpeg;base64,${results.gradcam_image}`}
                        alt="Heatmap"
                        style={{ 
                          maxWidth: '100%', 
                          maxHeight: '100%',
                          objectFit: 'contain',
                          display: 'block',
                          margin: '0 auto'
                        }}
                      />
                    </Box>
                    <Typography variant="caption" sx={{ mt: 1, display: 'block', fontStyle: 'italic' }}>
                      Areas highlighted in red have the strongest influence on the model's predictions
                    </Typography>
                  </Paper>
                </Grid>
              )}
            </Grid>
          )}
          
          {/* Analysis Results */}
          {results && (
            <Box sx={{ mt: 4 }}>
              <Typography variant="h5" gutterBottom>Analysis Results</Typography>
              
              <Grid container spacing={3}>
                {Object.entries(results.results).map(([branch, data]) => (
                  <Grid item xs={12} md={6} key={branch}>
                    <ResultCard 
                      branch={branch} 
                      data={data} 
                      showDescriptions={showDescriptions}
                      showRecommendations={showRecommendations}
                    />
                  </Grid>
                ))}
              </Grid>
              
              {/* Summary Statistics */}
              <Box sx={{ mt: 4 }}>
                <Typography variant="h5" gutterBottom>Summary Statistics</Typography>
                <SummaryStatistics 
                  summary={results.summary}
                  totalBranches={Object.keys(results.results).length}
                />
              </Box>
              
              {/* Overall Assessment */}
              <Box sx={{ mt: 4 }}>
                <Typography variant="h5" gutterBottom>Overall Assessment</Typography>
                <OverallAssessment 
                  needsReview={results.needs_review}
                  avgConfidence={results.avg_confidence}
                />
              </Box>
            </Box>
          )}
        </Box>
      </Box>
      
      {/* Footer */}
      <Box 
        component="footer" 
        sx={{ 
          mt: 6, 
          mb: 3, 
          p: 3, 
          bgcolor: '#f8f9fa', 
          borderRadius: 2,
          borderTop: '3px solid #e9ecef',
          textAlign: 'center'
        }}
      >
        <Typography variant="caption" color="text.secondary">
          This tool is generated by an AI system and should be reviewed by a qualified healthcare professional 
          before making any medical decisions. AI analysis is meant to assist, not replace, professional medical judgment.
        </Typography>
        <Typography variant="caption" display="block" color="text.secondary" sx={{ mt: 1 }}>
          © Medical Imaging AI Tools - Generated on {timestamp}
        </Typography>
      </Box>
    </Layout>
  );
}