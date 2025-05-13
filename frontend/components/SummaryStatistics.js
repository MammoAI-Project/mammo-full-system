import { Grid, Paper, Typography } from '@mui/material';

export default function SummaryStatistics({ summary, totalBranches }) {
  const getBiradsColor = (birads) => {
    switch (birads) {
      case 'CL1': return '#4CAF50';
      case 'CL3': return '#FF9800';
      case 'CL4': return '#F44336';
      case 'CL5': return '#D32F2F';
      default: return '#757575';
    }
  };
  
  return (
    <Grid container spacing={2}>
      {Object.entries(summary).map(([birads, count]) => (
        <Grid item xs={6} sm={3} key={birads}>
          <Paper 
            elevation={1} 
            sx={{ 
              p: 2, 
              textAlign: 'center',
              borderLeft: `5px solid ${getBiradsColor(birads)}`,
              bgcolor: getBiradsColor(birads) + '10'
            }}
          >
            <Typography variant="h5" sx={{ color: getBiradsColor(birads) }}>
              {birads}
            </Typography>
            <Typography variant="h4">
              {count}/{totalBranches}
            </Typography>
            <Typography variant="caption">branches</Typography>
          </Paper>
        </Grid>
      ))}
    </Grid>
  );
}

