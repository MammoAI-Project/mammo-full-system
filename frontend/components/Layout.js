import Head from 'next/head';
import { Box, Container } from '@mui/material';

export default function Layout({ children, title = 'MammoAI - Breast Cancer Analysis Tool' }) {
  return (
    <>
      <Head>
        <title>{title}</title>
        <meta name="description" content="AI-powered breast cancer analysis from mammogram images" />
        <link rel="icon" href="/favicon.ico" />
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
        />
      </Head>
      <Container maxWidth="lg">
        {children}
      </Container>
    </>
  );
}


