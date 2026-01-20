import { useState, useEffect } from 'react';
import StudentForm from './components/StudentForm';
import PredictionResult from './components/PredictionResult';
import BatchUpload from './components/BatchUpload';
import { predictDropout } from './services/api';

function App() {
  const [result, setResult] = useState(null);
  const [studentData, setStudentData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('single');
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('theme') || 'light';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };

  const handleSubmit = async (formData) => {
    setIsLoading(true);
    setError(null);
    setResult(null);
    setStudentData(formData);

    try {
      const prediction = await predictDropout(formData);
      setResult(prediction);
    } catch (err) {
      console.error('Prediction error:', err);
      if (err.response?.data?.detail) {
        setError(err.response.data.detail);
      } else if (err.message) {
        setError(err.message);
      } else {
        setError('Failed to get prediction. Please ensure the API server is running.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container">
      <div className="header">
        <div className="header-top">
          <div className="theme-toggle" onClick={toggleTheme}>
            <span className="theme-icon">{theme === 'light' ? '☀️' : '🌙'}</span>
            <span className="theme-toggle-label">{theme === 'light' ? 'Light' : 'Dark'}</span>
            <div className="theme-toggle-switch"></div>
          </div>
        </div>
        <h1>Student Dropout Prediction</h1>
        <p className="subtitle">AI-powered early warning system for student success</p>
      </div>

      <div className="tab-navigation">
        <button
          className={`tab-button ${activeTab === 'single' ? 'active' : ''}`}
          onClick={() => setActiveTab('single')}
        >
          <span className="tab-icon">👤</span>
          Single Student
        </button>
        <button
          className={`tab-button ${activeTab === 'batch' ? 'active' : ''}`}
          onClick={() => setActiveTab('batch')}
        >
          <span className="tab-icon">📤</span>
          Batch Upload
        </button>
      </div>

      {activeTab === 'single' ? (
        <>
          <StudentForm onSubmit={handleSubmit} isLoading={isLoading} />

          {isLoading && (
            <div className="card loading">
              <div className="loading-spinner"></div>
              <p>Analyzing student data...</p>
            </div>
          )}

          {error && (
            <div className="card error-message">
              <p>{error}</p>
            </div>
          )}

          {result && !isLoading && <PredictionResult result={result} studentData={studentData} />}
        </>
      ) : (
        <BatchUpload />
      )}
    </div>
  );
}

export default App;
