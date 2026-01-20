import { useState, useRef } from 'react';
import { predictBatch, downloadBatchReport } from '../services/api';

const REQUIRED_COLUMNS = [
  { name: 'student_id', type: 'string', description: 'Unique identifier' },
  { name: 'enrollment_date', type: 'date', description: 'YYYY-MM-DD format' },
  { name: 'program', type: 'string', description: 'Computer Science, Business, Engineering, Liberal Arts, or Nursing' },
  { name: 'enrollment_status', type: 'string', description: 'full_time or part_time' },
  { name: 'gpa', type: 'float', description: '0.0 to 4.0' },
  { name: 'credits_attempted', type: 'int', description: 'Total credits attempted' },
  { name: 'credits_completed', type: 'int', description: 'Total credits completed' },
  { name: 'failed_courses', type: 'int', description: 'Number of failed courses' },
  { name: 'attendance_rate', type: 'float', description: '0 to 100 (percentage)' },
  { name: 'lms_logins_last_30d', type: 'int', description: 'LMS logins in last 30 days' },
  { name: 'assignments_submitted', type: 'int', description: 'Assignments submitted' },
  { name: 'assignments_total', type: 'int', description: 'Total assignments' },
  { name: 'late_submissions', type: 'int', description: 'Late submissions count' },
  { name: 'advisor_meetings', type: 'int', description: 'Advisor meetings count' },
  { name: 'financial_aid', type: 'boolean', description: 'true or false' },
];

function BatchUpload() {
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isDownloadingPDF, setIsDownloadingPDF] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [showColumns, setShowColumns] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.name.endsWith('.csv')) {
      setFile(droppedFile);
      setError(null);
    } else {
      setError('Please upload a CSV file');
    }
  };

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const clearFile = () => {
    setFile(null);
    setResults(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSubmit = async () => {
    if (!file) return;

    setIsLoading(true);
    setError(null);
    setResults(null);

    try {
      const data = await predictBatch(file);
      setResults(data);
    } catch (err) {
      console.error('Batch prediction error:', err);
      if (err.response?.data?.detail) {
        setError(err.response.data.detail);
      } else {
        setError('Failed to process batch. Please check your CSV format.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const exportToCSV = () => {
    if (!results) return;

    const headers = ['Student ID', 'Risk Score', 'Risk %', 'Status', 'Top Risk Factor'];
    const rows = results.predictions.map(p => [
      p.student_id,
      p.risk_score.toFixed(3),
      p.risk_percentage,
      p.risk_status,
      p.risk_factors[0]?.factor || 'None'
    ]);

    const csvContent = [
      headers.join(','),
      ...rows.map(row => row.join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'dropout_predictions.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadPDF = async () => {
    if (!file) return;

    setIsDownloadingPDF(true);
    try {
      await downloadBatchReport(file);
    } catch (error) {
      console.error('Failed to download PDF report:', error);
      alert('Failed to download PDF report. Please try again.');
    } finally {
      setIsDownloadingPDF(false);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  const getRiskClass = (status) => {
    if (status === 'High Risk') return 'high';
    if (status === 'Medium Risk') return 'medium';
    return 'low';
  };

  return (
    <div className="card">
      <div
        className={`batch-upload-area ${isDragging ? 'drag-over' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleUploadClick}
      >
        <div className="upload-icon">📄</div>
        <div className="upload-text">
          {file ? file.name : 'Drop your CSV file here or click to browse'}
        </div>
        <div className="upload-hint">
          CSV file must contain all 15 required columns
        </div>
        <input
          type="file"
          ref={fileInputRef}
          className="file-input"
          accept=".csv"
          onChange={handleFileSelect}
        />
      </div>

      {/* Required Columns Section */}
      <div className="columns-section">
        <button
          className="columns-toggle"
          onClick={(e) => { e.stopPropagation(); setShowColumns(!showColumns); }}
        >
          <span>{showColumns ? '▼' : '▶'}</span>
          <span>Required CSV Columns ({REQUIRED_COLUMNS.length})</span>
        </button>

        {showColumns && (
          <div className="columns-table-wrapper">
            <table className="columns-table">
              <thead>
                <tr>
                  <th>Column Name</th>
                  <th>Type</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                {REQUIRED_COLUMNS.map((col, index) => (
                  <tr key={index}>
                    <td><code>{col.name}</code></td>
                    <td><span className="type-badge">{col.type}</span></td>
                    <td>{col.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {file && (
        <div className="selected-file">
          <span className="file-name">{file.name}</span>
          <span className="file-size">({formatFileSize(file.size)})</span>
          <button className="clear-file-btn" onClick={(e) => { e.stopPropagation(); clearFile(); }}>
            Remove
          </button>
        </div>
      )}

      {file && !results && (
        <button
          className="submit-btn"
          onClick={handleSubmit}
          disabled={isLoading}
          style={{ marginTop: '1.5rem' }}
        >
          {isLoading ? 'Processing...' : 'Analyze Students'}
        </button>
      )}

      {isLoading && (
        <div className="loading" style={{ marginTop: '2rem' }}>
          <div className="loading-spinner"></div>
          <p>Processing batch predictions...</p>
        </div>
      )}

      {error && (
        <div className="error-message" style={{ marginTop: '1.5rem' }}>
          {error}
        </div>
      )}

      {results && (
        <div className="batch-results">
          <div className="batch-summary">
            <div className="summary-stat">
              <div className="summary-stat-value">{results.total}</div>
              <div className="summary-stat-label">Total Students</div>
            </div>
            <div className="summary-stat high">
              <div className="summary-stat-value">{results.high_risk}</div>
              <div className="summary-stat-label">High Risk</div>
            </div>
            <div className="summary-stat medium">
              <div className="summary-stat-value">{results.medium_risk}</div>
              <div className="summary-stat-label">Medium Risk</div>
            </div>
            <div className="summary-stat low">
              <div className="summary-stat-value">{results.low_risk}</div>
              <div className="summary-stat-label">Low Risk</div>
            </div>
          </div>

          <h3>Prediction Results</h3>
          <table className="results-table">
            <thead>
              <tr>
                <th>Student ID</th>
                <th>Risk Score</th>
                <th>Status</th>
                <th>Top Risk Factor</th>
              </tr>
            </thead>
            <tbody>
              {results.predictions.map((prediction, index) => (
                <tr key={index}>
                  <td>{prediction.student_id}</td>
                  <td>{prediction.risk_percentage}%</td>
                  <td>
                    <span className={`risk-badge ${getRiskClass(prediction.risk_status)}`}>
                      {prediction.risk_status}
                    </span>
                  </td>
                  <td>{prediction.risk_factors[0]?.factor || 'None identified'}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="export-buttons">
            <button className="export-btn" onClick={exportToCSV}>
              <span>📥</span> Export to CSV
            </button>
            <button
              className="download-pdf-btn"
              onClick={handleDownloadPDF}
              disabled={isDownloadingPDF}
            >
              <span>📄</span>
              {isDownloadingPDF ? 'Generating...' : 'Download PDF Report'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default BatchUpload;
