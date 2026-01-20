import { useState } from 'react';
import RiskFactors from './RiskFactors';
import { downloadSingleReport } from '../services/api';

function PredictionResult({ result, studentData }) {
  const [isDownloading, setIsDownloading] = useState(false);

  const getRiskColor = (status) => {
    if (status === 'High Risk') return '#dc2626';
    if (status === 'Medium Risk') return '#d97706';
    return '#16a34a';
  };

  const getRiskClass = (status) => {
    if (status === 'High Risk') return 'high';
    if (status === 'Medium Risk') return 'medium';
    return 'low';
  };

  const riskColor = getRiskColor(result.risk_status);
  const riskClass = getRiskClass(result.risk_status);

  const handleDownloadPDF = async () => {
    if (!studentData) return;

    setIsDownloading(true);
    try {
      await downloadSingleReport(studentData);
    } catch (error) {
      console.error('Failed to download report:', error);
      alert('Failed to download report. Please try again.');
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <div className="results-container">
      {/* Risk Score Card */}
      <div className="card risk-score-card">
        <div
          className="risk-gauge"
          style={{
            '--risk-color': riskColor,
            '--risk-percentage': result.risk_percentage,
          }}
        >
          <div className="risk-gauge-circle">
            <div className="risk-gauge-inner">
              <span className="risk-percentage" style={{ color: riskColor }}>
                {result.risk_percentage}%
              </span>
              <span className="risk-label">Dropout Risk</span>
            </div>
          </div>
        </div>
        <div className={`risk-status ${riskClass}`}>{result.risk_status}</div>
        <div className="student-id-display">
          <span>Student: </span>
          <strong>{result.student_id}</strong>
        </div>
      </div>

      {/* Risk Factors & Recommendations */}
      <div>
        <RiskFactors factors={result.risk_factors} />

        <div className="card recommendations-card" style={{ marginTop: '1.5rem' }}>
          <h3>Recommendations</h3>
          {result.recommendations.length > 0 ? (
            result.recommendations.map((rec, index) => (
              <div key={index} className="recommendation">
                <span className="recommendation-icon">{index + 1}</span>
                <span className="recommendation-text">{rec}</span>
              </div>
            ))
          ) : (
            <p className="no-factors">No specific recommendations at this time.</p>
          )}
        </div>

        {/* Download PDF Button */}
        {studentData && (
          <button
            className="download-pdf-btn"
            onClick={handleDownloadPDF}
            disabled={isDownloading}
            style={{ marginTop: '1.5rem' }}
          >
            <span>📄</span>
            {isDownloading ? 'Generating Report...' : 'Download PDF Report'}
          </button>
        )}
      </div>
    </div>
  );
}

export default PredictionResult;
