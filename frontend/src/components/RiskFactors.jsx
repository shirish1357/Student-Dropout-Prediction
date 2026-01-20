function RiskFactors({ factors }) {
  if (!factors || factors.length === 0) {
    return (
      <div className="card risk-factors-card">
        <h3>Risk Factors</h3>
        <p className="no-factors">No significant risk factors identified.</p>
      </div>
    );
  }

  // Only show top 5 factors to keep it clean
  const displayFactors = factors.slice(0, 6);

  return (
    <div className="card risk-factors-card">
      <h3>Risk Factors ({factors.length})</h3>
      {displayFactors.map((factor, index) => (
        <div key={index} className={`risk-factor ${factor.severity}`}>
          <div className="risk-factor-content">
            <div className="risk-factor-name">{factor.factor}</div>
            <div className="risk-factor-value">{factor.value}</div>
          </div>
          <span className={`severity-badge ${factor.severity}`}>
            {factor.severity}
          </span>
        </div>
      ))}
      {factors.length > 6 && (
        <p className="no-factors" style={{ marginTop: '0.5rem' }}>
          +{factors.length - 6} more factors
        </p>
      )}
    </div>
  );
}

export default RiskFactors;
