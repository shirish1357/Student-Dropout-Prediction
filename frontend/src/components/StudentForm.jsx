import { useState } from 'react';

const PROGRAMS = [
  'Computer Science',
  'Business',
  'Engineering',
  'Liberal Arts',
  'Nursing',
];

const initialFormData = {
  student_id: '',
  enrollment_date: '',
  program: 'Computer Science',
  enrollment_status: 'full_time',
  gpa: '',
  credits_attempted: '',
  credits_completed: '',
  failed_courses: '0',
  attendance_rate: '',
  lms_logins_last_30d: '',
  assignments_submitted: '',
  assignments_total: '',
  late_submissions: '0',
  advisor_meetings: '0',
  financial_aid: false,
};

function StudentForm({ onSubmit, isLoading }) {
  const [formData, setFormData] = useState(initialFormData);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    // Convert string values to appropriate types
    const submitData = {
      ...formData,
      gpa: parseFloat(formData.gpa),
      credits_attempted: parseInt(formData.credits_attempted),
      credits_completed: parseInt(formData.credits_completed),
      failed_courses: parseInt(formData.failed_courses),
      attendance_rate: parseFloat(formData.attendance_rate),
      lms_logins_last_30d: parseInt(formData.lms_logins_last_30d),
      assignments_submitted: parseInt(formData.assignments_submitted),
      assignments_total: parseInt(formData.assignments_total),
      late_submissions: parseInt(formData.late_submissions),
      advisor_meetings: parseInt(formData.advisor_meetings),
    };

    onSubmit(submitData);
  };

  return (
    <form onSubmit={handleSubmit} className="card">
      {/* Student Information */}
      <div className="form-section">
        <h3>Student Information</h3>
        <div className="form-grid">
          <div className="form-group">
            <label htmlFor="student_id">Student ID</label>
            <input
              type="text"
              id="student_id"
              name="student_id"
              value={formData.student_id}
              onChange={handleChange}
              placeholder="e.g., STU00123"
              required
            />
          </div>
          <div className="form-group">
            <label htmlFor="enrollment_date">Enrollment Date</label>
            <input
              type="date"
              id="enrollment_date"
              name="enrollment_date"
              value={formData.enrollment_date}
              onChange={handleChange}
              required
            />
          </div>
          <div className="form-group">
            <label htmlFor="program">Program</label>
            <select
              id="program"
              name="program"
              value={formData.program}
              onChange={handleChange}
              required
            >
              {PROGRAMS.map((prog) => (
                <option key={prog} value={prog}>
                  {prog}
                </option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label htmlFor="enrollment_status">Enrollment Status</label>
            <select
              id="enrollment_status"
              name="enrollment_status"
              value={formData.enrollment_status}
              onChange={handleChange}
              required
            >
              <option value="full_time">Full Time</option>
              <option value="part_time">Part Time</option>
            </select>
          </div>
        </div>
      </div>

      {/* Academic Performance */}
      <div className="form-section">
        <h3>Academic Performance</h3>
        <div className="form-grid">
          <div className="form-group">
            <label htmlFor="gpa">GPA (0.0 - 4.0)</label>
            <input
              type="number"
              id="gpa"
              name="gpa"
              value={formData.gpa}
              onChange={handleChange}
              min="0"
              max="4"
              step="0.01"
              placeholder="e.g., 3.25"
              required
            />
          </div>
          <div className="form-group">
            <label htmlFor="credits_attempted">Credits Attempted</label>
            <input
              type="number"
              id="credits_attempted"
              name="credits_attempted"
              value={formData.credits_attempted}
              onChange={handleChange}
              min="0"
              placeholder="e.g., 45"
              required
            />
          </div>
          <div className="form-group">
            <label htmlFor="credits_completed">Credits Completed</label>
            <input
              type="number"
              id="credits_completed"
              name="credits_completed"
              value={formData.credits_completed}
              onChange={handleChange}
              min="0"
              placeholder="e.g., 42"
              required
            />
          </div>
          <div className="form-group">
            <label htmlFor="failed_courses">Failed Courses</label>
            <input
              type="number"
              id="failed_courses"
              name="failed_courses"
              value={formData.failed_courses}
              onChange={handleChange}
              min="0"
              placeholder="e.g., 0"
              required
            />
          </div>
        </div>
      </div>

      {/* Engagement Metrics */}
      <div className="form-section">
        <h3>Engagement Metrics</h3>
        <div className="form-grid">
          <div className="form-group">
            <label htmlFor="attendance_rate">Attendance Rate (%)</label>
            <input
              type="number"
              id="attendance_rate"
              name="attendance_rate"
              value={formData.attendance_rate}
              onChange={handleChange}
              min="0"
              max="100"
              step="0.1"
              placeholder="e.g., 85"
              required
            />
          </div>
          <div className="form-group">
            <label htmlFor="lms_logins_last_30d">LMS Logins (Last 30 Days)</label>
            <input
              type="number"
              id="lms_logins_last_30d"
              name="lms_logins_last_30d"
              value={formData.lms_logins_last_30d}
              onChange={handleChange}
              min="0"
              placeholder="e.g., 20"
              required
            />
          </div>
        </div>
      </div>

      {/* Assignment Metrics */}
      <div className="form-section">
        <h3>Assignment Metrics</h3>
        <div className="form-grid">
          <div className="form-group">
            <label htmlFor="assignments_submitted">Assignments Submitted</label>
            <input
              type="number"
              id="assignments_submitted"
              name="assignments_submitted"
              value={formData.assignments_submitted}
              onChange={handleChange}
              min="0"
              placeholder="e.g., 18"
              required
            />
          </div>
          <div className="form-group">
            <label htmlFor="assignments_total">Total Assignments</label>
            <input
              type="number"
              id="assignments_total"
              name="assignments_total"
              value={formData.assignments_total}
              onChange={handleChange}
              min="1"
              placeholder="e.g., 20"
              required
            />
          </div>
          <div className="form-group">
            <label htmlFor="late_submissions">Late Submissions</label>
            <input
              type="number"
              id="late_submissions"
              name="late_submissions"
              value={formData.late_submissions}
              onChange={handleChange}
              min="0"
              placeholder="e.g., 2"
              required
            />
          </div>
        </div>
      </div>

      {/* Support & Resources */}
      <div className="form-section">
        <h3>Support & Resources</h3>
        <div className="form-grid">
          <div className="form-group">
            <label htmlFor="advisor_meetings">Advisor Meetings</label>
            <input
              type="number"
              id="advisor_meetings"
              name="advisor_meetings"
              value={formData.advisor_meetings}
              onChange={handleChange}
              min="0"
              placeholder="e.g., 2"
              required
            />
          </div>
          <div className="form-group checkbox-group">
            <input
              type="checkbox"
              id="financial_aid"
              name="financial_aid"
              checked={formData.financial_aid}
              onChange={handleChange}
            />
            <label htmlFor="financial_aid">Receiving Financial Aid</label>
          </div>
        </div>
      </div>

      <button type="submit" className="submit-btn" disabled={isLoading}>
        {isLoading ? 'Analyzing...' : 'Predict Dropout Risk'}
      </button>
    </form>
  );
}

export default StudentForm;
