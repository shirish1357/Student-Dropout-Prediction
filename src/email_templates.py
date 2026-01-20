"""Email templates for student outreach."""

from .config import SUPPORT_EMAIL, SUPPORT_PHONE, APPOINTMENT_LINK

# Template placeholders:
# {student_name} - Student's first name
# {risk_level} - "High", "Moderate", or "Elevated"
# {support_email} - Advisor/support contact email
# {support_phone} - Support phone number
# {appointment_link} - Link to schedule appointment

TEMPLATES = {
    "at_risk_outreach": {
        "subject": "We're Here to Support You, {student_name}",
        "body_html": """
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background-color: #003366; color: white; padding: 20px; text-align: center; }}
        .content {{ padding: 20px; background-color: #f9f9f9; }}
        .cta-button {{
            display: inline-block;
            background-color: #0066cc;
            color: white;
            padding: 12px 24px;
            text-decoration: none;
            border-radius: 5px;
            margin: 15px 0;
        }}
        .footer {{ padding: 15px; text-align: center; font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Student Success Center</h1>
        </div>
        <div class="content">
            <p>Dear {student_name},</p>

            <p>We wanted to reach out and check in with you. As part of our commitment to student success,
            we regularly connect with students to ensure you have the support you need to thrive.</p>

            <p>We have a variety of resources available to help you succeed:</p>
            <ul>
                <li>Academic advising and tutoring services</li>
                <li>Study skills workshops</li>
                <li>Financial aid counseling</li>
                <li>Mental health and wellness support</li>
                <li>Career guidance</li>
            </ul>

            <p>We'd love to connect with you and discuss how we can support your academic journey.
            Please don't hesitate to reach out - we're here to help!</p>

            <p><a href="{appointment_link}" class="cta-button">Schedule a Meeting</a></p>

            <p>You can also reach us at:</p>
            <ul>
                <li>Email: <a href="mailto:{support_email}">{support_email}</a></li>
                <li>Phone: {support_phone}</li>
            </ul>

            <p>Best regards,<br>
            The Student Success Team</p>
        </div>
        <div class="footer">
            <p>This email was sent by the Student Success Center.</p>
            <p>If you believe you received this email in error, please contact us.</p>
        </div>
    </div>
</body>
</html>
""",
        "body_text": """Dear {student_name},

We wanted to reach out and check in with you. As part of our commitment to student success,
we regularly connect with students to ensure you have the support you need to thrive.

We have a variety of resources available to help you succeed:
- Academic advising and tutoring services
- Study skills workshops
- Financial aid counseling
- Mental health and wellness support
- Career guidance

We'd love to connect with you and discuss how we can support your academic journey.
Please don't hesitate to reach out - we're here to help!

Schedule a meeting: {appointment_link}

You can also reach us at:
- Email: {support_email}
- Phone: {support_phone}

Best regards,
The Student Success Team
""",
    },
    "high_risk_urgent": {
        "subject": "Important: Let's Connect, {student_name}",
        "body_html": """
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background-color: #990000; color: white; padding: 20px; text-align: center; }}
        .content {{ padding: 20px; background-color: #f9f9f9; }}
        .highlight {{ background-color: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 15px 0; }}
        .cta-button {{
            display: inline-block;
            background-color: #cc0000;
            color: white;
            padding: 12px 24px;
            text-decoration: none;
            border-radius: 5px;
            margin: 15px 0;
        }}
        .footer {{ padding: 15px; text-align: center; font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Student Success Center</h1>
        </div>
        <div class="content">
            <p>Dear {student_name},</p>

            <div class="highlight">
                <strong>We care about your success and want to help.</strong>
            </div>

            <p>We're reaching out because we want to make sure you have everything you need
            to succeed this semester. College can be challenging, and we're here to support you
            through any difficulties you may be facing.</p>

            <p>Our team can help with:</p>
            <ul>
                <li><strong>Academic Support:</strong> Tutoring, study groups, writing assistance</li>
                <li><strong>Personal Support:</strong> Counseling services, wellness resources</li>
                <li><strong>Financial Assistance:</strong> Emergency aid, financial planning</li>
                <li><strong>Career Services:</strong> Internships, job placement, resume help</li>
            </ul>

            <p><strong>We'd like to meet with you soon.</strong> Please schedule a time that works for you:</p>

            <p><a href="{appointment_link}" class="cta-button">Schedule Now</a></p>

            <p>If scheduling online doesn't work, please reach out directly:</p>
            <ul>
                <li>Email: <a href="mailto:{support_email}">{support_email}</a></li>
                <li>Phone: {support_phone}</li>
            </ul>

            <p>We're in your corner,<br>
            The Student Success Team</p>
        </div>
        <div class="footer">
            <p>This email was sent by the Student Success Center.</p>
        </div>
    </div>
</body>
</html>
""",
        "body_text": """Dear {student_name},

We care about your success and want to help.

We're reaching out because we want to make sure you have everything you need
to succeed this semester. College can be challenging, and we're here to support you
through any difficulties you may be facing.

Our team can help with:
- Academic Support: Tutoring, study groups, writing assistance
- Personal Support: Counseling services, wellness resources
- Financial Assistance: Emergency aid, financial planning
- Career Services: Internships, job placement, resume help

We'd like to meet with you soon. Please schedule a time that works for you:
{appointment_link}

If scheduling online doesn't work, please reach out directly:
- Email: {support_email}
- Phone: {support_phone}

We're in your corner,
The Student Success Team
""",
    },
}


def get_template(template_name: str) -> dict:
    """
    Get an email template by name.

    Args:
        template_name: Name of the template

    Returns:
        Dictionary with subject, body_html, and body_text

    Raises:
        ValueError: If template not found
    """
    if template_name not in TEMPLATES:
        available = ", ".join(TEMPLATES.keys())
        raise ValueError(f"Template '{template_name}' not found. Available: {available}")
    return TEMPLATES[template_name]


def format_template(
    template_name: str,
    student_name: str,
    risk_level: str = "Elevated",
) -> dict:
    """
    Format a template with student-specific values.

    Args:
        template_name: Name of the template to use
        student_name: Student's first name
        risk_level: Risk level (not shown to student, but affects template selection)

    Returns:
        Dictionary with formatted subject, body_html, and body_text
    """
    template = get_template(template_name)

    # Format with placeholder values
    format_values = {
        "student_name": student_name,
        "risk_level": risk_level,
        "support_email": SUPPORT_EMAIL,
        "support_phone": SUPPORT_PHONE,
        "appointment_link": APPOINTMENT_LINK,
    }

    return {
        "subject": template["subject"].format(**format_values),
        "body_html": template["body_html"].format(**format_values),
        "body_text": template["body_text"].format(**format_values),
    }


def list_templates() -> list:
    """Return list of available template names."""
    return list(TEMPLATES.keys())
