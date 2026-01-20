"""Vercel serverless function entry point."""
from api.main import app
# Vercel looks for 'app' or 'handler'
handler = app
