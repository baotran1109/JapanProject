FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Back_end directory
COPY Back_end/ ./Back_end/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV PYTHONPATH=/app

EXPOSE 8080

# Use gunicorn with wsgi.py entry point
# This is more reliable than trying to import the app directly
WORKDIR /app/Back_end
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "300", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "info", "wsgi:application"]

