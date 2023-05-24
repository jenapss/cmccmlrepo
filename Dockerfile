FROM python:3.9

WORKDIR /app

# Copy all project files to the container
COPY . /app

# Install project dependencies
RUN pip install -r requirements.txt

EXPOSE 8000
RUN echo "8000 PORT is EXPOSED"
# Set the command to run when the container starts
CMD ["python", "app.py"]
