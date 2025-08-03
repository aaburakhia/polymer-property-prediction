# Use a standard Python base image. Version 3.11 matches your Kaggle notebook.
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first, to leverage Docker's caching.
COPY requirements.txt ./

# Install all the Python dependencies from your requirements file.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application files (app.py, models folder, etc.)
COPY . .

# Expose the port that Streamlit will run on. Hugging Face expects port 7860.
EXPOSE 7860

# The command to run when the container starts.
# This tells the system to start your Streamlit app.
CMD ["streamlit", "run", "app.py", "--server.port=7860"]