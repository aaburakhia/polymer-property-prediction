# Use a standard Python base image. Version 3.11 matches your Kaggle notebook.
FROM python:3.11

# Create a non-root user and set its home directory
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory inside the user's home
WORKDIR /home/user/app

# Copy requirements and install dependencies
COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY --chown=user . .

# Expose the port and run the app
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860"]