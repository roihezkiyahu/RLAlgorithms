# Use an official Miniconda image from the Docker Hub as the base image
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /app

# Copy the environment.yml file to the working directory
COPY environment.yml .

# Create the Conda environment
RUN conda env create -f environment.yml

# Make sure the environment is activated when the container starts
SHELL ["conda", "run", "-n", "RLAlgorithms", "/bin/bash", "-c"]

# Copy the rest of the application code to the working directory
COPY . .

# Install any additional requirements
RUN conda run -n rlalgorithms_env pip install -e .

# Specify the command to run the application
ENTRYPOINT ["python", "-m", "trainer.task"]
