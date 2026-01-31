# Stage 1: Builder
FROM python:3.12-slim AS builder

# Install build dependencies
RUN pip install --no-cache-dir build==1.3.0

# Set the working directory
WORKDIR /app

# Copy the project files
COPY pyproject.toml .
COPY src/ ./src/
COPY README.md .
COPY LICENSE .

# Build the wheel
RUN python -m build --wheel --outdir /wheels


# Stage 2: Runtime
FROM python:3.12-slim AS runtime

# Create the user and the application directory explicitly
RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /home/appuser/app/logs

# Set the working directory
WORKDIR /home/appuser/app

# Copy the wheel from the builder stage
COPY --from=builder /wheels /wheels

# Install the application wheel and ensure appuser owns the entire application directory including logs
RUN pip install --no-cache-dir /wheels/*.whl && \
    chown -R appuser:appuser /home/appuser/app

# Switch to non-root user
USER appuser

# Add user's local bin to PATH
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Execution
EXPOSE 8000 50052 50055
CMD ["start", "serve"]
