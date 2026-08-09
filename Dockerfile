FROM python:3.13-slim

    # Copy uv binary from the official Astral uv image
    COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

    WORKDIR /app

    # python buffering of logs can get stuck in memory insteat of 
    # appearing in docker logs, this helps in stopping it
    ENV PYTHONUNBUFFERED=1

    # Copy dependency definition files first (enables Docker layer caching)
    COPY pyproject.toml uv.lock ./

    # Install dependencies using uv without installing the project yet
    RUN uv sync --frozen --no-install-project --no-dev

    # Copy the rest of the application code into the container
    COPY . .

    # Final sync to include project files
    RUN uv sync --frozen --no-dev

    # Expose FastAPI default port
    EXPOSE 8000

    # Run the FastAPI server using uv and uvicorn
    CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]