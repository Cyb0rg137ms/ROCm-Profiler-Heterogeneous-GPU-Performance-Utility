FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY profiler ./profiler
RUN pip install --no-cache-dir .
ENTRYPOINT ["rocm-profiler"]
CMD ["bench", "--quick"]
