FROM python:3.12-slim-bookworm
ARG RELEASE_VERSION="main"
WORKDIR /app
COPY . .
RUN pip install -r /app/requirements.txt
ENV CONTAINER=true
CMD ["python", "agents/product_planner.py"]

