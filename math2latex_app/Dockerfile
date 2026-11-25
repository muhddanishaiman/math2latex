FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir fastapi "uvicorn[standard]" pillow numpy httpx python-multipart
COPY app ./app
COPY math2latex_model.pt .
ENV MODEL_PATH=/srv/math2latex_model.pt
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
