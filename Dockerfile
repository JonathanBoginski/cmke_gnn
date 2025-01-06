# Basis-Image (Python 3.9 als Beispiel)
FROM python:3.9-slim

# Setze das Arbeitsverzeichnis im Container
WORKDIR /app

# Kopiere requirements.txt und installiere Abhängigkeiten
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiere den restlichen Code ins Arbeitsverzeichnis
COPY . .

# Exponiere den Port für den Uvicorn-Server
EXPOSE 8001

# Startbefehl für den Uvicorn-Server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]