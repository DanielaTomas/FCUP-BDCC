FROM python:3.9-slim-buster

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

ENV GOOGLE_CLOUD_PROJECT="bdcc-project1-417811"

EXPOSE 8080

CMD ["python3", "main.py"]

