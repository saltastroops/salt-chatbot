# chatbot_api/Dockerfile

FROM python:3.10-slim

WORKDIR /app
COPY ./src/ /app

COPY /requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

EXPOSE 8000
ENTRYPOINT ["sh", "entrypoint.sh"]