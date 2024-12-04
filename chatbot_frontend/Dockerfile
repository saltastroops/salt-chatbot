FROM python:3.10-slim

WORKDIR /app
COPY ./src/ /app

COPY /requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

CMD ["sh", "entrypoint.sh"]