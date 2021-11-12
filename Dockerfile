FROM python:3.9-slim

COPY . ./app/

WORKDIR /app

RUN pip3 install -r requirements.txt

EXPOSE 8004

ENTRYPOINT ["python", "manage.py", "runserver", "8004"]