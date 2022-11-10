FROM python:3.9-slim
RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["flask_deploy_predict.py", "logic_of_web_service.py", "modellog5.bin","./" ]

EXPOSE 7777

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:7777", "predict risk of stroke:app"]