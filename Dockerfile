FROM    python:3.9.7-bullseye AS app
COPY    . /opt/app
WORKDIR /opt/app

FROM    app
RUN     pip install -r /opt/app/requirements.txt
