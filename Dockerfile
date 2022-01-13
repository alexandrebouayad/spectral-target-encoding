FROM    python:3.9.7-bullseye AS app-src
COPY    . /opt/app
WORKDIR /opt/app

FROM    app-src
RUN     pip install -r /opt/app/requirements.txt
