FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt .
RUN apt-get update && apt-get install -y apt-utils && apt-get install -y gfortran libopenblas-dev liblapack-dev
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY input ./input
COPY OMDB_API_KEY .
COPY TMDB_API_KEY .
COPY run.sh .

RUN [ "/bin/bash", "-c", "source OMDB_API_KEY" ]
RUN [ "/bin/bash", "-c", "source TMDB_API_KEY" ]

EXPOSE 5000

CMD [ "/bin/bash", "run.sh", "-production" ]

#env FLASK_APP src/recommender-api.py

#CMD [ "flask", "run", "--host", "0.0.0.0" ]
