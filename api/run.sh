#!/bin/bash

export FLASK_APP=src/endpoints.py
if [ "$1" != '-production' ]
then
	export FLASK_ENV=development
else
	export FLASK_ENV=production
fi

if [ -f OMDB_API_KEY ]
then
	source OMDB_API_KEY
else
	echo "No OMDB_API_KEY file found."
fi

if [ -f TMDB_API_KEY ]
then
	source TMDB_API_KEY
else
	echo "No TMDB_API_KEY file found"
fi


flask run --host 0.0.0.0