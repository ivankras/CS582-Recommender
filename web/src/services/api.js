import axios from "axios";

export const Movies = type => {
  let movieAPI = 'd6b0b8fa'
  return axios.create({
    baseURL: `http://www.omdbapi.com/?apikey=${movieAPI}&${type}`
  });
};


export const TRENDING_MOVIES = () => {
  return axios.create({
    baseURL: `http://localhost:5000/api/trending`
  });
};

export const CONTENT_MOVIE_SUGGESTIONS = movie_id => {
  return axios.create({
    baseURL: `http://localhost:5000/api/similar/${movie_id}`
  });
};

export const COLLABORATIVE_MOVIE_SUGGESTIONS = userID => {
  return axios.create({
    baseURL: `http://localhost:5000/api/rate/${userID}`
  });
};

export const MOVIE = movie_id => {
  return axios.create({
    baseURL: `http://localhost:5000/api/movie/${movie_id}`
  });
};