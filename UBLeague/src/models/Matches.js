const mongoose = require('mongoose');

const MatchesSchema = new mongoose.Schema({
  matchId: {
    type: String,
    required: true,
    unique: true
  },
  date: {
    type: Date,
    required: true
  },
  team1: {
    type: String,
    required: true
  },
  team2: {
    type: String,
    required: true
  },
  score: {
    type: String,
    required: true
  },
  winner: {
    type: String,
    required: true
  },
  gameStats: {
    type: Object,
    required: true
  }
});

module.exports = mongoose.model('Matches', MatchesSchema);