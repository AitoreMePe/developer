const mongoose = require('mongoose');

const VotingSchema = new mongoose.Schema({
  voter: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  votedPlayer: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'PlayerListings',
    required: true
  },
  voteDate: {
    type: Date,
    default: Date.now
  }
});

module.exports = mongoose.model('Voting', VotingSchema);