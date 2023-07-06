const mongoose = require('mongoose');

const PlayerListingSchema = new mongoose.Schema({
  playerName: {
    type: String,
    required: true
  },
  teamName: {
    type: String,
    required: true
  },
  game: {
    type: String,
    required: true
  },
  position: {
    type: String,
    required: true
  },
  stats: {
    type: Object,
    required: true
  },
  price: {
    type: Number,
    required: true
  },
  isAvailable: {
    type: Boolean,
    default: true
  },
  ownerWalletAddress: {
    type: String,
    required: true
  }
});

module.exports = mongoose.model('PlayerListing', PlayerListingSchema);