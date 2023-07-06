const mongoose = require('mongoose');

const UserSchema = new mongoose.Schema({
  gmail: String,
  ethereumWallet: String,
});

const PlayerListingSchema = new mongoose.Schema({
  playerName: String,
  game: String,
  stats: Object,
});

const MatchSchema = new mongoose.Schema({
  matchId: String,
  players: Array,
  result: String,
  gameStats: Object,
});

const TransferSchema = new mongoose.Schema({
  player: String,
  buyer: String,
  seller: String,
  price: Number,
});

const SponsorshipSchema = new mongoose.Schema({
  sponsor: String,
  team: String,
  amount: Number,
});

const VotingSchema = new mongoose.Schema({
  voter: String,
  votedPlayer: String,
});

const User = mongoose.model('User', UserSchema);
const PlayerListing = mongoose.model('PlayerListing', PlayerListingSchema);
const Match = mongoose.model('Match', MatchSchema);
const Transfer = mongoose.model('Transfer', TransferSchema);
const Sponsorship = mongoose.model('Sponsorship', SponsorshipSchema);
const Voting = mongoose.model('Voting', VotingSchema);

mongoose.connect('mongodb://localhost/UBLeague', { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => console.log('Database connected successfully'))
  .catch(err => console.log(err));

module.exports = {
  User,
  PlayerListing,
  Match,
  Transfer,
  Sponsorship,
  Voting
};