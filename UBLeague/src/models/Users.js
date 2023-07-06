const mongoose = require('mongoose');

const UserSchema = new mongoose.Schema({
  gmail: {
    type: String,
    required: true,
    unique: true
  },
  ethereumWallet: {
    type: String,
    required: true,
    unique: true
  },
  registeredAt: {
    type: Date,
    default: Date.now
  }
});

module.exports = mongoose.model('User', UserSchema);