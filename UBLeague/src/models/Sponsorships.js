const mongoose = require('mongoose');

const SponsorshipSchema = new mongoose.Schema({
    sponsor: {
        type: String,
        required: true
    },
    team: {
        type: String,
        required: true
    },
    player: {
        type: String,
        required: true
    },
    amount: {
        type: Number,
        required: true
    },
    duration: {
        type: Number,
        required: true
    },
    contractAddress: {
        type: String,
        required: true
    }
});

module.exports = mongoose.model('Sponsorship', SponsorshipSchema);