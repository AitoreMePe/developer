const mongoose = require('mongoose');

const TransferSchema = new mongoose.Schema({
    player: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'PlayerListings',
        required: true
    },
    seller: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Users',
        required: true
    },
    buyer: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Users',
        required: true
    },
    price: {
        type: Number,
        required: true
    },
    transferDate: {
        type: Date,
        default: Date.now
    },
    nftToken: {
        type: String,
        required: true
    }
});

module.exports = mongoose.model('Transfers', TransferSchema);