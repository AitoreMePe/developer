const express = require('express');
const router = express.Router();
const mongoose = require('mongoose');

const User = require('../models/Users');
const PlayerListings = require('../models/PlayerListings');
const Matches = require('../models/Matches');
const Transfers = require('../models/Transfers');
const Sponsorships = require('../models/Sponsorships');
const Voting = require('../models/Voting');

router.get('/users', async (req, res) => {
    const users = await User.find();
    res.json(users);
});

router.get('/playerListings', async (req, res) => {
    const playerListings = await PlayerListings.find();
    res.json(playerListings);
});

router.get('/matches', async (req, res) => {
    const matches = await Matches.find();
    res.json(matches);
});

router.get('/transfers', async (req, res) => {
    const transfers = await Transfers.find();
    res.json(transfers);
});

router.get('/sponsorships', async (req, res) => {
    const sponsorships = await Sponsorships.find();
    res.json(sponsorships);
});

router.get('/voting', async (req, res) => {
    const voting = await Voting.find();
    res.json(voting);
});

module.exports = router;