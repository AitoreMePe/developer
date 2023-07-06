import React, { useState, useEffect } from 'react';
import axios from 'axios';

const PlayerDetailsWindow = ({ selectedPlayer }) => {
    const [playerDetails, setPlayerDetails] = useState(null);

    useEffect(() => {
        const fetchPlayerDetails = async () => {
            try {
                const response = await axios.get(`/api/players/${selectedPlayer}`);
                setPlayerDetails(response.data);
            } catch (error) {
                console.error('Error fetching player details:', error);
            }
        };

        if (selectedPlayer) {
            fetchPlayerDetails();
        }
    }, [selectedPlayer]);

    if (!playerDetails) {
        return <div>Loading...</div>;
    }

    return (
        <div>
            <h2>{playerDetails.name}</h2>
            <p>Game Stats: {playerDetails.gameStats}</p>
            <p>Match History: {playerDetails.matchHistory}</p>
            <p>Other Relevant Data: {playerDetails.otherData}</p>
        </div>
    );
};

export default PlayerDetailsWindow;