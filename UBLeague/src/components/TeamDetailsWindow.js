import React, { useState, useEffect } from 'react';
import axios from 'axios';

const TeamDetailsWindow = () => {
    const [teamDetails, setTeamDetails] = useState({});

    useEffect(() => {
        const fetchTeamDetails = async () => {
            const response = await axios.get('/api/teamDetails');
            setTeamDetails(response.data);
        };

        fetchTeamDetails();
    }, []);

    return (
        <div>
            <h2>Team Details</h2>
            <p>Team Name: {teamDetails.name}</p>
            <p>Team Members: {teamDetails.members && teamDetails.members.join(', ')}</p>
            <p>Team Stats: {teamDetails.stats}</p>
            <p>Match History: {teamDetails.matchHistory && teamDetails.matchHistory.join(', ')}</p>
        </div>
    );
};

export default TeamDetailsWindow;