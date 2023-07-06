import React, { useState } from 'react';
import axios from 'axios';

const SearchWindow = () => {
    const [searchTerm, setSearchTerm] = useState('');
    const [playerData, setPlayerData] = useState(null);

    const handleSearch = async () => {
        try {
            const response = await axios.get(`/api/players?name=${searchTerm}`);
            setPlayerData(response.data);
        } catch (error) {
            console.error('Error fetching player data:', error);
        }
    };

    return (
        <div>
            <input
                type="text"
                placeholder="Search for a player"
                value={searchTerm}
                onChange={e => setSearchTerm(e.target.value)}
            />
            <button onClick={handleSearch}>Search</button>
            {playerData && (
                <div>
                    <h2>{playerData.name}</h2>
                    <p>{playerData.description}</p>
                </div>
            )}
        </div>
    );
};

export default SearchWindow;