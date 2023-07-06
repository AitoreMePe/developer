import React, { useState, useEffect } from 'react';
import axios from 'axios';

const ItemDetailsWindow = () => {
    const [itemDetails, setItemDetails] = useState({});

    useEffect(() => {
        // Assuming we have an endpoint that fetches the item details
        axios.get('/api/items/details')
            .then(response => {
                setItemDetails(response.data);
            })
            .catch(error => {
                console.error('Error fetching item details:', error);
            });
    }, []);

    return (
        <div>
            <h2>Item Details</h2>
            <div>
                <h3>Game Statistics</h3>
                <p>{itemDetails.gameStats}</p>
            </div>
            <div>
                <h3>Player Performance</h3>
                <p>{itemDetails.playerPerformance}</p>
            </div>
            <div>
                <h3>Game Time</h3>
                <p>{itemDetails.gameTime}</p>
            </div>
            <div>
                <h3>Other Relevant Data</h3>
                <p>{itemDetails.otherData}</p>
            </div>
        </div>
    );
};

export default ItemDetailsWindow;