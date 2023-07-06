import React, { useState, useEffect } from 'react';
import ethers from 'ethers';
import { Transfers } from '../../models/Transfers';

const TransfersWindow = () => {
    const [transfers, setTransfers] = useState([]);

    useEffect(() => {
        fetchTransfers();
    }, []);

    const fetchTransfers = async () => {
        const transfersData = await Transfers.find();
        setTransfers(transfersData);
    };

    const handleTransfer = async (playerId, buyerWallet) => {
        const provider = new ethers.providers.Web3Provider(window.ethereum);
        const signer = provider.getSigner();
        const contract = new ethers.Contract(playerId, Transfers.abi, signer);

        const transaction = await contract.transferFrom(
            ethers.utils.getAddress(window.ethereum.selectedAddress),
            ethers.utils.getAddress(buyerWallet),
            playerId
        );

        await transaction.wait();
        fetchTransfers();
    };

    return (
        <div>
            <h1>Transfers</h1>
            <table>
                <thead>
                    <tr>
                        <th>Player ID</th>
                        <th>Owner Wallet</th>
                        <th>Price</th>
                        <th>Action</th>
                    </tr>
                </thead>
                <tbody>
                    {transfers.map((transfer) => (
                        <tr key={transfer._id}>
                            <td>{transfer.playerId}</td>
                            <td>{transfer.ownerWallet}</td>
                            <td>{transfer.price}</td>
                            <td>
                                <button onClick={() => handleTransfer(transfer.playerId, transfer.buyerWallet)}>
                                    Buy
                                </button>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

export default TransfersWindow;