import React, { useState, useEffect } from 'react';
import ethers from 'ethers';
import SponsorshipContract from '../blockchain/SmartContracts';

const SponsorshipWindow = () => {
    const [sponsorships, setSponsorships] = useState([]);
    const [wallet, setWallet] = useState(null);

    useEffect(() => {
        // Assume wallet is connected and set
        const provider = new ethers.providers.Web3Provider(window.ethereum);
        const signer = provider.getSigner();
        setWallet(signer);
    }, []);

    useEffect(() => {
        const fetchSponsorships = async () => {
            const contract = new ethers.Contract(SponsorshipContract.address, SponsorshipContract.abi, wallet);
            const sponsorships = await contract.getSponsorships();
            setSponsorships(sponsorships);
        };

        if (wallet) {
            fetchSponsorships();
        }
    }, [wallet]);

    const handleSponsorshipPurchase = async (sponsorshipId) => {
        const contract = new ethers.Contract(SponsorshipContract.address, SponsorshipContract.abi, wallet);
        await contract.purchaseSponsorship(sponsorshipId);
    };

    return (
        <div>
            <h1>Sponsorship Window</h1>
            <table>
                <thead>
                    <tr>
                        <th>Sponsorship ID</th>
                        <th>Player</th>
                        <th>Team</th>
                        <th>Price</th>
                        <th>Action</th>
                    </tr>
                </thead>
                <tbody>
                    {sponsorships.map((sponsorship, index) => (
                        <tr key={index}>
                            <td>{sponsorship.id}</td>
                            <td>{sponsorship.player}</td>
                            <td>{sponsorship.team}</td>
                            <td>{ethers.utils.formatEther(sponsorship.price)}</td>
                            <td>
                                <button onClick={() => handleSponsorshipPurchase(sponsorship.id)}>
                                    Purchase Sponsorship
                                </button>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

export default SponsorshipWindow;