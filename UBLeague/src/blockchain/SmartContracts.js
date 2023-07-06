const ethers = require('ethers');

// Define the ABI (Application Binary Interface)
const abi = [
  // Player Transfer
  "function transferPlayer(address from, address to, uint256 tokenId) external payable",
  
  // Sponsorship
  "function createSponsorship(address to, uint256 tokenId) external payable",
  "function purchaseSponsorship(address from, address to, uint256 tokenId) external payable",
];

// Define the smart contract address
const contractAddress = "0xYourContractAddress";

// Connect to the Ethereum network
const provider = ethers.getDefaultProvider('mainnet');

// Create a new instance of the contract
const contract = new ethers.Contract(contractAddress, abi, provider);

// Define the functions for player transfer and sponsorship
async function transferPlayer(from, to, tokenId) {
  const transaction = await contract.transferPlayer(from, to, tokenId, {
    value: ethers.utils.parseEther("1.0"),
  });
  await transaction.wait();
}

async function createSponsorship(to, tokenId) {
  const transaction = await contract.createSponsorship(to, tokenId, {
    value: ethers.utils.parseEther("1.0"),
  });
  await transaction.wait();
}

async function purchaseSponsorship(from, to, tokenId) {
  const transaction = await contract.purchaseSponsorship(from, to, tokenId, {
    value: ethers.utils.parseEther("1.0"),
  });
  await transaction.wait();
}

module.exports = {
  transferPlayer,
  createSponsorship,
  purchaseSponsorship,
};