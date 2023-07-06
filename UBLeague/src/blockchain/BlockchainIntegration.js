const ethers = require('ethers');

class BlockchainIntegration {
    constructor() {
        this.provider = new ethers.providers.JsonRpcProvider('http://localhost:8545');
    }

    async getWallet(privateKey) {
        return new ethers.Wallet(privateKey, this.provider);
    }

    async createTransaction(wallet, to, value) {
        const transaction = {
            to: to,
            value: ethers.utils.parseEther(value.toString())
        };

        return await wallet.sendTransaction(transaction);
    }

    async getTransactionReceipt(transactionHash) {
        return await this.provider.getTransactionReceipt(transactionHash);
    }

    async getBalance(address) {
        const balance = await this.provider.getBalance(address);
        return ethers.utils.formatEther(balance);
    }
}

module.exports = BlockchainIntegration;