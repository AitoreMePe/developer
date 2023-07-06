import { Component, OnInit } from '@angular/core';
import { WalletService } from '../services/wallet.service';

@Component({
  selector: 'app-wallet',
  templateUrl: './wallet.component.html',
  styleUrls: ['./wallet.component.css']
})
export class WalletComponent implements OnInit {
  walletAddress: string;
  connectionStatus: string;

  constructor(private walletService: WalletService) { }

  ngOnInit() {
    this.walletService.walletAssigned.subscribe(() => {
      this.walletAddress = this.walletService.getWalletAddress();
      this.connectionStatus = this.walletService.getConnectionStatus();
    });
  }
}