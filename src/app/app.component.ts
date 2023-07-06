import { Component, OnInit } from '@angular/core';
import { AuthService } from './services/auth.service';
import { WalletService } from './services/wallet.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'Ethereum Wallet App';

  constructor(private authService: AuthService, private walletService: WalletService) {}

  ngOnInit() {
    this.authService.loginSuccess.subscribe(user => {
      this.walletService.assignWallet(user);
    });
  }
}