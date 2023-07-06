import { Injectable } from '@angular/core';
import { User } from '../models/user.model';
import { Wallet } from '../models/wallet.model';
import { BehaviorSubject, Observable } from 'rxjs';
import * as EthereumWallet from 'ethereumjs-wallet';

@Injectable({
  providedIn: 'root'
})
export class WalletService {
  private wallet: Wallet;
  private walletSubject: BehaviorSubject<Wallet>;
  public wallet$: Observable<Wallet>;

  constructor() {
    this.walletSubject = new BehaviorSubject<Wallet>(this.wallet);
    this.wallet$ = this.walletSubject.asObservable();
  }

  assignWallet(user: User): void {
    // Here we would generate and assign a new Ethereum wallet to the user.
    // This is a placeholder implementation.
    this.wallet = {
      address: EthereumWallet.generate().getAddressString(),
      user: user
    };
    this.walletSubject.next(this.wallet);
  }

  getWalletAddress(): string {
    return this.wallet ? this.wallet.address : null;
  }

  getConnectionStatus(): string {
    // Here we would check the connection status of the wallet.
    // This is a placeholder implementation.
    return this.wallet ? 'Connected' : 'Not connected';
  }
}