1. Exported Variables:
   - `environment`: Variable exported from `environment.ts` and `environment.prod.ts` files, used in `main.ts` and `app.module.ts` to set up the application environment.

2. Data Schemas:
   - `User`: Schema used to represent a user, shared between `auth.service.ts` and `wallet.service.ts`.
   - `Wallet`: Schema used to represent a wallet, shared between `wallet.service.ts` and `wallet.component.ts`.

3. ID Names of DOM Elements:
   - `loginButton`: ID for the login button in `login.component.html`, used in `login.component.ts`.
   - `walletAddress`: ID for the wallet address display in `wallet.component.html`, used in `wallet.component.ts`.
   - `connectionStatus`: ID for the connection status display in `wallet.component.html`, used in `wallet.component.ts`.

4. Message Names:
   - `loginSuccess`: Message dispatched from `auth.service.ts` upon successful login, listened to in `app.component.ts`.
   - `walletAssigned`: Message dispatched from `wallet.service.ts` upon successful wallet assignment, listened to in `wallet.component.ts`.

5. Function Names:
   - `loginWithGoogle`: Function in `auth.service.ts` used in `login.component.ts`.
   - `assignWallet`: Function in `wallet.service.ts` used in `app.component.ts` and `wallet.component.ts`.
   - `getWalletAddress`: Function in `wallet.service.ts` used in `wallet.component.ts`.
   - `getConnectionStatus`: Function in `wallet.service.ts` used in `wallet.component.ts`.

6. Shared Dependencies:
   - `@angular/core`, `@angular/common`, `@angular/forms`, `@angular/router`: Angular modules used across multiple components.
   - `firebase`, `angularfire2`: Dependencies for Google login and Firebase integration, used in `auth.service.ts` and `app.module.ts`.
   - `ethereumjs-wallet`: Dependency for Ethereum wallet management, used in `wallet.service.ts`.
   - `rxjs`: Reactive Extensions Library for JavaScript, used across multiple services and components for handling asynchronous data streams.