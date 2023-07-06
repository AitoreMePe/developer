import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { RouterModule, Routes } from '@angular/router';
import { AngularFireModule } from 'angularfire2';
import { AngularFireAuthModule } from 'angularfire2/auth';

import { AppComponent } from './app.component';
import { LoginComponent } from './login/login.component';
import { WalletComponent } from './wallet/wallet.component';

import { AuthService } from './services/auth.service';
import { WalletService } from './services/wallet.service';

import { environment } from '../environments/environment';

const appRoutes: Routes = [
  { path: 'login', component: LoginComponent },
  { path: 'wallet', component: WalletComponent },
  { path: '', redirectTo: '/login', pathMatch: 'full' }
];

@NgModule({
  declarations: [
    AppComponent,
    LoginComponent,
    WalletComponent
  ],
  imports: [
    BrowserModule,
    FormsModule,
    RouterModule.forRoot(appRoutes),
    AngularFireModule.initializeApp(environment.firebase),
    AngularFireAuthModule
  ],
  providers: [AuthService, WalletService],
  bootstrap: [AppComponent]
})
export class AppModule { }