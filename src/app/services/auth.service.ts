import { Injectable } from '@angular/core';
import { AngularFireAuth } from '@angular/fire/auth';
import { User } from '../models/user.model';
import { Subject } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private loginSuccess = new Subject<User>();

  constructor(private afAuth: AngularFireAuth) { }

  loginWithGoogle() {
    return this.afAuth.signInWithPopup(new firebase.auth.GoogleAuthProvider())
      .then((result) => {
        const user = new User();
        user.email = result.user.email;
        user.name = result.user.displayName;
        this.loginSuccess.next(user);
      });
  }

  getLoginSuccess() {
    return this.loginSuccess.asObservable();
  }
}