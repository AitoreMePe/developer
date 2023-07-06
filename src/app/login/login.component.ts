import { Component, OnInit } from '@angular/core';
import { AuthService } from '../services/auth.service';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent implements OnInit {

  constructor(private authService: AuthService) { }

  ngOnInit(): void {
  }

  loginWithGoogle(): void {
    this.authService.loginWithGoogle().then(() => {
      document.getElementById('loginButton').disabled = true;
    }).catch((error) => {
      console.error('Error logging in with Google', error);
    });
  }
}