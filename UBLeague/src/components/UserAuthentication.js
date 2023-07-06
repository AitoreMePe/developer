```javascript
import React, { useState, useEffect } from 'react';
import firebase from 'firebase/app';
import 'firebase/auth';
import { ethers } from 'ethers';

const UserAuthentication = () => {
  const [user, setUser] = useState(null);
  const [wallet, setWallet] = useState(null);

  useEffect(() => {
    firebase.auth().onAuthStateChanged((user) => {
      if (user) {
        setUser(user);
        assignWallet(user);
      } else {
        setUser(null);
        setWallet(null);
      }
    });
  }, []);

  const assignWallet = async (user) => {
    const provider = new ethers.providers.Web3Provider(window.ethereum);
    const signer = provider.getSigner();
    const wallet = ethers.Wallet.createRandom().connect(signer);
    // Save the wallet address to the user's profile in the database
    // This is a placeholder and should be replaced with actual database code
    user.wallet = wallet.address;
    setWallet(wallet);
  };

  const login = () => {
    const provider = new firebase.auth.GoogleAuthProvider();
    firebase.auth().signInWithPopup(provider);
  };

  const logout = () => {
    firebase.auth().signOut();
  };

  return (
    <div>
      {user ? (
        <button onClick={logout}>Logout</button>
      ) : (
        <button onClick={login}>Login with Google</button>
      )}
    </div>
  );
};

export default UserAuthentication;
```