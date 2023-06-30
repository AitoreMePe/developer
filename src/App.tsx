import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import { createGlobalStyle } from 'styled-components';

import { useAuth } from './utils/hooks/useAuth';
import Login from './components/Login';
import Signup from './components/Signup';
import Dashboard from './components/Dashboard';

import GlobalStyles from './styles/global';

const Global = createGlobalStyle`${GlobalStyles}`;

const App: React.FC = () => {
  const { initializing, user } = useAuth();

  if (initializing) {
    return <div>Loading...</div>;
  }

  return (
    <Router>
      <Global />
      <Switch>
        <Route path="/login" component={Login} />
        <Route path="/signup" component={Signup} />
        <Route path="/" render={() => user ? <Dashboard /> : <Login />} />
      </Switch>
    </Router>
  );
};

export default App;