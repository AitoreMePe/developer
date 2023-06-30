import React from 'react';
import { useHistory } from 'react-router-dom';
import { useAuth } from '../utils/hooks/useAuth';
import { DashboardStyles } from '../styles/DashboardStyles';

const Dashboard: React.FC = () => {
  const { user, logout } = useAuth();
  const history = useHistory();

  const handleLogout = async () => {
    try {
      await logout();
      history.push('/login');
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div style={DashboardStyles.container}>
      <h1>Welcome, {user?.displayName}</h1>
      <button style={DashboardStyles.button} onClick={handleLogout}>
        Logout
      </button>
    </div>
  );
};

export default Dashboard;