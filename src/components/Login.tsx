import React from 'react';
import { useForm } from '../utils/hooks/useForm';
import { useAuth } from '../utils/hooks/useAuth';
import { loginStyles } from '../styles/LoginStyles';

const Login: React.FC = () => {
  const { values, handleChange, handleSubmit } = useForm(loginUser);
  const { login } = useAuth();

  async function loginUser() {
    try {
      await login(values.email, values.password);
    } catch (error) {
      console.error(error);
    }
  }

  return (
    <div style={loginStyles.container}>
      <form onSubmit={handleSubmit} style={loginStyles.form}>
        <label>
          Email
          <input
            type="email"
            name="email"
            value={values.email || ''}
            onChange={handleChange}
            style={loginStyles.input}
          />
        </label>
        <label>
          Password
          <input
            type="password"
            name="password"
            value={values.password || ''}
            onChange={handleChange}
            style={loginStyles.input}
          />
        </label>
        <button type="submit" style={loginStyles.button}>Login</button>
      </form>
    </div>
  );
};

export default Login;