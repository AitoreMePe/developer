import React from 'react';
import { useForm } from '../utils/hooks/useForm';
import { useAuth } from '../utils/hooks/useAuth';
import { SignupStyles } from '../styles/SignupStyles';

const Signup: React.FC = () => {
  const { formState, updateFormState } = useForm({
    email: '',
    password: '',
  });

  const { signup } = useAuth();

  const handleSignup = async (event: React.FormEvent) => {
    event.preventDefault();
    const { email, password } = formState;
    try {
      await signup(email, password);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div style={SignupStyles.container}>
      <form onSubmit={handleSignup} style={SignupStyles.form}>
        <input
          type="email"
          name="email"
          placeholder="Email"
          value={formState.email}
          onChange={updateFormState}
          style={SignupStyles.input}
        />
        <input
          type="password"
          name="password"
          placeholder="Password"
          value={formState.password}
          onChange={updateFormState}
          style={SignupStyles.input}
        />
        <button type="submit" style={SignupStyles.button}>
          Sign Up
        </button>
      </form>
    </div>
  );
};

export default Signup;