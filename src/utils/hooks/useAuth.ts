import { useState, useEffect, useContext, createContext } from 'react';
import { auth } from '../../services/auth';

interface ContextProps {
  user: firebase.User | null;
  loading: boolean;
}

const authContext = createContext<Partial<ContextProps>>({});

export function ProvideAuth({ children }: any) {
  const auth = useAuthProvider();
  return <authContext.Provider value={auth}>{children}</authContext.Provider>;
}

export const useAuth = () => {
  return useContext(authContext);
};

function useAuthProvider() {
  const [user, setUser] = useState<firebase.User | null>(null);
  const [loading, setLoading] = useState(true);

  const handleUser = (rawUser: firebase.User | null) => {
    if (rawUser) {
      setUser(rawUser);
      setLoading(false);
      return rawUser;
    } else {
      setUser(null);
      setLoading(false);
      return false;
    }
  };

  useEffect(() => {
    const unsubscribe = auth.onAuthStateChanged(handleUser);
    return () => unsubscribe();
  }, []);

  return {
    user,
    loading,
  };
}