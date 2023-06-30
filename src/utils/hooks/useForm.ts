import { useState } from 'react';

interface FormState {
  [key: string]: string;
}

const useForm = (initialState: FormState) => {
  const [values, setValues] = useState<FormState>(initialState);

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setValues({
      ...values,
      [event.target.name]: event.target.value,
    });
  };

  const resetForm = () => {
    setValues(initialState);
  };

  return { values, handleChange, resetForm };
};

export default useForm;