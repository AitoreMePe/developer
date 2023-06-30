# Starter React App

This is a starter React application built with Typescript and Firebase Authentication.

## Getting Started

First, clone the repository to your local machine:

```bash
git clone <repository-url>
```

Then, install the dependencies:

```bash
npm install
```

## Running the App

To start the development server:

```bash
npm start
```

The app will be available at `http://localhost:3000`.

## Features

- User Authentication: Users can sign up and log in using their email and password.

## Project Structure

- `src/index.tsx`: The entry point of the application.
- `src/App.tsx`: The main App component.
- `src/components`: Contains the Login, Signup, and Dashboard components.
- `src/services/auth.ts`: Contains the authentication service.
- `src/types/user.ts`: Defines the User type.
- `src/styles`: Contains global and component-specific styles.
- `src/utils`: Contains utility functions and hooks.
- `public`: Contains public assets like the favicon and logos.
- `package.json`: Defines the project's npm dependencies.
- `tsconfig.json`: Configures Typescript.
- `.env`: Defines environment variables.
- `.gitignore`: Specifies files to ignore in git.
- `README.md`: This file.

## Environment Variables

You need to set the following environment variables in the `.env` file:

- `REACT_APP_FIREBASE_API_KEY`
- `REACT_APP_FIREBASE_AUTH_DOMAIN`
- `REACT_APP_FIREBASE_PROJECT_ID`
- `REACT_APP_FIREBASE_STORAGE_BUCKET`
- `REACT_APP_FIREBASE_MESSAGING_SENDER_ID`
- `REACT_APP_FIREBASE_APP_ID`

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)