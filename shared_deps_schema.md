# Shared Dependencies Schema (`shared_deps.json`)

This document outlines the structure of `shared_deps.json`, a file used to define shared dependencies between different components or modules of a system.

## Top-Level Keys

The `shared_deps.json` file is a JSON object with the following top-level keys:

### 1. `domElementIds`
- **Type**: `Array` of `String`
- **Description**: This array lists the `id` attributes of DOM elements that are considered shared resources or are frequently accessed/manipulated by different parts of the application.
- **Example**:
  ```json
  "domElementIds": [
    "main-container",
    "user-profile-modal",
    "notification-badge"
  ]
  ```

### 2. `functionSignatures`
- **Type**: `Array` of `Object`
- **Description**: Defines a list of functions that are shared or can be called across different modules. Each object in the array represents a single function and has the following structure:
    - `name` (String, Mandatory): The name of the function.
    - `description` (String, Optional): A brief explanation of what the function does.
    - `parameters` (Array of Object, Mandatory): Describes the parameters the function accepts. Each parameter object has:
        - `name` (String, Mandatory): The name of the parameter.
        - `type` (String, Mandatory): The data type of the parameter (e.g., "string", "number", "boolean", "object", "MyCustomType").
        - `description` (String, Optional): A brief explanation of the parameter.
    - `returns` (Object, Mandatory): Describes the value returned by the function. It has:
        - `type` (String, Mandatory): The data type of the return value.
        - `description` (String, Optional): A brief explanation of what is returned.
- **Example**:
  ```json
  "functionSignatures": [
    {
      "name": "getUserProfile",
      "description": "Fetches user profile data from the server.",
      "parameters": [
        {
          "name": "userId",
          "type": "string",
          "description": "The unique identifier of the user."
        }
      ],
      "returns": {
        "type": "object",
        "description": "An object containing user profile information."
      }
    }
  ]
  ```

### 3. `globalVariables`
- **Type**: `Array` of `Object`
- **Description**: Lists global variables that are shared across the system. Each object in the array defines a single global variable:
    - `name` (String, Mandatory): The name of the global variable.
    - `type` (String, Mandatory): The data type of the variable.
    - `description` (String, Optional): A brief explanation of the variable's purpose.
- **Example**:
  ```json
  "globalVariables": [
    {
      "name": "CURRENT_USER_LOCALE",
      "type": "string",
      "description": "Stores the active locale for the current user (e.g., 'en-US')."
    }
  ]
  ```

### 4. `messageNames`
- **Type**: `Array` of `Object`
- **Description**: Defines a list of message names or event types used for inter-component communication (e.g., via a pub/sub system). Each object represents a message:
    - `name` (String, Mandatory): The unique name of the message or event.
    - `payloadSchema` (Object, Optional): A JSON Schema object that defines the structure of the payload accompanying this message. This helps ensure consistency in message data.
    - `description` (String, Optional): A brief explanation of when and why this message is dispatched and what it signifies.
- **Example**:
  ```json
  "messageNames": [
    {
      "name": "USER_LOGGED_IN",
      "payloadSchema": {
        "type": "object",
        "properties": {
          "userId": { "type": "string" },
          "timestamp": { "type": "string", "format": "date-time" }
        },
        "required": ["userId", "timestamp"]
      },
      "description": "Fired when a user successfully logs into the system."
    }
  ]
  ```

### 5. `dataSchemas`
- **Type**: `Object`
- **Description**: A collection of named data schemas. The keys are string names identifying the schema, and the values can either be:
    - A full JSON Schema `Object`: Defining the structure, types, and constraints for a specific data model.
    - A `String`: A textual description or reference to a schema defined elsewhere.
- **Example**:
  ```json
  "dataSchemas": {
    "UserProfile": {
      "type": "object",
      "properties": {
        "id": { "type": "string" },
        "username": { "type": "string" },
        "email": { "type": "string", "format": "email" },
        "preferences": { "$ref": "#/dataSchemas/UserPreferences" }
      },
      "required": ["id", "username", "email"]
    },
    "UserPreferences": "Schema for user-specific application preferences, including theme and notification settings.",
    "ProductRecord": {
      "type": "object",
      // ... more schema definition
    }
  }
  ```

### 6. `version`
- **Type**: `String`
- **Description**: A version string for this `shared_deps.json` schema itself. This allows for tracking changes and evolution of the shared dependencies contract.
- **Example**:
  ```json
  "version": "1.0.1"
  ```

### 7. `notes`
- **Type**: `String`
- **Description**: An optional field for any general comments, explanations, or outstanding issues related to the shared dependencies defined in this file.
- **Example**:
  ```json
  "notes": "This version of shared dependencies is aligned with the Q3 feature release. The 'ProductRecord' schema is still under review."
  ```

## Integrity and Usage

This `shared_deps.json` file serves as a contract. Components relying on these shared elements should conform to the definitions provided herein to ensure interoperability and reduce integration issues. When a shared dependency is updated (e.g., a function signature changes), this file should be updated accordingly, and consuming components should be checked for compatibility.
