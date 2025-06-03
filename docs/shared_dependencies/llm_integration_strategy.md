# LLM Integration Strategy for `shared_deps.json`

This document outlines a strategy for effectively using `shared_deps.json` to guide Large Language Models (LLMs) in code generation and modification tasks. The goal is to ensure the LLM leverages and respects these shared definitions, promoting consistency and reducing redundancy.

## 1. Contextual Loading of `shared_deps.json`

**Determining the Correct `shared_deps.json`:**

*   **Proximity-Based Loading**: The primary method will be to look for a `shared_deps.json` file in the same directory as the file currently being processed by the LLM. If not found, the system will traverse up the directory tree, using the first `shared_deps.json` it encounters. This allows for module-specific shared definitions while also permitting a global fallback.
*   **Project Configuration**: For more complex projects, a configuration file at the project root could explicitly map directories or modules to specific `shared_deps.json` files.
*   **Explicit User/System Directive**: In some cases, the specific `shared_deps.json` to use might be explicitly provided as part of the task given to the LLM.

**When to Load:**

*   **Session Start**: The relevant `shared_deps.json` should be loaded and parsed at the beginning of any code generation or modification session that pertains to a specific module or file group covered by that shared dependencies file.
*   **Context Switch**: If the LLM's focus shifts to a different module or part of the codebase that might be governed by a different `shared_deps.json`, the new context's dependencies should be loaded.

## 2. Injecting Shared Definitions into the Prompt

**Formatting for Prompts:**

The information from `shared_deps.json` needs to be presented to the LLM in a clear, concise, and easily parsable format. The goal is to provide necessary context without overwhelming the LLM or consuming too much of the context window.

**Context Window Considerations & Compact Representation:**

*   **Relevance Filtering (Recommended)**: Instead of injecting the entire `shared_deps.json` content, it's often better to inject only the definitions relevant to the current task.
    *   For example, if the task is to implement a UI interaction, relevant `domElementIds` and `messageNames` might be prioritized.
    *   If the task is to write business logic, relevant `functionSignatures` and `dataSchemas` are more critical.
    *   Heuristics or even a preliminary LLM call could determine relevance.
*   **Compact Formats**:
    *   **DOM Element IDs**: `Available DOM IDs: [userPrompt, sendButton, loadingIndicator].`
    *   **Function Signatures**: `Shared functions: getUser(id:string):UserObject, submitForm(data:FormData):Promise<boolean>. Use these where appropriate.`
    *   **Message Names**: `System messages: USER_LOGIN, DATA_UPDATED. Listen for or dispatch these as needed.`
    *   **Data Schemas**: For complex schemas, provide the name and a brief description or key fields. `Data models: UserProfile (fields: id, name, email), Order (fields: orderId, items, total).` If a specific schema is central to the task, its full structure might be included.

**Example Prompt Snippets (General Context):**

```
You are an AI assistant helping to write JavaScript code for a web application.
The following shared definitions are available. Adhere to them strictly.

Available DOM IDs: [userPrompt, stylePrompt, maxTokens, sendButton, content, loadingIndicator].
Shared functions:
- requestAnthropicSummary(promptText:string, stylePrompt:string, maxTokens:number):Promise<string>
- storePageContent(pageData:pageContent):void
System messages:
- storePageContent
- getPageContent
Data schemas:
- pageContent: { title: string, content: string }

Your task is to [LLM's specific task, e.g., "implement the event listener for the 'sendButton'..."].
Only use the DOM IDs, functions, and message names listed above. Do not define new ones unless explicitly asked.
```

## 3. Instructing the LLM on Usage

**Guiding Phrases:**

*   "Utilize the provided shared `domElementIds` for any DOM manipulations."
*   "When you need to perform [specific action, e.g., 'fetch user data'], use the shared function `getUserProfile`."
*   "Ensure all data structures passed to `storePageContent` conform to the `pageContent` schema."
*   "Dispatch the `DATA_UPDATED` message when the operation is complete."

**Preventing Redefinition or Undefined Usage:**

*   "**Crucially, do not redefine any elements (functions, DOM IDs, messages, schemas) listed above.** Use them as they are defined."
*   "Only use identifiers (DOM IDs, function names, message names, schema names) that are explicitly provided in the shared definitions section of this prompt."
*   "If a required functionality or identifier is missing from the shared definitions, state that it is missing rather than inventing a new one."

## 4. Instructing the LLM on Defining *New* Shared Elements

**Identifying the Need for New Shared Elements:**

*   "As you develop the solution, if you identify a piece of data (a data structure), a function, or a UI element ID that is likely to be reused in other parts of the application, or that facilitates communication between modules, please flag it as a candidate for a new shared dependency."
*   "If the current task requires a function or data schema that is not defined in the provided shared dependencies but seems generally useful for other components, propose its definition."

**Instructing for Definition Output:**

*   "If you propose a new shared function, provide its signature including: name, parameters (each with name and type), and return type."
*   "If you propose a new shared data schema, define its structure, including field names and their types. If possible, provide this as a JSON schema snippet."
*   "For new DOM IDs or message names, simply list the proposed names and a brief description of their purpose."
*   "**Format your proposed new shared definitions as a JSON object suitable for direct inclusion in the `shared_deps.json` file under the appropriate key (e.g., `functionSignatures`, `dataSchemas`).**"

**Example Prompt Snippets for New Definitions:**

```
Your task is to [LLM's specific task].
While implementing this, if you determine that a new function, data schema, DOM ID, or message name is needed AND it would be beneficial for other parts of the application (i.e., it should be shared),
then:
1. Implement the current task using your proposed new shared element.
2. At the end of your response, provide the definition for this new shared element in a JSON format like this:
   {
     "functionSignatures": [ /* your new function here */ ],
     "dataSchemas": { /* your new schema here */ },
     "domElementIds": [ /* your new DOM ID here */ ],
     "messageNames": [ /* your new message name here */ ]
   }
   Only include the key (`functionSignatures`, `dataSchemas`, etc.) relevant to the new element(s) you are defining.
   For example, if you define a new function 'calculateTotal(items:Array):number', output:
   {
     "functionSignatures": [{ "name": "calculateTotal", "parameters": [{"name": "items", "type": "Array"}], "returns": {"type": "number"}, "description": "Calculates total from line items." }]
   }
```

## 5. Handling Updates and Versioning

**Process for Proposed New Definitions:**

1.  **LLM Output**: The LLM outputs the code for the current task and, if prompted, a separate JSON snippet containing definitions for any *new* shared elements it identified as necessary.
2.  **System Validation**:
    *   The system (or a human reviewer) parses the JSON snippet.
    *   It validates the proposed definitions against the project's `shared_deps.schema.json`.
    *   It checks for naming conflicts with existing definitions.
3.  **Merging**: If valid and approved, the new definitions are merged into the appropriate `shared_deps.json` file. See `maintenance_and_updates.md` for details.
4.  **Versioning**: After significant additions or modifications to `shared_deps.json`, its `version` field should be updated.
5.  **Communication**: Changes to shared dependencies should be communicated to teams or other LLM instances that might be affected.

This strategy aims to make the LLM a more integrated and context-aware coding assistant, leveraging shared knowledge to build more robust and maintainable applications.
