# Update Process for `shared_deps.json`

This document details the workflow for updating `shared_deps.json` files when a Large Language Model (LLM) proposes new shared elements, expanding on the "Handling Updates and Versioning" section of `llm_prompting_strategy.md`.

The goal is to have a structured and safe process for incorporating LLM-suggested additions into the shared dependencies contract.

## 1. LLM Output of New Definitions

As defined in the `llm_prompting_strategy.md`, the LLM is instructed to output any *new* proposed shared elements in a specific JSON structure. This structure should mirror the relevant parts of `shared_deps.json`.

**Example LLM Output for a new function:**
```json
{
  "functionSignatures": [{
    "name": "calculateDiscount",
    "parameters": [
      {"name": "price", "type": "number"},
      {"name": "discountRate", "type": "number"}
    ],
    "returns": {"type": "number"},
    "description": "Calculates the discounted price."
  }]
}
```

**Example LLM Output for a new data schema:**
```json
{
  "dataSchemas": {
    "ProductReview": {
      "type": "object",
      "properties": {
        "reviewId": {"type": "string"},
        "productId": {"type": "string"},
        "rating": {"type": "number", "minimum": 1, "maximum": 5},
        "comment": {"type": "string"}
      },
      "required": ["reviewId", "productId", "rating"]
    }
  }
}
```
The LLM should only include keys for the types of elements it is proposing (e.g., only `functionSignatures` if proposing a function, not empty keys for `dataSchemas`, `domElementIds`, etc.).

## 2. Extraction of Proposed JSON

The system responsible for interacting with the LLM will be designed to automatically identify and extract this JSON block from the LLM's overall response. This might involve looking for specific markers or assuming the last JSON block in a response is the proposed definitions.

## 3. Validation

Once extracted, the proposed JSON snippet undergoes several validation steps before it can be considered for merging:

*   **Schema Validation**: The extracted JSON is validated against the `shared_deps.schema.json`. This ensures that the LLM's proposal (e.g., a new function signature) adheres to the defined structure (e.g., `name` is a string, `parameters` is an array, etc.).
*   **Naming Conflict Check**: The names of the proposed new elements (e.g., function names, schema names) are checked against the existing definitions in the target `shared_deps.json` file for the relevant context.
    *   A new element must not have the same name as an existing element of the same type (e.g., a new function cannot have the same name as an existing function).
*   **Semantic Plausibility (Optional, Human-Aided)**: While harder to automate, a quick check for semantic plausibility might be done, especially if the merge is automated. Does the description make sense for the name and type?

**Handling Validation Failures:**

*   **Error Logging**: All validation failures are logged with details about the proposed JSON and the reason for failure.
*   **User Notification**: The user or a designated administrator is notified of the failed attempt to update `shared_deps.json`.
*   **LLM Correction (Iterative Approach)**: If the interaction is part of an iterative development loop, the LLM could be informed of the validation error (e.g., "The proposed function 'calculateDiscount' conflicts with an existing function name.") and asked to provide a corrected definition.
*   **Rejection**: If validation fails and no correction is immediately possible, the proposed change is rejected.

## 4. Merge Strategy

For incorporating validated new definitions into the `shared_deps.json` file, a cautious approach is recommended, especially in initial implementations.

*   **User-Confirmed Merge (Recommended)**:
    *   **Process**: After successful validation, the system presents the proposed new definitions clearly to a human reviewer (e.g., the developer working with the LLM, or a designated code owner). This could be a diff view or a simple listing of what's new.
    *   **Pros**: Highest safety, ensures human oversight, prevents accidental or nonsensical additions, allows for nuanced judgment on the utility and naming of new elements.
    *   **Cons**: Slower, requires human intervention.
*   **Automated Merge**:
    *   **Process**: New definitions that pass all validation steps are automatically merged into `shared_deps.json`.
    *   **Pros**: Faster, streamlines the workflow if LLM suggestions are consistently high quality.
    *   **Cons**: Higher risk of introducing unwanted or poorly named shared elements if validation is not perfectly comprehensive. Could lead to "schema bloat."
*   **Hybrid Approach**:
    *   **Process**: Minor additions (e.g., new DOM IDs, perhaps new simple data schemas) might be automated, while more impactful additions like new function signatures (which form part of an API contract) always require user confirmation.
    *   **Pros**: Balances speed with safety.
    *   **Cons**: More complex to implement the conditional logic for automation vs. confirmation.

**Initial Recommendation**: Start with **User-Confirmed Merge**. As confidence in the LLM's proposals and the robustness of the validation process grows, consider moving to a hybrid or more automated approach for certain types of additions.

## 5. File Update

Once approved (either automatically or by a user), the `shared_deps.json` file is programmatically updated:

*   The system parses the existing `shared_deps.json` into a mutable data structure (e.g., a Python dictionary or JavaScript object).
*   The new, validated, and approved definitions are added to the appropriate arrays or objects within this structure.
    *   For `domElementIds`: Add new strings to the array.
    *   For `functionSignatures`, `globalVariables`, `messageNames`: Add new objects to the respective arrays.
    *   For `dataSchemas`: Add new key-value pairs (schema name and schema definition) to the object.
*   The modified data structure is then serialized back into a well-formatted JSON string and overwrites the existing `shared_deps.json` file.

## 6. Versioning

After the `shared_deps.json` file has been successfully updated with new definitions, the `version` field within the JSON object **must** be updated.

*   **Strategy**:
    *   Minor, non-breaking additions (like adding a new function or data schema) could increment a patch or minor version (e.g., "1.0.0" to "1.0.1", or "1.0" to "1.1").
    *   If, in the future, this process also handles modifications that could be breaking, major version increments would be needed (e.g., "1.1.0" to "2.0.0"). (Currently, this document focuses on *new* additions).
*   The specific versioning scheme (e.g., SemVer) should be decided and applied consistently.

## 7. Post-Update Actions

Following a successful update and version change to `shared_deps.json`:

*   **Version Control**: The modified `shared_deps.json` file should be committed to the project's version control system (e.g., Git). This provides a history of changes to the shared dependencies.
*   **Team Notification (if applicable)**: In a team environment, changes to shared dependencies (which act as a contract) should be communicated to relevant team members. This could be via automated notifications from version control or direct communication.
*   **Update Dependent Processes/Tooling**: If any build tools, linters, or other automated processes consume `shared_deps.json`, they might need to be re-run or made aware of the update.
*   **Propagate to LLM Context**: For ongoing LLM sessions, the updated shared dependencies should ideally be reloaded or provided to the LLM to ensure it works with the latest definitions.

This structured update process aims to maintain the integrity and reliability of `shared_deps.json` while allowing for its evolution through LLM-assisted development.
