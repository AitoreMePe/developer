# Maintaining and Updating `shared_deps.json`

This document details the workflow for updating `shared_deps.json` files, particularly when a Large Language Model (LLM) proposes new shared elements. It expands on the "Handling Updates and Versioning" section of `llm_integration_strategy.md`.

The goal is to have a structured and safe process for incorporating LLM-suggested additions and other manual changes into the shared dependencies contract.

## 1. LLM Output of New Definitions

As defined in `llm_integration_strategy.md`, the LLM is instructed to output any *new* proposed shared elements in a specific JSON structure, mirroring relevant parts of `shared_deps.json`.

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

The LLM should only include keys for the types of elements it is proposing.

## 2. Extraction of Proposed JSON

The system interacting with the LLM automatically extracts this JSON block from the LLM's response.

## 3. Validation

Once extracted, the proposed JSON snippet undergoes several validation steps:

*   **Schema Validation**: The extracted JSON is validated against the `shared_deps.schema.json` (detailed in `schema_definition.md`). This ensures structural correctness.
*   **Naming Conflict Check**: Proposed new element names are checked against existing definitions in the target `shared_deps.json` to prevent duplicates within the same category (e.g., two functions with the same name).
*   **Semantic Plausibility (Optional, Human-Aided)**: A check to ensure the proposal makes sense in context.

**Handling Validation Failures:**

*   **Error Logging**: Failures are logged with details.
*   **User Notification**: The user or an administrator is notified.
*   **LLM Correction (Iterative)**: The LLM might be asked to correct its proposal based on the validation error.
*   **Rejection**: If uncorrected, the proposal is rejected.

## 4. Merge Strategy

A cautious approach is recommended for incorporating new definitions:

*   **User-Confirmed Merge (Recommended)**:
    *   **Process**: Validated proposals are presented to a human reviewer. This could be a diff view or a list of new items.
    *   **Pros**: Maximum safety, human oversight, prevents erroneous or unhelpful additions.
    *   **Cons**: Slower, requires human intervention.
*   **Automated Merge**:
    *   **Process**: Validated proposals are automatically merged.
    *   **Pros**: Faster, streamlines workflow.
    *   **Cons**: Higher risk of introducing unwanted elements if validation isn't exhaustive.
*   **Hybrid Approach**:
    *   **Process**: Minor additions might be automated; significant ones (like new function APIs) require confirmation.
    *   **Pros**: Balances speed and safety.
    *   **Cons**: More complex logic.

**Initial Recommendation**: Start with **User-Confirmed Merge**. Evaluate moving to more automation as confidence grows. Manual additions or modifications by developers should also undergo review, possibly through a pull request process.

## 5. File Update

Once approved, `shared_deps.json` is programmatically updated:

1.  Parse the existing `shared_deps.json` into a mutable structure.
2.  Add the new, validated, and approved definitions to the correct arrays or objects.
3.  Serialize the structure back to JSON and overwrite the file.

## 6. Versioning

After `shared_deps.json` is updated, its `version` field **must** be updated.

*   **Strategy**: Use a consistent scheme like Semantic Versioning (SemVer).
    *   Non-breaking additions (new function, new schema): Increment patch or minor version (e.g., "1.0.0" to "1.0.1", or "1.0" to "1.1").
    *   Breaking changes (modifying existing signatures, removing elements): Increment major version (e.g., "1.1.0" to "2.0.0").
*   The choice of version increment should reflect the nature of the change.

## 7. Post-Update Actions

*   **Version Control**: Commit the updated `shared_deps.json` (and its corresponding `shared_deps.schema.json` if that also changed) to version control (e.g., Git).
*   **Team Notification**: Communicate changes to relevant team members, especially for breaking changes.
*   **Update Dependent Processes**: CI/CD pipelines, linters, or other tools consuming `shared_deps.json` might need to be updated or re-run.
*   **Propagate to LLM Context**: Ensure LLMs use the latest definitions for future tasks.

## Manual Updates

While this document focuses on LLM-proposed changes, developers may also manually update `shared_deps.json`. These changes should ideally follow a similar review and validation process (e.g., via pull requests that are linted against the schema and reviewed for semantic correctness and necessity).

This structured update process helps maintain the integrity, reliability, and usefulness of `shared_deps.json` as a central contract for shared software elements.
