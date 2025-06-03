# Shared Dependency Management System

## Introduction

The `shared_deps.json` files are a core part of our strategy for ensuring consistency, reducing redundancy, and improving collaboration when developing software, especially when leveraging Large Language Models (LLMs) for code generation or modification.

These files serve as a central "source of truth" for definitions of shared software elements that are used across different modules or components of a project. This includes:

*   DOM Element IDs
*   Reusable Function Signatures
*   Global Variables
*   Standardized Message Names (for pub/sub or eventing systems)
*   Common Data Schemas

By defining these elements in a structured way, we provide LLMs with clear context, enabling them to generate code that aligns with existing patterns and contracts, rather than inventing new, potentially conflicting, definitions.

## Overview of the System

**What is `shared_deps.json`?**

It's a JSON file that explicitly lists and defines shared elements within a specific scope (e.g., a module, a project). The structure of this file is formally defined by a JSON schema (`shared_deps.schema.json`).

**How it helps when working with LLMs:**

1.  **Context Provision**: Provides LLMs with a clear understanding of available shared resources.
2.  **Consistency**: Encourages LLMs to use pre-defined names and structures, leading to more consistent code.
3.  **Reduced Redundancy**: Prevents LLMs from redefining already existing functions or variables.
4.  **Improved Collaboration**: Acts as a contract that both human developers and LLMs adhere to.
5.  **Easier Maintenance**: Centralizes definitions, making updates and refactoring more manageable.

## Key Documentation

For a comprehensive understanding of the shared dependency management system, please refer to the following documents:

*   **[Schema Definition (`schema_definition.md`)](./schema_definition.md)**: Details the exact structure, fields, and types used within a `shared_deps.json` file. This is essential for anyone creating or modifying these files.
*   **[LLM Integration Strategy (`llm_integration_strategy.md`)](./llm_integration_strategy.md)**: Explains how `shared_deps.json` files are loaded, processed, and injected into LLM prompts. It also covers how LLMs are instructed to use these definitions and propose new ones.
*   **[Maintenance and Updates (`maintenance_and_updates.md`)](./maintenance_and_updates.md)**: Outlines the process for updating `shared_deps.json` files, including validating and merging new definitions proposed by LLMs or added manually by developers.

## Example File

An example of a populated `shared_deps.json` file can be found in the codebase at:
`v0/code2prompt2code/shared_deps.json`

*(Note: The actual path to example files may vary per project setup.)*

## How to Use

**For Developers:**

*   **Creating `shared_deps.json`**: When starting a new module or identifying a set of commonly used elements within an existing project that are not yet tracked, consider creating a `shared_deps.json` file in the root directory of that module/project.
*   **Scope**: A `shared_deps.json` file typically governs the shared elements within its own directory and any subdirectories that do not have their own `shared_deps.json` file.
*   **Consulting**: Before defining a new function, global variable, or data structure that you suspect might be used elsewhere, check the relevant `shared_deps.json` to see if a suitable definition already exists.
*   **LLM Interaction**: When working with an LLM on code that should use shared elements, ensure the LLM is made aware of the relevant `shared_deps.json` content (our tooling aims to automate this based on the [LLM Integration Strategy](./llm_integration_strategy.md)).

**Importance of Keeping it Updated:**

`shared_deps.json` is a living document. As the project evolves, so will its shared elements. It's crucial to:
*   Add new shared elements as they are identified and agreed upon.
*   Update existing definitions if their signatures or structures change (this often requires careful consideration of backward compatibility and versioning).
*   Remove definitions that are no longer used.

Following the processes outlined in [Maintenance and Updates](./maintenance_and_updates.md) will help ensure the integrity and usefulness of these files.
An up-to-date `shared_deps.json` significantly enhances the effectiveness of LLM-assisted development and promotes better code quality overall.
