# Smol Developer - Feature: Support for Local LLMs (Ollama/LMStudio)

This document outlines the functionality for enabling `smol_dev` to utilize Large Language Models (LLMs) running locally via providers like Ollama and LM Studio.

## 1. Core Concept & User Benefit

*   **Objective:** To allow `smol_dev` to interface with LLMs hosted on the user's local machine, typically through local servers provided by tools such as Ollama or LM Studio. This provides an alternative to relying solely on cloud-based LLM APIs.
*   **User Benefits:**
    *   **Cost Savings:** Eliminates API call costs associated with proprietary cloud LLMs.
    *   **Privacy:** Code and prompts are processed locally, offering enhanced data privacy as information does not leave the user's machine.
    *   **Access to Open-Weight Models:** Users can leverage a wide variety of open-source and fine-tuned models available for local execution.
    *   **Offline Capability:** Potential for `smol_dev` to function without an active internet connection (once models are downloaded).
    *   **Customization & Control:** Users have more control over the models they use and their configurations.

## 2. Interaction Mechanism

The primary approach will be to leverage the OpenAI-compatible API endpoints that many local LLM servers provide.

*   **Ollama:**
    *   Ollama typically exposes an OpenAI-compatible API endpoint, often at `http://localhost:11434/v1/`.
    *   This endpoint supports standard paths like `/chat/completions`.
    *   `smol_dev` can use its existing OpenAI API call logic (or a slightly modified version) by directing requests to this local base URL.
*   **LM Studio:**
    *   LM Studio also provides a local server that is often OpenAI-compatible (e.g., `http://localhost:1234/v1/`).
    *   The interaction would be very similar to Ollama, involving API requests to the chat completions endpoint.
*   **Leveraging Existing Libraries:**
    *   The goal is to continue using robust OpenAI client libraries (like Python's `openai` library).
    *   This is typically achievable by configuring the `base_url` (or `api_base`) parameter of the client to point to the local server's address.

## 3. Configuration Changes in `smol_dev`

Users will need to configure `smol_dev` to use a local LLM provider and its specific settings.

*   **Specifying Local LLM Provider:**
    *   A new CLI flag, e.g., `--llm_provider ollama` or `--llm_provider lmstudio`.
    *   Alternatively, a more generic `--model_source local` combined with other flags.
    *   The existing `--model` flag could be used to specify the local model name, e.g., `--model ollama/mistral:7b-instruct-q4_K_M` or simply `mistral:7b-instruct-q4_K_M` if the provider is already set.
*   **Configuring API Endpoint:**
    *   A new CLI flag: `--api_base_url <URL>` (e.g., `http://localhost:11434/v1`).
    *   This could default to common ports if a provider is specified (e.g., `http://localhost:11434/v1` if `--llm_provider ollama`).
*   **API Key Handling:**
    *   Local LLM servers usually do not require an API key.
    *   `smol_dev`'s API key handling logic should allow for an empty or placeholder key when a local provider is selected (e.g., the OpenAI client can often be initialized with `api_key="local"` or similar).
*   **Model Name Mapping:**
    *   The `--model` parameter in `smol_dev` should directly accept the model name as it is recognized by the local LLM server (e.g., `mistral:instruct`, `codellama:13b`, `lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF`).
    *   No complex mapping within `smol_dev` should be necessary; the user is responsible for providing a valid model name for their local server.

## 4. Model Selection & Capabilities

*   **Model Selection:** The user is responsible for selecting and specifying the exact model name available on their local server using the `--model` flag. `smol_dev` will pass this name directly in the API request.
*   **Varying Capabilities:**
    *   It's crucial to acknowledge that local models (especially smaller quantized versions) can vary significantly in their instruction-following capabilities, code generation quality, and context window sizes compared to large proprietary models like GPT-4.
    *   `smol_dev`'s existing prompts, which are likely tuned for more capable models, might require adjustments or simplification by the user to achieve optimal results with certain local LLMs.
    *   The quality of output will be highly dependent on the chosen local model and its configuration.

## 5. Workflow Integration

The choice of a local LLM should integrate smoothly into the existing `smol_dev` workflow.

*   The core functions like `plan`, `specify_file_paths`, and `generate_code_sync` (or their equivalents that make LLM calls) would internally use the configured local LLM client instead of the default OpenAI client.
*   The main change is at the point where the LLM client is initialized and API calls are made. The rest of the logic (prompt construction, file writing, etc.) should remain largely unaffected.

## 6. Potential Challenges

*   **Performance:** Local LLMs can be slower than cloud-based APIs, especially on consumer hardware, potentially increasing `smol_dev`'s overall execution time.
*   **User Setup:** Requires the user to have correctly installed and configured Ollama/LM Studio and downloaded the desired models. `smol_dev` will not manage the local server itself.
*   **Model Quality & Instruction Following:** Output quality and adherence to complex instructions can be inconsistent across different local models. Users may need to experiment.
*   **Error Handling:**
    *   Robust error handling for local server issues: server not running, incorrect endpoint, model not found on the server, model failing to load, request timeouts.
    *   Clear feedback to the user if `smol_dev` cannot connect to or get a valid response from the local LLM.
*   **Context Window Limitations:** Local models often have smaller context windows, which might impact `smol_dev`'s ability to handle large prompts or maintain coherence in extensive projects without advanced context management strategies.

## 7. Key `smol_dev` Code Areas for Modification

*   **LLM Client Instantiation:** The primary area for modification will be where the OpenAI (or other LLM) client is initialized. This will need to be updated to accept a custom `base_url` and handle potentially absent API keys.
*   **Configuration Parsing:** Logic that parses CLI flags or configuration files to read the new settings (`--llm_provider`, `--api_base_url`, potentially modified `--model` behavior).
*   **API Call Logic:** While aiming to use the standard OpenAI library features, some error handling or request parameter adjustments might be needed for compatibility with local servers.

## 8. Security Considerations

*   **User Responsibility:** Users are responsible for the security and origin of the models they download and run locally via Ollama, LM Studio, or similar tools. `smol_dev` itself is not responsible for the content or behavior of these third-party models.
*   **Local Network Exposure:** If the local LLM server is configured to be accessible on the network, users should be aware of the implications. `smol_dev` will typically connect to `localhost` by default.

By supporting local LLMs, `smol_dev` can become more versatile, accessible, and appealing to a broader range of users with different priorities regarding cost, privacy, and model choice.
