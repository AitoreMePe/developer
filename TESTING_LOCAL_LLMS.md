# Testing Local LLM Support in smol_dev (Ollama & LM Studio)

## 1. Objective

This document provides guidance for manually testing and verifying that `smol_dev` can successfully utilize locally hosted Large Language Models (LLMs) via Ollama and LM Studio. These steps also serve as a basis for user documentation on setting up and using this feature.

## 2. Prerequisites

*   **`smol_dev` Updated:** Ensure you have the version of `smol_dev` that includes the CLI arguments for local LLM support (`--llm_provider`, `--api_base_url`, `--api_key`, and updated `--model` usage).
*   **Ollama Installed (for Ollama testing):**
    *   Install Ollama from [ollama.ai](https://ollama.ai/).
    *   Pull a model, for example: `ollama pull mistral:7b-instruct-q4_K_M` or any other model you wish to test.
*   **LM Studio Installed (for LM Studio testing):**
    *   Install LM Studio from [lmstudio.ai](https://lmstudio.ai/).
    *   Download a model within LM Studio (e.g., a Llama GGUF model).
    *   Ensure the model is loaded and the local server is started within LM Studio.

## 3. Testing with Ollama

*   **Ensure Ollama Server is Running:**
    *   Ollama usually runs as a background service after installation. You can check its status or start it manually if needed (often `ollama serve`, though this is typically not required for desktop installations).
    *   The default API endpoint for Ollama is `http://localhost:11434`. The OpenAI-compatible endpoint is usually at `/v1` under this base.

*   **Construct `smol_dev` Command:**
    Open your terminal and run a command similar to the following, replacing `<your-ollama-model-name>` with the name of the model you pulled (e.g., `mistral:7b-instruct-q4_K_M` or `llama3:8b-instruct-q4_K_M`).

    ```bash
    python smol_dev/main.py \
      --prompt "a very simple python script that prints 'Hello, Local LLM!'" \
      --llm_provider ollama \
      --api_base_url http://localhost:11434/v1 \
      --model mistral:7b-instruct-q4_K_M \
      --debug True
    ```
    *Note: For some Ollama setups or if you are using the `openai` python library v1.x+, ensure your `--api_key` is set, even if to a dummy value like `local` or `NA` if Ollama doesn't require one. `smol_dev` attempts to set a placeholder key "local" if `llm_provider` is `ollama` and no explicit key is given.*


*   **Expected Outcome:**
    *   `smol_dev` should start processing the prompt.
    *   With `--debug True`, you should see log messages, including lines indicating the API base URL being used.
    *   Code files (e.g., `generated/script.py`, `generated/shared_deps.md`) should be generated.
    *   Check the Ollama server logs. If Ollama is run from a terminal, you should see requests coming in when `smol_dev` makes API calls.

*   **Troubleshooting Tips:**
    *   **Ollama Server Not Running:** Ensure Ollama is running. Try `ollama list` in your terminal to see available models; this also indicates if the service is responsive.
    *   **Incorrect Model Name:** Double-check the model name matches exactly what `ollama list` shows.
    *   **Incorrect URL:** Verify the `--api_base_url`.
    *   **Firewall Issues:** Ensure your firewall isn't blocking local connections to port 11434.
    *   **API Key (if applicable):** While `smol_dev` sets a placeholder, if your Ollama instance or a proxy is configured to expect a key, provide it via `--api_key`.

## 4. Testing with LM Studio

*   **Start LM Studio Server:**
    *   Open LM Studio.
    *   Go to the "Local Server" tab (usually looks like `<->`).
    *   Select a loaded model from the dropdown at the top.
    *   Click "Start Server".
    *   Note the server address displayed (e.g., `http://localhost:1234`). The OpenAI-compatible endpoint is typically at `/v1`.

*   **Construct `smol_dev` Command:**
    Replace `<your-lm-studio-model-identifier>` with the actual model identifier used by LM Studio (this might be the filename or a name shown in the UI). It's often something like `local-model` if you haven't changed the default, or the path/name of the GGUF file.

    ```bash
    python smol_dev/main.py \
      --prompt "a very simple javascript function that returns the sum of two numbers" \
      --llm_provider lmstudio \
      --api_base_url http://localhost:1234/v1 \
      --model local-model \
      --debug True
    ```
    *Note: LM Studio also typically doesn't require an API key. `smol_dev` will use a placeholder "local" if `--api_key` is not set and `--llm_provider` is `lmstudio`.*

*   **Expected Outcome:**
    *   `smol_dev` processes the prompt.
    *   Debug output should show the custom API base URL.
    *   Generated files should appear in the `generated` folder.
    *   The LM Studio server UI should show incoming requests and responses in its log panel.

*   **Troubleshooting Tips:**
    *   **LM Studio Server Not Started:** Ensure you've clicked "Start Server" in LM Studio.
    *   **Model Not Loaded/Selected:** Make sure a model is selected and loaded in the LM Studio server tab.
    *   **Incorrect URL/Port:** Verify the URL and port from the LM Studio server tab.
    *   **Model Identifier:** The `--model` value must match what LM Studio expects. This can sometimes be just `local-model` or a more specific identifier if you have multiple models served.
    *   **Firewall Issues:** Ensure local connections to the specified port (e.g., 1234) are not blocked.

## 5. General Verification Points

*   **File Generation:** Check that `generated/shared_deps.md` and the relevant code file(s) (e.g., `script.py`, `script.js`) are created in the output folder.
*   **Code Quality:** Review the generated code. Acknowledge that the quality will heavily depend on the capability of the chosen local LLM. Simple prompts are better for initial testing.
*   **API Connectivity:** Ensure there are no errors related to API key issues (unless expected due to your local server setup) or connection failures in the `smol_dev` console output. Debug logs should indicate attempts to connect to the specified local URL.
*   **Content of `shared_deps.md`:** Verify that it contains a plausible plan, even if simple.

## 6. Notes for User Documentation

When documenting this feature for users, include:

*   **User Responsibility:** Emphasize that the user is responsible for installing, configuring, and running their chosen local LLM server (Ollama, LM Studio, etc.) and downloading models. `smol_dev` only acts as a client to these servers.
*   **Model Naming:** Stress the importance of using the correct model name in the `--model` argument, exactly as it's identified by the local LLM server.
*   **Prompting Differences:** Remind users that smaller or specialized local models might require different, possibly simpler or more explicit, prompting strategies compared to large cloud-based models like GPT-4. Results will vary based on model capability.
*   **API Base URL:** Clearly explain how to find the correct `--api_base_url` for their specific local server (including the `/v1` path for OpenAI compatibility).
*   **API Key:** Explain that for most local setups, an API key is not needed, and `smol_dev` will use a placeholder. If their server *is* configured to require a key, they should use the `--api_key` argument.
*   **Troubleshooting:** Include basic troubleshooting steps similar to those above.

This testing guide should help confirm the local LLM integration and pave the way for user documentation.
