import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';

export function activate(context: vscode.ExtensionContext) {
    console.log('Congratulations, your extension "smol-dev-prompt-builder" is now active!');

    let helloWorldDisposable = vscode.commands.registerCommand('smol-dev-prompt-builder.helloWorld', () => {
        vscode.window.showInformationMessage('Hello World from smol-dev-prompt-builder!');
    });
    context.subscriptions.push(helloWorldDisposable);

    let openInterfaceDisposable = vscode.commands.registerCommand('smol-dev-prompt-builder.openInterface', () => {
        const panel = vscode.window.createWebviewPanel(
            'smolDevPromptBuilder',
            'Smol Dev Prompt Builder',
            vscode.ViewColumn.One,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
                localResourceRoots: [
                    vscode.Uri.joinPath(context.extensionUri, 'src')
                ]
            }
        );

        const webviewHtmlPath = vscode.Uri.joinPath(context.extensionUri, 'src', 'webview.html');
        let htmlContent = fs.readFileSync(webviewHtmlPath.fsPath, 'utf8');

        const scriptNonce = getNonce();
        const cssNonce = getNonce(); // Could use the same nonce or a different one

        // Make resource URIs
        const cssPathOnDisk = vscode.Uri.joinPath(context.extensionUri, 'src', 'webview.css');
        const cssUri = panel.webview.asWebviewUri(cssPathOnDisk);

        // Replace placeholders in HTML
        htmlContent = htmlContent.replace(/nonce-webview-script/g, scriptNonce);
        htmlContent = htmlContent.replace(
            /(<link rel="stylesheet" type="text\/css" href=")(webview.css)(">)/,
            `$1${cssUri}$3`
        );

        // Update CSP dynamically
        htmlContent = htmlContent.replace(
            /(<meta http-equiv="Content-Security-Policy" content=")([^"]*)(">)/,
            `$1default-src 'none'; style-src ${panel.webview.cspSource} 'unsafe-inline'; script-src 'nonce-${scriptNonce}'; img-src ${panel.webview.cspSource} data:;$3`
            // Using panel.webview.cspSource for style-src is a safe way to allow linked stylesheets from our extension.
            // 'unsafe-inline' for styles is kept if needed by VS Code themes, but webview.css uses VS Code CSS vars which is preferred.
        );

        panel.webview.html = htmlContent;

        panel.webview.postMessage({ command: 'greeting', text: 'Hello from the extension! Webview loaded.' });

        panel.webview.onDidReceiveMessage(
            message => {
                switch (message.command) {
                    case 'generatePrompt':
                        if (message.data && message.data.projectOverview) {
                            vscode.window.showInformationMessage(`Received project overview: ${message.data.projectOverview.substring(0, 30)}... Processing to prompt.md`);
                        } else {
                            vscode.window.showInformationMessage('Received prompt data from webview. Processing to prompt.md. Check console.');
                        }
                        console.log('[Extension] Received prompt data from webview:', message.data);

                        generatePromptMdFile(message.data);
                        return;
                    case 'webviewReady':
                        console.log('[Extension] Webview signaled ready (optional handling)');
                        return;
                    default:
                        console.log('[Extension] Received unknown command from webview:', message);
                        return;
                }
            },
            undefined,
            context.subscriptions
        );
    });

    context.subscriptions.push(openInterfaceDisposable);
}

export function deactivate() {}

function getNonce() {
    let text = '';
    const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    for (let i = 0; i < 32; i++) {
        text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
}

interface PromptData {
    projectOverview?: string;
    keyFeatures?: string;
    targetTechnologies?: string;
    nonFunctionalRequirements?: {
        performance?: string;
        security?: string;
        logging?: string;
    };
    existingCodeContext?: string;
    fileStructurePreferences?: string;
}

function formatDataToPromptMd(data: PromptData): string {
    let mdContent = '';

    if (data.projectOverview) {
        mdContent += `# Project Overview/Goal\n${data.projectOverview}\n\n`;
    }

    if (data.keyFeatures) {
        mdContent += `## Key Features/User Stories\n`;
        // Assuming one feature per line, convert to bullet points
        const features = data.keyFeatures.split('\n').filter(f => f.trim() !== '');
        features.forEach(feature => {
            mdContent += `- ${feature.trim()}\n`;
        });
        mdContent += `\n`;
    }

    if (data.targetTechnologies) {
        mdContent += `## Target Technologies/Frameworks\n${data.targetTechnologies}\n\n`;
    }

    if (data.nonFunctionalRequirements) {
        let nfrContent = '';
        if (data.nonFunctionalRequirements.performance) {
            nfrContent += `### Performance\n${data.nonFunctionalRequirements.performance}\n\n`;
        }
        if (data.nonFunctionalRequirements.security) {
            nfrContent += `### Security\n${data.nonFunctionalRequirements.security}\n\n`;
        }
        if (data.nonFunctionalRequirements.logging) {
            nfrContent += `### Logging\n${data.nonFunctionalRequirements.logging}\n\n`;
        }
        if (nfrContent) {
            mdContent += `## Non-Functional Requirements\n${nfrContent}`; // The sub-sections already add \n\n
        }
    }

    if (data.existingCodeContext) {
        mdContent += `## Existing Code Context (Optional)\n${data.existingCodeContext}\n\n`;
    }

    if (data.fileStructurePreferences) {
        mdContent += `## File/Folder Structure Preferences (Optional)\n${data.fileStructurePreferences}\n\n`;
    }

    return mdContent.trim();
}

async function generatePromptMdFile(data: PromptData) {
    const markdownContent = formatDataToPromptMd(data);

    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders || workspaceFolders.length === 0) {
        vscode.window.showErrorMessage('No workspace folder open. Please open a workspace to save prompt.md.');
        return;
    }

    const workspaceRoot = workspaceFolders[0].uri;
    const promptFilePath = vscode.Uri.joinPath(workspaceRoot, 'prompt.md');

    try {
        await vscode.workspace.fs.writeFile(promptFilePath, Buffer.from(markdownContent, 'utf8'));
        vscode.window.showInformationMessage(`prompt.md generated successfully at ${promptFilePath.fsPath}`);

        // Optionally open the file
        // vscode.workspace.openTextDocument(promptFilePath).then(doc => {
        //     vscode.window.showTextDocument(doc);
        // });

    } catch (error) {
        console.error('[Extension] Error writing prompt.md file:', error);
        vscode.window.showErrorMessage(`Error generating prompt.md: ${error instanceof Error ? error.message : String(error)}`);
    }
}
