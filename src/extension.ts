import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
    console.log('Congratulations, your extension "smol-dev-prompt-builder" is now active!');

    let disposable = vscode.commands.registerCommand('smol-dev-prompt-builder.helloWorld', () => {
        vscode.window.showInformationMessage('Hello World from smol-dev-prompt-builder!');
    });

    context.subscriptions.push(disposable);
}

export function deactivate() {}
