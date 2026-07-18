# Run AFTER: gh auth login
# Creates GitHub repo and pushes all committed code

$ErrorActionPreference = "Stop"
Set-Location "d:\Data analysis with AI"

Write-Host "Checking git status..."
git status

Write-Host "Adding all files ( .env is ignored )..."
git add -A
git status

if (git status --porcelain) {
    git commit -m "Update project files before GitHub push."
}

$repoName = "ai-data-analysis-assistant"
Write-Host "Creating GitHub repo: $repoName"
gh repo create $repoName --public --source=. --remote=origin --push

Write-Host "Done! Repo URL:"
gh repo view --web 2>$null
git remote -v
