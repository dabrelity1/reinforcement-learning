param(
  [Parameter(Mandatory=$true)][string]$RepoName,
  [string]$Private = 'true'
)

# Requires: gh CLI logged in (gh auth login)
# Usage:  .\scripts\seed_github.ps1 -RepoName "reinforcement-learning" -Private true

$ErrorActionPreference = 'Stop'

# Initialize git if needed
if (-not (Test-Path ".git")) {
  git init
  git add .
  git commit -m "chore: project scaffolding"
}

# Create repo on GitHub
$vis = if ($Private -eq 'true') { '--private' } else { '--public' }
& gh repo create $RepoName $vis --source "." --remote origin --push

# Create labels used by templates
& gh label create task -c '#0366d6' -d 'Small actionable task' -f
& gh label create copilot -c '#0e8a16' -d 'For Copilot Agent' -f

# Seed issues from docs/COPILOT_AGENT_ISSUES.md (simple splitter by headings)
$issues = @()
$lines = Get-Content "docs/COPILOT_AGENT_ISSUES.md"
$title = $null
$body = ""
foreach ($line in $lines) {
  if ($line -match '^[0-9]+\. ') {
    if ($title) { $issues += ,@($title, $body.Trim()) ; $body = "" }
    $title = $line -replace '^[0-9]+\. ', ''
  } else {
    $body += ($line + "`n")
  }
}
if ($title) { $issues += ,@($title, $body.Trim()) }

foreach ($it in $issues) {
  $t = $it[0]
  $b = $it[1]
  & gh issue create --title "[Task] $t" --label "task,copilot" --body $b
}

Write-Host "Repository created and issues seeded."
