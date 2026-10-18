$repos = @(
    @{ Name = "trl"; Url = "https://github.com/huggingface/trl.git" },
    @{ Name = "unsloth"; Url = "https://github.com/unslothai/unsloth.git" },
    @{ Name = "OpenRLHF"; Url = "https://github.com/OpenRLHF/OpenRLHF.git" },
    @{ Name = "axolotl"; Url = "https://github.com/axolotl-ai-cloud/axolotl.git" },
    @{ Name = "letta"; Url = "https://github.com/letta-ai/letta.git" },
    @{ Name = "dspy"; Url = "https://github.com/stanfordnlp/dspy.git" },
    @{ Name = "transformers"; Url = "https://github.com/huggingface/transformers.git" },
    @{ Name = "peft"; Url = "https://github.com/huggingface/peft.git" },
    @{ Name = "langgraph"; Url = "https://github.com/langchain-ai/langgraph.git" },
    @{ Name = "mem0"; Url = "https://github.com/mem0ai/mem0.git" },
    @{ Name = "OpenHands"; Url = "https://github.com/All-Hands-AI/OpenHands.git" },
    @{ Name = "llama_index"; Url = "https://github.com/run-llama/llama_index.git" },
    @{ Name = "llama.cpp"; Url = "https://github.com/ggml-org/llama.cpp.git" },
    @{ Name = "vllm"; Url = "https://github.com/vllm-project/vllm.git" },
    @{ Name = "Qwen"; Url = "https://github.com/QwenLM/Qwen.git" },
    @{ Name = "Qwen2.5"; Url = "https://github.com/QwenLM/Qwen2.5.git" },
    @{ Name = "OLMo-core"; Url = "https://github.com/allenai/OLMo-core.git" },
    @{ Name = "OLMo"; Url = "https://github.com/allenai/OLMo.git" }
)

$targetDir = "C:\Users\hp\Desktop\EVOLYX"

Write-Host "=== Starting EVOLYX Repository Cloning ==="
Write-Host "Destination: $targetDir"

foreach ($repo in $repos) {
    $dest = Join-Path $targetDir $repo.Name
    $freeGB = [math]::Round((Get-PSDrive C).Free / 1GB, 2)
    Write-Host "`n----------------------------------------"
    Write-Host "Repo: $($repo.Name) ($($repo.Url))"
    Write-Host "Free disk space before clone: $freeGB GB"

    if ($freeGB -lt 0.5) {
        Write-Warning "Disk space critically low (< 0.5 GB). Stopping to protect system."
        break
    }

    if (Test-Path $dest) {
        Write-Host "Already exists at $dest, skipping clone."
    } else {
        Write-Host "Cloning $($repo.Name) with --depth 1..."
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        git clone --depth 1 $repo.Url $dest
        $sw.Stop()

        if ($LASTEXITCODE -eq 0) {
            Write-Host "Successfully cloned $($repo.Name) in $($sw.Elapsed.TotalSeconds.ToString('F1'))s"
        } else {
            Write-Error "Failed to clone $($repo.Name) with exit code $LASTEXITCODE"
        }
    }

    if (Test-Path $dest) {
        $sizeMB = [math]::Round(((Get-ChildItem -Path $dest -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1MB), 2)
        Write-Host "Folder size: $sizeMB MB"
    }
}

$finalFreeGB = [math]::Round((Get-PSDrive C).Free / 1GB, 2)
Write-Host "`n========================================"
Write-Host "All cloning tasks finished. Current free disk space: $finalFreeGB GB"
Write-Host "========================================"
