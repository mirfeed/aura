# Управление Аурой: start | stop | restart | status | log | unload
# Вызывается из aura.bat (меню), можно и напрямую:
#   powershell -ExecutionPolicy Bypass -File scripts\bot.ps1 restart
param([Parameter(Position = 0)][string]$Action = "status")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Flag = Join-Path $Root "data\stop.flag"
$Log = Join-Path $Root "logs\bot.log"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Get-Bot {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match 'main\.py' }
}

function Test-Url($url) {
    try { Invoke-RestMethod -Uri $url -TimeoutSec 2 | Out-Null; return $true } catch { return $false }
}

function Start-Bot {
    if (Get-Bot) { Write-Host "Аура уже работает." -ForegroundColor Yellow; return }
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$Root\run_bot.bat`"" -WorkingDirectory $Root
    Write-Host "Запускаю... (окно «Aura bot» — не закрывай его, это и есть бот)"
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 1
        if ((Get-Content $Log -Tail 5 -Encoding UTF8 -ErrorAction SilentlyContinue) -match '"Bot ready"') {
            Write-Host "Аура в сети." -ForegroundColor Green; return
        }
    }
    Write-Host "За минуту не дождалась «Bot ready» — посмотри лог (пункт меню «Лог»)." -ForegroundColor Yellow
}

function Stop-Bot {
    if (-not (Get-Bot)) { Write-Host "Аура и так не запущена."; return }
    # Аккуратно: бот сам выйдет из войса и закроет базу.
    New-Item -ItemType File -Force -Path $Flag | Out-Null
    Write-Host -NoNewline "Останавливаю аккуратно"
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 1
        Write-Host -NoNewline "."
        if (-not (Get-Bot)) { Write-Host " готово." -ForegroundColor Green; return }
    }
    Write-Host " не ответила за 30 с — выключаю принудительно." -ForegroundColor Yellow
    # Сначала окно run_bot.bat — иначе оно примет это за падение и перезапустит её.
    Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" |
        Where-Object { $_.CommandLine -match 'run_bot\.bat' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    Get-Bot | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    Remove-Item $Flag -ErrorAction SilentlyContinue
}

function Show-Status {
    $bot = Get-Bot
    if ($bot) { Write-Host "Аура:    работает (PID $($bot.ProcessId))" -ForegroundColor Green }
    else      { Write-Host "Аура:    выключена" -ForegroundColor DarkGray }

    if (Test-Url "http://127.0.0.1:11434/api/version") {
        $ps = Invoke-RestMethod "http://127.0.0.1:11434/api/ps" -TimeoutSec 2
        $loaded = ($ps.models | ForEach-Object { "$($_.name) ($([math]::Round($_.size_vram / 1GB, 1)) ГБ)" }) -join ", "
        if (-not $loaded) { $loaded = "модель не загружена" }
        Write-Host "Ollama:  работает — $loaded" -ForegroundColor Green
    } else { Write-Host "Ollama:  выключена" -ForegroundColor DarkGray }

    if (Test-Url "http://127.0.0.1:8081/docs") { Write-Host "F5 голос: работает" -ForegroundColor Green }
    else { Write-Host "F5 голос: выключен (говорит Silero)" -ForegroundColor DarkGray }

    try {
        $gpu = (& nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits) -split ",\s*"
        Write-Host "Видеокарта: занято $([math]::Round($gpu[0] / 1024, 1)) из $([math]::Round($gpu[1] / 1024, 1)) ГБ, загрузка $($gpu[2])%"
    } catch {}

    $last = Get-Content $Log -Tail 300 -Encoding UTF8 -ErrorAction SilentlyContinue |
        Select-String '"level": "(ERROR|WARNING)"' | Select-Object -Last 3
    if ($last) {
        Write-Host "`nПоследние предупреждения:" -ForegroundColor Yellow
        $last | ForEach-Object {
            $j = $_.Line | ConvertFrom-Json
            Write-Host "  $($j.timestamp.Substring(11, 8))  $($j.event)"
        }
    }
}

function Show-Log {
    # Последние 30 событий коротко: время, событие, суть.
    Get-Content $Log -Tail 30 -Encoding UTF8 | ForEach-Object {
        try {
            $j = $_ | ConvertFrom-Json
            $d = $j.data
            $extra = @($d.text, $d.thought, $d.error, $d.reason, $d.query) | Where-Object { $_ } | Select-Object -First 1
            if ($extra -and $extra.Length -gt 90) { $extra = $extra.Substring(0, 90) + "…" }
            $color = switch ($j.level) { "ERROR" { "Red" } "WARNING" { "Yellow" } default { "Gray" } }
            Write-Host "$($j.timestamp.Substring(11, 8))  $($j.event)  $extra" -ForegroundColor $color
        } catch { }
    }
}

function Unload-Ollama {
    # llama-server, у которого умерла Ollama (её убили, а процесс модели остался), зря держит видеопамять.
    $alive = @(Get-Process ollama -ErrorAction SilentlyContinue | ForEach-Object { $_.Id })
    Get-CimInstance Win32_Process -Filter "name='llama-server.exe'" |
        Where-Object { $alive -notcontains $_.ParentProcessId } |
        ForEach-Object {
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
            Write-Host "Убит брошенный llama-server (PID $($_.ProcessId)) — держал видеопамять."
        }
    if (-not (Test-Url "http://127.0.0.1:11434/api/version")) { Write-Host "Ollama не запущена."; return }
    $ps = Invoke-RestMethod "http://127.0.0.1:11434/api/ps" -TimeoutSec 5
    if (-not $ps.models) { Write-Host "В видеопамяти ничего нет."; return }
    foreach ($m in $ps.models) {
        $body = @{ model = $m.name; keep_alive = 0 } | ConvertTo-Json
        Invoke-RestMethod "http://127.0.0.1:11434/api/generate" -Method Post -Body $body -ContentType "application/json" | Out-Null
        Write-Host "Выгружена: $($m.name)"
    }
    Write-Host "Модель загрузится снова при первом сообщении Ауре."
}

switch ($Action.ToLower()) {
    "start"   { Start-Bot }
    "stop"    { Stop-Bot }
    "restart" { Stop-Bot; Start-Sleep -Seconds 2; Start-Bot }
    "status"  { Show-Status }
    "log"     { Show-Log }
    "unload"  { Unload-Ollama }
    default   { Write-Host "Неизвестная команда: $Action (start | stop | restart | status | log | unload)" }
}
