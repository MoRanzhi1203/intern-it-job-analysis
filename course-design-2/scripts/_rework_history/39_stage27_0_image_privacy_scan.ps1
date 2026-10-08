# Stage27.0 截图隐私门禁：用 Windows.Media.Ocr 对复用截图做词级文本扫描
# 用法（PowerShell）：
#   powershell -ExecutionPolicy Bypass -File scripts\39_stage27_0_image_privacy_scan.ps1
# 输出：命中行（word bbox + 文本）与命中计数
param(
    [string[]]$Paths = @(
        "outputs\figures\evidence\01_source\E01_source_pages.png",
        "outputs\figures\evidence\01_source\E03_crawl_runtime.png"
    ),
    [string]$Root = (Resolve-Path ".").Path
)

Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function Await($WinRtTask, $ResultType) {
    $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
    $netTask = $asTask.Invoke($null, @($WinRtTask))
    $netTask.Wait(-1) | Out-Null
    $netTask.Result
}
[Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime] | Out-Null
[Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType=WindowsRuntime] | Out-Null
[Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType=WindowsRuntime] | Out-Null

# 只匹配“凭证形态”而非技术名词：如网页正文出现技术名 MySQL / token 词根不算命中，
# 但连接串、账号密码参数、本地绝对路径、代理地址一律命中。
$patterns = 'password','passwd','pwd=','token=','cookie','authorization','api_key','secret',
            'root@','127.0.0.1','C:\Users','mysql+pymysql','mysql://','3306','proxy',
            'socks5','http://127'

$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
if ($null -eq $engine) { Write-Output "OCR_ENGINE_UNAVAILABLE"; exit 2 }

$totalHits = 0
foreach ($rel in $Paths) {
    $path = Join-Path $Root $rel
    Write-Output "=== $rel ==="
    if (-not (Test-Path $path)) { Write-Output "  MISSING"; continue }
    $file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($path)) ([Windows.Storage.StorageFile])
    $stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
    $decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
    $bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
    $result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
    $text = $result.Text
    $text = $text -replace '\s', ''
    $hits = 0
    foreach ($p in $patterns) {
        $key = $p -replace '\s', ''
        if ($text.ToLower().Contains($key.ToLower())) { Write-Output "  HIT: $p"; $hits++ }
    }
    Write-Output "  命中数 = $hits"
    $totalHits += $hits
}
Write-Output "TOTAL_HITS = $totalHits"
if ($totalHits -eq 0) { Write-Output "IMAGE_PRIVACY_GATE = PASS" } else { Write-Output "IMAGE_PRIVACY_GATE = FAIL" }
