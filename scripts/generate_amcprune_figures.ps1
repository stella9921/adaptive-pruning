Add-Type -AssemblyName System.Drawing

$outDir = Join-Path (Get-Location) "figures\amcprune"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

function New-Canvas($path, $title) {
    $bmp = New-Object System.Drawing.Bitmap 1800, 1050
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    $g.Clear([System.Drawing.Color]::FromArgb(250, 252, 255))
    $fontTitle = New-Object System.Drawing.Font "Arial", 38, ([System.Drawing.FontStyle]::Bold)
    $brushNavy = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(9, 45, 80))
    $g.DrawString($title, $fontTitle, $brushNavy, 70, 45)
    $penNavy = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(9, 45, 80)), 5
    $g.DrawLine($penNavy, 70, 105, 1730, 105)
    return @($bmp, $g)
}

function Save-Canvas($bmp, $g, $path) {
    $g.Dispose()
    $bmp.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Dispose()
}

function Draw-RoundRect($g, $x, $y, $w, $h, $fill, $stroke, $sw=3, $r=22) {
    $rect = New-Object System.Drawing.RectangleF $x, $y, $w, $h
    $path = New-Object System.Drawing.Drawing2D.GraphicsPath
    $d = 2 * $r
    $path.AddArc($x, $y, $d, $d, 180, 90)
    $path.AddArc($x + $w - $d, $y, $d, $d, 270, 90)
    $path.AddArc($x + $w - $d, $y + $h - $d, $d, $d, 0, 90)
    $path.AddArc($x, $y + $h - $d, $d, $d, 90, 90)
    $path.CloseFigure()
    $b = New-Object System.Drawing.SolidBrush $fill
    $p = New-Object System.Drawing.Pen $stroke, $sw
    $g.FillPath($b, $path)
    $g.DrawPath($p, $path)
    $b.Dispose(); $p.Dispose(); $path.Dispose()
}

function Draw-Text($g, $text, $x, $y, $size=24, $style="Regular", $colorRgb=@(20,20,20), $w=500, $h=120) {
    $fontStyle = [System.Drawing.FontStyle]::Regular
    if ($style -eq "Bold") { $fontStyle = [System.Drawing.FontStyle]::Bold }
    $font = New-Object System.Drawing.Font "Arial", $size, $fontStyle
    $brush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb($colorRgb[0], $colorRgb[1], $colorRgb[2]))
    $format = New-Object System.Drawing.StringFormat
    $format.Alignment = [System.Drawing.StringAlignment]::Near
    $format.LineAlignment = [System.Drawing.StringAlignment]::Near
    $rect = New-Object System.Drawing.RectangleF $x, $y, $w, $h
    $g.DrawString($text, $font, $brush, $rect, $format)
    $font.Dispose(); $brush.Dispose(); $format.Dispose()
}

function Draw-CenteredText($g, $text, $x, $y, $w, $h, $size=20, $style="Bold", $colorRgb=@(20,20,20)) {
    $fontStyle = [System.Drawing.FontStyle]::Regular
    if ($style -eq "Bold") { $fontStyle = [System.Drawing.FontStyle]::Bold }
    $font = New-Object System.Drawing.Font "Arial", $size, $fontStyle
    $brush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb($colorRgb[0], $colorRgb[1], $colorRgb[2]))
    $format = New-Object System.Drawing.StringFormat
    $format.Alignment = [System.Drawing.StringAlignment]::Center
    $format.LineAlignment = [System.Drawing.StringAlignment]::Center
    $rect = New-Object System.Drawing.RectangleF $x, $y, $w, $h
    $g.DrawString($text, $font, $brush, $rect, $format)
    $font.Dispose(); $brush.Dispose(); $format.Dispose()
}

function Draw-Arrow($g, $x1, $y1, $x2, $y2, $colorRgb=@(9,45,80), $sw=4) {
    $pen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb($colorRgb[0], $colorRgb[1], $colorRgb[2])), $sw
    $cap = New-Object System.Drawing.Drawing2D.AdjustableArrowCap 8, 10
    $pen.CustomEndCap = $cap
    $g.DrawLine($pen, $x1, $y1, $x2, $y2)
    $pen.Dispose(); $cap.Dispose()
}

function Draw-BlockRow($g, $x, $y, $n, $selectedStart, $selectedEnd, $labelPrefix="B") {
    $bw = 100; $bh = 82; $gap = 14
    for ($i=0; $i -lt $n; $i++) {
        $fill = [System.Drawing.Color]::FromArgb(235, 241, 248)
        $stroke = [System.Drawing.Color]::FromArgb(76, 113, 150)
        if ($i -ge $selectedStart -and $i -le $selectedEnd) {
            $fill = [System.Drawing.Color]::FromArgb(255, 224, 224)
            $stroke = [System.Drawing.Color]::FromArgb(204, 55, 75)
        }
        Draw-RoundRect $g ($x + $i*($bw+$gap)) $y $bw $bh $fill $stroke 3 16
        Draw-CenteredText $g "$labelPrefix$i" ($x + $i*($bw+$gap)) $y $bw $bh 20 "Bold"
    }
}

# Panel A
$pathA = Join-Path $outDir "panel_a_depth_interval_pruning.png"
$cv = New-Canvas $pathA "Panel A. Depth Interval Pruning"
$bmp = $cv[0]; $g = $cv[1]
Draw-Text $g "1) Calibration forward on the dense LLM" 95 145 25 "Bold" @(9,45,80) 900 40
Draw-BlockRow $g 110 220 12 5 8
Draw-Text $g "Contiguous n-block interval" 615 315 24 "Bold" @(204,55,75) 500 40
Draw-Arrow $g 675 355 675 450 @(204,55,75) 4
Draw-RoundRect $g 195 455 1410 210 ([System.Drawing.Color]::FromArgb(255,255,255)) ([System.Drawing.Color]::FromArgb(190,205,220)) 3 24
Draw-Text $g "2) Score every interval [i, i+n)" 240 490 27 "Bold" @(9,45,80) 700 45
Draw-Text $g "BI(i,n) = 1 - cos( h_i , h_{i+n} )" 240 555 34 "Bold" @(40,40,40) 780 55
Draw-Text $g "Small BI means the interval changes hidden representation only slightly." 880 505 25 "Regular" @(70,70,70) 590 90
Draw-Text $g "Select the interval with the smallest representation change." 880 575 25 "Bold" @(204,55,75) 600 70
Draw-Arrow $g 900 675 900 760 @(9,45,80) 4
Draw-Text $g "3) Physically remove the redundant block interval" 95 775 25 "Bold" @(9,45,80) 900 40
Draw-BlockRow $g 110 850 12 5 8
Draw-Text $g "removed" 635 948 24 "Bold" @(204,55,75) 300 40
Save-Canvas $bmp $g $pathA

# Panel B
$pathB = Join-Path $outDir "panel_b_post_depth_recalibration.png"
$cv = New-Canvas $pathB "Panel B. Post-depth Recalibration"
$bmp = $cv[0]; $g = $cv[1]
Draw-Text $g "Physical depth pruning changes the representation flow." 95 145 26 "Bold" @(9,45,80) 1000 45
Draw-Text $g "Therefore, width candidates are selected after recalibrating the depth-pruned model." 95 190 24 "Regular" @(70,70,70) 1300 50
Draw-RoundRect $g 120 280 1560 160 ([System.Drawing.Color]::FromArgb(241,247,253)) ([System.Drawing.Color]::FromArgb(76,113,150)) 3 24
$labels = @("B0","B1","B2","B3","B4","B9","B10","B11","B12")
for ($i=0; $i -lt $labels.Length; $i++) {
    $x = 185 + $i*155
    $fill = [System.Drawing.Color]::FromArgb(235,241,248)
    Draw-RoundRect $g $x 320 105 75 $fill ([System.Drawing.Color]::FromArgb(76,113,150)) 3 14
    Draw-CenteredText $g $labels[$i] $x 320 105 75 20 "Bold"
}
Draw-CenteredText $g "Depth-pruned LLM" 710 235 400 40 25 "Bold" @(9,45,80)
Draw-Arrow $g 900 455 900 535 @(9,45,80) 4
Draw-RoundRect $g 150 555 1500 230 ([System.Drawing.Color]::FromArgb(255,255,255)) ([System.Drawing.Color]::FromArgb(190,205,220)) 3 24
Draw-Text $g "Re-run calibration forward" 205 590 28 "Bold" @(9,45,80) 600 45
Draw-Text $g "For each surviving block l:" 205 650 24 "Regular" @(70,70,70) 520 45
Draw-Text $g "Delta_l = 1 - cos( h_in_l , h_out_l )" 205 700 30 "Bold" @(40,40,40) 760 55
Draw-Text $g "Small Delta_l" 1050 610 24 "Bold" @(204,55,75) 430 40
Draw-Text $g "-> Bottom-K width candidates" 1050 655 24 "Bold" @(204,55,75) 520 40
Draw-Arrow $g 900 800 900 875 @(9,45,80) 4
Draw-Text $g "Candidate blocks only" 95 890 26 "Bold" @(9,45,80) 500 45
for ($i=0; $i -lt $labels.Length; $i++) {
    $x = 185 + $i*155
    $isCand = ($i -eq 2 -or $i -eq 6)
    $fill = if ($isCand) { [System.Drawing.Color]::FromArgb(225,245,233) } else { [System.Drawing.Color]::FromArgb(235,241,248) }
    $stroke = if ($isCand) { [System.Drawing.Color]::FromArgb(36,150,90) } else { [System.Drawing.Color]::FromArgb(76,113,150) }
    Draw-RoundRect $g $x 940 105 75 $fill $stroke 4 14
    Draw-CenteredText $g $labels[$i] $x 940 105 75 20 "Bold"
    if ($isCand) { Draw-CenteredText $g "width" $x 1010 105 30 15 "Bold" @(36,150,90) }
}
Save-Canvas $bmp $g $pathB

# Panel C
$pathC = Join-Path $outDir "panel_c_width_pruning_allocation.png"
$cv = New-Canvas $pathC "Panel C. Width Unit Scoring and Lagrangian Allocation"
$bmp = $cv[0]; $g = $cv[1]
Draw-Text $g "Only selected candidate blocks are inspected at unit level." 95 145 26 "Bold" @(9,45,80) 1100 45
Draw-RoundRect $g 105 215 710 665 ([System.Drawing.Color]::FromArgb(255,255,255)) ([System.Drawing.Color]::FromArgb(190,205,220)) 3 24
Draw-CenteredText $g "Candidate Transformer Block" 245 240 420 45 25 "Bold" @(9,45,80)
Draw-RoundRect $g 170 320 580 180 ([System.Drawing.Color]::FromArgb(245,248,255)) ([System.Drawing.Color]::FromArgb(76,113,150)) 3 20
Draw-Text $g "Attention heads" 205 342 24 "Bold" @(9,45,80) 300 40
for ($i=0; $i -lt 6; $i++) {
    $x = 210 + $i*85
    $fill = if ($i -eq 1 -or $i -eq 4) { [System.Drawing.Color]::FromArgb(255,224,224) } else { [System.Drawing.Color]::FromArgb(225,245,233) }
    $stroke = if ($i -eq 1 -or $i -eq 4) { [System.Drawing.Color]::FromArgb(204,55,75) } else { [System.Drawing.Color]::FromArgb(36,150,90) }
    Draw-RoundRect $g $x 405 62 55 $fill $stroke 3 12
    Draw-CenteredText $g "H$($i+1)" $x 405 62 55 16 "Bold"
}
Draw-RoundRect $g 170 555 580 245 ([System.Drawing.Color]::FromArgb(252,247,238)) ([System.Drawing.Color]::FromArgb(190,140,65)) 3 20
Draw-Text $g "FFN neurons" 205 578 24 "Bold" @(130,80,20) 300 40
for ($row=0; $row -lt 3; $row++) {
    for ($col=0; $col -lt 9; $col++) {
        $idx = $row*9 + $col
        $x = 205 + $col*58
        $y = 642 + $row*45
        $prune = ($idx % 4 -eq 1 -or $idx % 7 -eq 0)
        $fill = if ($prune) { [System.Drawing.Color]::FromArgb(255,224,224) } else { [System.Drawing.Color]::FromArgb(225,245,233) }
        $stroke = if ($prune) { [System.Drawing.Color]::FromArgb(204,55,75) } else { [System.Drawing.Color]::FromArgb(36,150,90) }
        Draw-RoundRect $g $x $y 36 28 $fill $stroke 2 8
    }
}
Draw-Arrow $g 835 545 935 545 @(9,45,80) 4
Draw-RoundRect $g 955 215 725 665 ([System.Drawing.Color]::FromArgb(255,255,255)) ([System.Drawing.Color]::FromArgb(190,205,220)) 3 24
Draw-Text $g "For each unit u in U_head union U_ffn" 1010 250 24 "Bold" @(9,45,80) 620 45
Draw-Text $g "s_u : selective HVP sensitivity" 1025 325 23 "Bold" @(40,40,40) 600 35
Draw-Text $g "o_u : activation outlier risk, E[x_u^2]" 1025 380 23 "Bold" @(40,40,40) 620 35
Draw-Text $g "c_u : type-specific resource cost" 1025 435 23 "Bold" @(40,40,40) 600 35
Draw-Text $g "Head cost: projection + attention FLOPs + KV-cache" 1050 495 20 "Regular" @(70,70,70) 570 55
Draw-Text $g "FFN cost: parameters + MLP FLOPs + intermediate activation" 1050 555 20 "Regular" @(70,70,70) 570 55
Draw-RoundRect $g 1015 635 600 150 ([System.Drawing.Color]::FromArgb(241,247,253)) ([System.Drawing.Color]::FromArgb(76,113,150)) 3 20
Draw-Text $g "Pruning risk: r_u = s_hat_u + alpha o_hat_u" 1050 660 21 "Bold" @(9,45,80) 540 40
Draw-Text $g "Lagrangian allocation decides prune/keep under the resource budget." 1050 708 19 "Regular" @(70,70,70) 540 55
Draw-Text $g "green = keep   red = prune" 1130 815 22 "Bold" @(36,150,90) 420 35
Save-Canvas $bmp $g $pathC

Write-Output "Generated:"
Write-Output $pathA
Write-Output $pathB
Write-Output $pathC
