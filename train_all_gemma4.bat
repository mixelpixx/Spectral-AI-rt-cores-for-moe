@echo off
REM ============================================================
REM train_all_gemma4.bat
REM Train 30 BranchSpecificBVHRouter layers for Gemma 4 (128 experts)
REM
REM Prerequisites:
REM   1. Run gemma4_extract.py --all-layers first
REM   2. This creates data/gemma4_hiddens/real_hiddens_layer{0..29}.pt
REM
REM Estimated time: ~150 min on RTX 5070 Ti
REM
REM Copyright (c) 2026 Jordi Silvestre Lopez -- Apache 2.0
REM ============================================================

setlocal enabledelayedexpansion

set PYTHON=python
set SCRIPT=python\olmoe_bvh_distill.py
set DATA_DIR=data\gemma4_hiddens
set SAVE_DIR=checkpoints\gemma4_distill_branch
set EPOCHS=30
set BATCH_SIZE=2048
set N_TRAIN=500000
set N_EXPERTS=128
set EMBED_DIM=2816

echo ============================================================
echo  SpectralAI: BranchSpecific BVH Router Training - Gemma 4
echo  Architecture: 8x4x4 = 128 experts, 37 projections, 27D
echo  Model: Gemma 4 26B A4B (128 experts, top-8)
echo ============================================================
echo.

REM Create output directory
if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"

REM Log file
set LOGFILE=%SAVE_DIR%\training_log.txt
echo Training started at %date% %time% > "%LOGFILE%"
echo Model: Gemma 4 26B A4B (128 experts, 2816 dim) >> "%LOGFILE%"
echo Architecture: BranchSpecificBVHRouter (8x4x4) >> "%LOGFILE%"
echo. >> "%LOGFILE%"

set TOTAL_START=%time%

for /L %%L in (0,1,29) do (
    echo.
    echo ============================================================
    echo  Layer %%L / 29
    echo ============================================================

    set REAL_DATA=%DATA_DIR%\real_hiddens_layer%%L.pt

    if exist "!REAL_DATA!" (
        echo  Data: !REAL_DATA!
        echo  Training BranchSpecific BVH (8x4x4 = 128 experts)...

        %PYTHON% %SCRIPT% ^
            --layer %%L ^
            --real-data "!REAL_DATA!" ^
            --no-upcycle ^
            --spectral ^
            --branch-specific ^
            --n-experts %N_EXPERTS% ^
            --embed-dim %EMBED_DIM% ^
            --epochs %EPOCHS% ^
            --batch-size %BATCH_SIZE% ^
            --n-train %N_TRAIN% ^
            --save-dir "%SAVE_DIR%" ^
            --device cuda

        if errorlevel 1 (
            echo  [FAILED] Layer %%L >> "%LOGFILE%"
            echo  [FAILED] Layer %%L training failed!
        ) else (
            echo  [OK] Layer %%L completed >> "%LOGFILE%"
            echo  [OK] Layer %%L completed successfully
        )
    ) else (
        echo  [SKIP] No data for layer %%L: !REAL_DATA!
        echo  [SKIP] Layer %%L - no data >> "%LOGFILE%"
    )
)

echo.
echo ============================================================
echo  All 30 Gemma 4 layers completed!
echo  Checkpoints: %SAVE_DIR%
echo  Started: %TOTAL_START%
echo  Finished: %time%
echo ============================================================
echo Training finished at %date% %time% >> "%LOGFILE%"

endlocal
