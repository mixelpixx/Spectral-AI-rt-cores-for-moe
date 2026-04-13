@echo off
REM ============================================================
REM train_all_branch_specific.bat
REM Train 16 BranchSpecificBVHRouter layers for OLMoE (64 experts)
REM Estimated time: ~80-100 min on RTX 5070 Ti
REM
REM Copyright (c) 2026 Jordi Silvestre Lopez -- Apache 2.0
REM ============================================================

setlocal enabledelayedexpansion

set PYTHON=python
set SCRIPT=python\olmoe_bvh_distill.py
set DATA_DIR=data
set SAVE_DIR=checkpoints\olmoe_distill_branch
set EPOCHS=30
set BATCH_SIZE=2048
set N_TRAIN=500000

echo ============================================================
echo  SpectralAI: BranchSpecific BVH Router Training (16 layers)
echo  Architecture: 4x4x4 = 64 experts, 21 projections, 27D
echo  Data: Pre-extracted real hidden states
echo ============================================================
echo.

REM Create output directory
if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"

REM Log file
set LOGFILE=%SAVE_DIR%\training_log.txt
echo Training started at %date% %time% > "%LOGFILE%"
echo. >> "%LOGFILE%"

set TOTAL_START=%time%

for /L %%L in (0,1,15) do (
    echo.
    echo ============================================================
    echo  Layer %%L / 15
    echo ============================================================

    set REAL_DATA=%DATA_DIR%\real_hiddens_layer%%L.pt

    if exist "!REAL_DATA!" (
        echo  Data: !REAL_DATA!
        echo  Starting training...

        %PYTHON% %SCRIPT% ^
            --layer %%L ^
            --real-data "!REAL_DATA!" ^
            --no-upcycle ^
            --spectral ^
            --branch-specific ^
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
echo  All 16 layers completed!
echo  Checkpoints: %SAVE_DIR%
echo  Started: %TOTAL_START%
echo  Finished: %time%
echo ============================================================
echo Training finished at %date% %time% >> "%LOGFILE%"

endlocal
