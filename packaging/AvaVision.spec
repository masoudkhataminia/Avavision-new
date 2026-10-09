# PyInstaller build of the Windows station (one folder):
#   cd web && npm ci && npm run build && cd ..
#   pip install .[desktop,package] onnxruntime-directml   (or onnxruntime-gpu on NVIDIA)
#   pyinstaller packaging/AvaVision.spec
# Output: dist/AvaVision/AvaVision.exe (window) and avavision-cli.exe (command line)
import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hiddenimports = (
    collect_submodules("uvicorn")
    + collect_submodules("avavision")
    + ["webview", "keyring.backends.Windows"]
)

a = Analysis(
    [os.path.join(SPECPATH, "launcher.py")],
    datas=collect_data_files("avavision"),
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "torch", "torchvision"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AvaVision",
    console=False,
    disable_windowed_traceback=False,
)
# The same program with a console, for the command line (avavision-cli verify-audit, benchmark, ...).
cli = EXE(pyz, a.scripts, [], exclude_binaries=True, name="avavision-cli", console=True)
coll = COLLECT(exe, cli, a.binaries, a.datas, name="AvaVision")
