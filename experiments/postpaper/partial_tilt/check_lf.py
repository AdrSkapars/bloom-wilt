#!/usr/bin/env python3
"""Fail if any tracked shell script has CRLF line endings.

Linux bash reads the \r as part of each token and dies with "set: -: invalid option" or
"syntax error near unexpected token $'in\r'", so a CRLF .sh is a guaranteed failure the
moment it reaches a container -- and it is invisible locally, because Git Bash tolerates it.

This has now bitten twice. The first time core.autocrlf=true rewrote the scripts on checkout,
fixed with .gitattributes. The second time a patch script wrote one back: Python's
io.open(path, "w") translates \n to os.linesep on Windows, so ANY tooling that rewrites a
.sh from Python must pass newline="" or open in binary. Run this after editing one.

  python -X utf8 experiments/postpaper/partial_tilt/check_lf.py
"""
import io
import subprocess
import sys

# .sh only. The chat templates are pinned to LF in .gitattributes as a convention, but CRLF in
# one is harmless: they are read with Path.read_text(), which opens in text mode with universal
# newlines and normalises \r\n to \n before the template is rendered -- two of them have had
# CRLF all along with no consequence. bash reads bytes and gets no such courtesy. A check that
# flags harmless cases gets ignored, which is how the real one slips through.
files = subprocess.run(["git", "ls-files", "*.sh"],
                       capture_output=True, text=True).stdout.split()
bad = [f for f in files if b"\r\n" in io.open(f, "rb").read()]
for f in bad:
    print("CRLF: %s" % f)
print("%d shell scripts checked, %d with CRLF" % (len(files), len(bad)))
sys.exit(1 if bad else 0)
