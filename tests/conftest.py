"""Test-suite settings.

The C++ trainers find their nodes by numeric keys (docs/backends.md, "Numeric node keys").  In the
tests they run in verify mode: every node lookup also builds the key string the old way and
compares it with the node's stored string, so every key a test visits checks that numeric keys
and key strings correspond one to one.  Production runs leave it off (it costs a string per node).
"""
import os

os.environ.setdefault("NEGPLURIBUS_VERIFY_KEYS", "1")
