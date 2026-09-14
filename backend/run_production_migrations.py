"""
Run all explicit production database migrations in order.

This script is intentionally separate from app startup so multiple
Gunicorn workers never race to perform schema migrations.
"""

import subprocess
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent

MIGRATIONS = [
    "upgrade_payment_schema.py",
    "upgrade_whatsapp_message_schema.py",
]


def main():
    print("=" * 64)
    print("BOTIFY AI — PRODUCTION DATABASE MIGRATIONS")
    print("=" * 64)

    for migration in MIGRATIONS:
        path = BACKEND_DIR / migration

        if not path.exists():
            raise FileNotFoundError(
                f"Migration not found: {path}"
            )

        print()
        print(f">>> Running {migration}")

        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(BACKEND_DIR),
            check=False,
        )

        if result.returncode != 0:
            raise SystemExit(
                f"Migration failed: {migration} "
                f"(exit code {result.returncode})"
            )

        print(f"✓ {migration} completed")

    print()
    print("=" * 64)
    print("✓ ALL PRODUCTION MIGRATIONS COMPLETED")
    print("=" * 64)


if __name__ == "__main__":
    main()
