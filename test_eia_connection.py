"""
test_eia_connection.py
------------------------
A tiny standalone script to sanity-check your EIA API key BEFORE relying on
it inside the full app or the GitHub Actions workflow. Run this first when
setting up a new key -- it's much faster to debug a connection issue here
than inside Streamlit.

Usage:
    export EIA_API_KEY=your_key_here     # Mac/Linux
    set EIA_API_KEY=your_key_here        # Windows cmd
    $env:EIA_API_KEY="your_key_here"     # Windows PowerShell

    python test_eia_connection.py
"""

import os
import sys

from data_sources import EIA_SERIES, _eia_seriesid_request


def main():
    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        print("❌ No EIA_API_KEY environment variable found.")
        print("   Get a free key at: https://www.eia.gov/opendata/register.php")
        print("   Then set it and re-run this script (see usage in the file header).")
        sys.exit(1)

    print(f"Testing EIA API key: {api_key[:4]}...{api_key[-4:]}\n")

    all_ok = True
    for commodity, legacy_id in EIA_SERIES.items():
        df = _eia_seriesid_request(legacy_id, api_key, length=5)
        if df is not None and len(df) > 0:
            latest = df.sort_values("date").iloc[-1]
            print(f"✅ {commodity:28s} ({legacy_id:16s}) -> "
                  f"latest: {latest['date'].date()} = {latest['price']:.2f}")
        else:
            all_ok = False
            print(f"❌ {commodity:28s} ({legacy_id:16s}) -> no data returned")

    print()
    if all_ok:
        print("All series returned data successfully. Your key is working correctly.")
    else:
        print("Some series failed. Common causes:")
        print("  - Key not yet activated (can take a few minutes after registering)")
        print("  - Rate limit temporarily exceeded")
        print("  - EIA API temporarily unavailable")
        sys.exit(1)


if __name__ == "__main__":
    main()
