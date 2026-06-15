#!/bin/bash -e

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"/../

EXIT_CODE=0
EXPECTED="copyright = f\"{datetime.date.today().year} Flower Labs GmbH\""

while IFS= read -r -d '' file; do
    COPYRIGHT=$(grep -h -m1 "copyright = " "$file" || true)
    if [ "$COPYRIGHT" = "$EXPECTED" ]; then
        continue
    fi

    BASE_FILE="$(dirname "$file")/conf_base.py"
    if [ -f "$BASE_FILE" ]; then
        BASE_COPYRIGHT=$(grep -h -m1 "copyright = " "$BASE_FILE" || true)
        if [ "$BASE_COPYRIGHT" = "$EXPECTED" ]; then
            continue
        fi
    fi

    echo "::error file=$file::Wrong copyright line. Expected: $EXPECTED"
    EXIT_CODE=1
done < <(find . -path "*/docs/source/conf.py" -print0)

exit $EXIT_CODE
