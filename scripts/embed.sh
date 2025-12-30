#!/usr/bin/env bash
# Embed a GGUF model into the why binary
# Usage: ./scripts/embed.sh <binary> <model.gguf> [output] [family]
# family: qwen, gemma, or smollm (auto-detected from filename if not specified)

set -euo pipefail

BINARY="${1:?Usage: embed.sh <binary> <model.gguf> [output] [family]}"
MODEL="${2:?Usage: embed.sh <binary> <model.gguf> [output] [family]}"
OUTPUT="${3:-${BINARY}-embedded}"
FAMILY="${4:-}"  # Auto-detect if not provided

if [[ ! -f "$BINARY" ]]; then
    echo "Error: Binary not found: $BINARY" >&2
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "Error: Model not found: $MODEL" >&2
    exit 1
fi

# Auto-detect family from model filename if not specified
if [[ -z "$FAMILY" ]]; then
    MODEL_LOWER=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')
    case "$MODEL_LOWER" in
        *gemma*)
            FAMILY="gemma"
            echo "Auto-detected family: gemma"
            ;;
        *smol*)
            FAMILY="smollm"
            echo "Auto-detected family: smollm"
            ;;
        *)
            FAMILY="qwen"
            echo "Auto-detected family: qwen (default)"
            ;;
    esac
fi

# Validate family
case "$FAMILY" in
    qwen|gemma|smollm) ;;
    *)
        echo "Error: Invalid family '$FAMILY'. Must be: qwen, gemma, or smollm" >&2
        exit 1
        ;;
esac

# Get file sizes (works with both GNU and BSD stat)
BINARY_SIZE=$(stat -c%s "$BINARY" 2>/dev/null || stat -f%z "$BINARY")
MODEL_SIZE=$(stat -c%s "$MODEL" 2>/dev/null || stat -f%z "$MODEL")

echo "Binary: $BINARY ($BINARY_SIZE bytes)"
echo "Model:  $MODEL ($MODEL_SIZE bytes)"
echo "Family: $FAMILY"
echo "Output: $OUTPUT"

# Copy binary
cp "$BINARY" "$OUTPUT"

# Append model
cat "$MODEL" >> "$OUTPUT"

# Write trailer using Python for reliable little-endian encoding
# Trailer format: WHYMODEL (8) + offset (8) + size (8) + family (1) = 25 bytes
# Family: 0=qwen, 1=gemma, 2=smollm
python3 -c "
import struct
import sys
offset = $BINARY_SIZE
size = $MODEL_SIZE
family_map = {'qwen': 0, 'gemma': 1, 'smollm': 2}
family = family_map['$FAMILY']
trailer = b'WHYMODEL' + struct.pack('<Q', offset) + struct.pack('<Q', size) + struct.pack('<B', family)
sys.stdout.buffer.write(trailer)
" >> "$OUTPUT"

chmod +x "$OUTPUT"

FINAL_SIZE=$(stat -c%s "$OUTPUT" 2>/dev/null || stat -f%z "$OUTPUT")

echo "Done! Final size: $FINAL_SIZE bytes ($(echo "scale=1; $FINAL_SIZE/1024/1024" | bc) MB)"
