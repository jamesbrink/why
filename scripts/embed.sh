#!/usr/bin/env bash
# Embed a GGUF model into the why binary
# Usage: ./scripts/embed.sh <binary> <model.gguf> [output]

set -euo pipefail

BINARY="${1:?Usage: embed.sh <binary> <model.gguf> [output]}"
MODEL="${2:?Usage: embed.sh <binary> <model.gguf> [output]}"
OUTPUT="${3:-${BINARY}-embedded}"

if [[ ! -f "$BINARY" ]]; then
    echo "Error: Binary not found: $BINARY" >&2
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "Error: Model not found: $MODEL" >&2
    exit 1
fi

# Get file sizes (works with both GNU and BSD stat)
BINARY_SIZE=$(stat -c%s "$BINARY" 2>/dev/null || stat -f%z "$BINARY")
MODEL_SIZE=$(stat -c%s "$MODEL" 2>/dev/null || stat -f%z "$MODEL")

echo "Binary: $BINARY ($BINARY_SIZE bytes)"
echo "Model:  $MODEL ($MODEL_SIZE bytes)"
echo "Output: $OUTPUT"

# Copy binary
cp "$BINARY" "$OUTPUT"

# Append model
cat "$MODEL" >> "$OUTPUT"

# Write trailer using Python for reliable little-endian encoding
python3 -c "
import struct
import sys
offset = $BINARY_SIZE
size = $MODEL_SIZE
trailer = b'WHYMODEL' + struct.pack('<Q', offset) + struct.pack('<Q', size)
sys.stdout.buffer.write(trailer)
" >> "$OUTPUT"

chmod +x "$OUTPUT"

FINAL_SIZE=$(stat -c%s "$OUTPUT" 2>/dev/null || stat -f%z "$OUTPUT")

echo "Done! Final size: $FINAL_SIZE bytes ($(echo "scale=1; $FINAL_SIZE/1024/1024" | bc) MB)"
