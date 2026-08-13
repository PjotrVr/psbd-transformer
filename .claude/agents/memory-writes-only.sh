#!/bin/bash
# research-reviewer is a read-only auditor, but setting `memory: project`
# auto-enables Read, Write, and Edit so it can curate its own memory files.
# This restores read-only for everything except the memory directory.

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

if [ -z "$FILE_PATH" ]; then
  exit 0
fi

case "$FILE_PATH" in
  *.claude/agent-memory/*) exit 0 ;;
esac

echo "Blocked: research-reviewer is read-only. Report the finding instead of editing $FILE_PATH." >&2
exit 2
