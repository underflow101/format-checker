#!/bin/bash

LP_CONFIG_RPATH="infra/config"
LP_COMMAND_RPATH="infra/command"
LP_PROJECT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

function Usage()
{
  echo "Usage: $0 [COMMAND] ..."
  echo "Command:"
  for file in "$LP_COMMAND_RPATH"/*;
  do
    echo "  $(basename "$file")"
  done
}

function import()
{
  source "${LP_PROJECT_PATH}/${LP_CONFIG_RPATH}/$1"
}

COMMAND=$1; shift

if [[ -z "${COMMAND}" ]]; then
  Usage
  exit 255
fi

COMMAND_FILE="${LP_PROJECT_PATH}/${LP_COMMAND_RPATH}/${COMMAND}"

if [[ ! -f "${COMMAND_FILE}" ]]; then
  echo "ERROR: '${COMMAND}' is not supported"
  exit 255
fi

export LP_PROJECT_PATH

source "${COMMAND_FILE}" "$@"
