#!/bin/bash

LP_CONFIG_RPATH="infra/config"
LP_COMMAND_RPATH="infra/command"
LP_PACKAGE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

function Usage()
{
  echo "Usage: $0 format [DIRECTORY] ..."
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

LP_PROJECT_PATH=$1; shift

if [[ -z "${LP_PROJECT_PATH}" ]]; then
  Usage
  exit 255
fi

COMMAND_FILE="${LP_PACKAGE_PATH}/${LP_COMMAND_RPATH}/${COMMAND}"

if [[ ! -f "${COMMAND_FILE}" ]]; then
  echo "ERROR: Use exact directory, or ${COMMAND} is not valid."
  exit 255
fi

export LP_PROJECT_PATH

source "${COMMAND_FILE}" "$@"
