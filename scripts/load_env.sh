# Sourced by the run scripts: export the KEY=VALUE lines of an env file (default:
# $REPO/.env) without overriding variables that are already set, so a variable given on
# the command line or exported always wins over the file.
#
#     . scripts/load_env.sh [file]
#
# Blank lines and # comments are skipped, as is any line that is not KEY=VALUE with a
# valid shell name. An optional leading "export " and one pair of quotes around the value
# are accepted, and a CRLF line end is tolerated. A missing file is not an error.
#
# The repo's .env is git-ignored because it holds secrets (WANDB_API_KEY): never commit
# it; copy it to each machine that runs the sweeps.

load_env_file () {
  local file=$1 line key val
  [[ -f $file ]] || return 0
  while IFS= read -r line || [[ -n $line ]]; do
    line=${line%$'\r'}
    line=${line#"${line%%[![:space:]]*}"}             # leading whitespace
    [[ -z $line || $line == \#* ]] && continue
    line=${line#export }
    [[ $line == *=* ]] || continue
    key=${line%%=*}
    val=${line#*=}
    [[ $key =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    if (( ${#val} >= 2 )) && [[ $val == \"*\" || $val == \'*\' ]]; then
      val=${val:1:${#val}-2}
    fi
    [[ -n ${!key+set} ]] && continue                   # already set: it wins
    export "$key=$val"
  done < "$file"
}

load_env_file "${1:-${REPO:?set REPO or pass the env file}/.env}"
