#!/bin/bash
# Run one pipeline script in the `solar` conda env without activating it:
#   ./run.sh scripts/05_shading.py --vintage 2025
# Override the env location with SOLAR_ENV=/path/to/env if it is not <conda base>/envs/solar.
if [ -z "$SOLAR_ENV" ]; then
  if [ -n "$CONDA_PREFIX" ] && [ "$(basename "$CONDA_PREFIX")" = "solar" ]; then
    SOLAR_ENV="$CONDA_PREFIX"
  else
    SOLAR_ENV="$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")/envs/solar"
  fi
fi
export PYTHONNOUSERSITE=1 PROJ_DATA="$SOLAR_ENV/share/proj" PROJ_LIB="$SOLAR_ENV/share/proj" GDAL_DATA="$SOLAR_ENV/share/gdal"
exec "$SOLAR_ENV/bin/python" -W ignore "$@"
