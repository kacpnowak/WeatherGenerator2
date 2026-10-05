#!/bin/bash
#SBATCH --job-name=regrid_ifs_025
#SBATCH --account=hclimrep
#SBATCH --output=/e/scratch/hclimrep/nowak2/WeatherGenerator2/logs/regrid_ifs_025_%j.out
#SBATCH --error=/e/scratch/hclimrep/nowak2/WeatherGenerator2/logs/regrid_ifs_025_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Initialize modules correctly for JUPITER
if [ -f "$HOME/.local/bin/env" ]; then
    . "$HOME/.local/bin/env"
fi
. "$HOME/.bashrc"
if [ -n "$LMOD_ROOT" ]; then
    . $LMOD_ROOT/lmod/init/bash
else
    source /etc/profile.d/modules.sh 2>/dev/null
fi

module --force purge
module load Stages/2026
module load GCC/14.3.0
module load OpenMPI/5.0.8
module load CDO/2.5.4

IN_DIR="/e/data1/climateai/hclimrep/data/glorys_forcings/ifs"
OUT_DIR="/e/data1/climateai/hclimrep/data/glorys_forcings/ifs_025"

mkdir -p "$OUT_DIR"

FILES=$(find "$IN_DIR" -type f -name "*.nc" | sort)

if [ -z "$FILES" ]; then
    echo "No files found in $IN_DIR"
    exit 0
fi

# CDO configuration
# We use r1440x720 for a global 0.25x0.25 degree grid
export CDO_PTHREADS=16

# Pre-calculate interpolation weights from the first file to save massive CPU time
FIRST_FILE=$(echo "$FILES" | head -n 1)
WEIGHTS_FILE="/e/scratch/hclimrep/nowak2/WeatherGenerator2/logs/weights_ifs_025.nc"

echo "Calculating remapping weights ONCE for 0.25 degree using $FIRST_FILE..."
cdo -O -P 16 gencon,r1440x720 "$FIRST_FILE" "$WEIGHTS_FILE"

for file in $FILES; do
    basename=$(basename "$file")
    out_file="$OUT_DIR/$basename"
    
    # Run extremely fast remapping using pre-calculated weights
    if [ ! -f "$out_file" ]; then
        echo "Regridding: $file -> $out_file"
        if [ -f "$WEIGHTS_FILE" ]; then
            cdo -O -P 16 remap,r1440x720,"$WEIGHTS_FILE" "$file" "$out_file"
        else
            cdo -O -P 16 remapcon,r1440x720 "$file" "$out_file"
        fi
    else
        echo "Skipping (already exists): $out_file"
    fi
done

# Clean up temporary weights file
rm -f "$WEIGHTS_FILE"

echo "Finished regridding IFS forcings to 0.25 degree!"
