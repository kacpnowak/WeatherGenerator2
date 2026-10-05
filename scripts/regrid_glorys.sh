#!/bin/bash
#SBATCH --job-name=regrid_glorys
#SBATCH --account=hclimrep
#SBATCH --output=/e/scratch/hclimrep/nowak2/WeatherGenerator2/logs/regrid_%A_%a.out
#SBATCH --error=/e/scratch/hclimrep/nowak2/WeatherGenerator2/logs/regrid_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=1-34

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

IN_DIR="/e/data1/climateai/hclimrep/data/glorys/GLOBAL_MULTIYEAR_PHY_001_030/cmems_mod_glo_phy_my_0.083deg_P1D-m_202311"
OUT_DIR="/e/data1/climateai/hclimrep/data/glorys_025"

YEAR=$(( 1992 + SLURM_ARRAY_TASK_ID ))

echo "Processing year: $YEAR"

# Find all netCDF files for this year
FILES=$(find ${IN_DIR}/${YEAR} -type f -name "*.nc" 2>/dev/null)

if [ -z "$FILES" ]; then
    echo "No files found for year $YEAR"
    exit 0
fi

# CDO configuration
# We use r1440x720 for a global 0.25x0.25 degree grid
export CDO_PTHREADS=16

# Pre-calculate interpolation weights from the first file to save massive CPU time
FIRST_FILE=$(echo "$FILES" | head -n 1)
WEIGHTS_FILE="/e/scratch/hclimrep/nowak2/WeatherGenerator2/logs/weights_${YEAR}.nc"

echo "Calculating remapping weights ONCE for $YEAR using $FIRST_FILE..."
cdo -O -P 16 gencon,r1440x720 "$FIRST_FILE" "$WEIGHTS_FILE"

for file in $FILES; do
    # Extract the relative path (e.g. 1993/01/filename.nc)
    rel_path="${file#$IN_DIR/}"
    out_file="$OUT_DIR/$rel_path"
    
    # Create the output directory for this file
    mkdir -p "$(dirname "$out_file")"
    
    # Run extremely fast remapping using pre-calculated weights
    # -O to overwrite if it exists, -P 16 for parallel processing
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

echo "Finished processing year $YEAR"
