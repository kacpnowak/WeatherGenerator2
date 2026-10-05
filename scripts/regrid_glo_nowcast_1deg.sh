#!/bin/bash
#SBATCH --job-name=regrid_nowcast_1deg
#SBATCH --account=hclimrep
#SBATCH --output=/e/scratch/hclimrep/nowak2/WeatherGenerator2/logs/regrid_nowcast_1deg_%j.out
#SBATCH --error=/e/scratch/hclimrep/nowak2/WeatherGenerator2/logs/regrid_nowcast_1deg_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

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

export CDO_PTHREADS=16

IN_FILE="/e/data1/climateai/hclimrep/data/glorys_nowcasting/glo_native/glo_native_2024.nc"
OUT_DIR="/e/data1/climateai/hclimrep/data/glorys_nowcasting/glo_1deg"
OUT_FILE="$OUT_DIR/glo_1deg_2024.nc"

mkdir -p "$OUT_DIR"

if [ ! -f "$OUT_FILE" ]; then
    echo "Regridding $IN_FILE to 1 degree..."
    # Since all timesteps are in one file, CDO will automatically calculate weights once and apply them to all steps!
    cdo -O -P 16 remapcon,r360x180 "$IN_FILE" "$OUT_FILE"
else
    echo "Skipping (already exists): $OUT_FILE"
fi

echo "Finished regridding nowcast to 1 degree!"
