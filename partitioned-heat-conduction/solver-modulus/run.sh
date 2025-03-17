set -e -u
. ../../tools/log.sh
exec > >(tee --append "$LOGFILE") 2>&1
echo "Running simulation with default Modulus implementation"
python3 ../solver-modulus/heat.py