#!/bin/sh

# This script takes the log output of a redGrapes application
# and visualises the heap allocations in SVG output.
#

## Read data from logfile
##

HWLOC_BLOCKS=$(grep -Po 'hwloc_alloc \K.*' $1)
ALLOCS=$(grep -Po 'allocate \K[0-9]*,[0-9]*,[a-zA-Z0-9:_]*' $1 | sed -e 's/</&lt;/g' -e 's/>/&gt;/g')
RESERVED=$(grep -Po 'ChunkedBumpAlloc: alloc \K.*' $1)

## Scaling Configuration
##

BYTES_PER_PIXEL=4
BYTES_PER_ROW=$(( 32 * 1024 ))

ROW_WIDTH=1000
ROW_PAD_WIDTH=900

## determine Dimensions
##

MAX_ADDR=0
MAX_ROW=0
for block in $HWLOC_BLOCKS; do
  LOWER=$(echo $block | cut -d, -f1)
  LEN=$(echo $block | cut -d, -f2)
  ADDR=$(echo "${LOWER} ${LEN} + p" | dc)
  if [ $ADDR -gt $MAX_ADDR ]; then MAX_ADDR=$ADDR; fi
done

MIN_ADDR=$MAX_ADDR
MIN_ROW=$MAX_ROW
for block in $HWLOC_BLOCKS; do
  ADDR=$(echo $block | cut -d, -f1)
  if [ $ADDR -lt $MIN_ADDR ]; then MIN_ADDR=$ADDR; fi
done

MIN_ROW=$(echo "${MIN_ADDR} ${BYTES_PER_ROW} / p" | dc)
MAX_ROW=$(echo "${MAX_ADDR} 1 - ${BYTES_PER_ROW} / p" | dc)
ADDR_DIST=$(echo "${MAX_ADDR} ${MIN_ADDR} - p" | dc)



## layouting function
##

addr2row() {
  echo "${MAX_ROW} $1 ${BYTES_PER_ROW} / - p" | dc
}

addr2pos_x() {
  echo "$(addr2row $1) ${ROW_WIDTH} * p" | dc
}

addr2col() {
  echo "$1 ${BYTES_PER_ROW} % p" | dc
}

addr2pos_y() {
  echo "$BYTES_PER_ROW $(addr2col $1) - ${BYTES_PER_PIXEL} / p" | dc
}

## Output SVG
##

SVG_WIDTH=$(echo "${MAX_ROW} ${MIN_ROW} - 1 + ${ROW_WIDTH} * p" | dc)
SVG_HEIGHT=$(echo "${BYTES_PER_ROW} ${BYTES_PER_PIXEL} / p" | dc)

echo '<?xml version="1.0" encoding="UTF-8" standalone="no"?>'
echo '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">'
echo "<svg width=\"${SVG_WIDTH}\" height=\"${SVG_HEIGHT}\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">"

echo "<rect fill=\"#333\" x=\"0\" y=\"0\" width=\"${SVG_WIDTH}\" height=\"${SVG_HEIGHT}\" />"

make_block() {
  LOWER=$(echo $1 | cut -d, -f1)
  SIZE=$(echo $1 | cut -d, -f2)
  LABEL=$(echo $1 | cut -d, -f3-)
  COLOR=$2
  OPACITY=$3
  WIDTH=$4
  UPPER=$(echo "${LOWER} ${SIZE} + 1 - p" | dc)

  echo "lower=$LOWER upper=$UPPER label=$LABEL"

  ROW=$(echo "${UPPER} 1 - ${BYTES_PER_ROW} / p" | dc)
  LOWER_ROW=$(echo "${LOWER} ${BYTES_PER_ROW} /  p" | dc)

  echo "row=$ROW lower_row=$LOWER_ROW"

  while [ "$ROW" -gt "$LOWER_ROW" ]
  do
    echo "current row = ${ROW}"
    X=$(echo "${MAX_ROW} ${ROW} - ${ROW_WIDTH} * 50 + p" | dc)
    echo "x=$X"
    Y=$(addr2pos_y ${UPPER})
    H=$(echo "${UPPER} ${BYTES_PER_ROW} ${ROW} * - ${BYTES_PER_PIXEL} / p" | dc)

    echo "<rect fill=\"$COLOR\" opacity=\"$OPACITY\" stroke-width=\"1\" stroke=\"#400\" x=\"${X}\" y=\"${Y}\" width=\"${WIDTH}\" height=\"${H}\" />"

    TEXT_X=$(echo "${X} 100 + p" | dc)
    TEXT_Y=$(echo "${Y} 100 + p" | dc)
#    echo "<text x=\"${TEXT_X}\" y=\"${TEXT_Y}\" font-size=\"60px\" opacity=\"1.0\" fill=\"#fff\">${LABEL}--></text>"

    UPPER=$(( ROW * BYTES_PER_ROW - 1 ))
    ROW=$(( ROW - 1 ))
  done

  # LOWER HALF
  X=$(echo "${MAX_ROW} ${ROW} - ${ROW_WIDTH} * 50 + p" | dc)
  echo "x=$X, UPPER=$UPPER"
  Y=$(addr2pos_y ${UPPER})
  echo "y=$Y"
  H=$(echo "${UPPER} ${LOWER} - ${BYTES_PER_PIXEL} / p" | dc)
  echo "<rect fill=\"$COLOR\" opacity=\"$OPACITY\" stroke-width=\"1\" stroke=\"#400\" x=\"${X}\" y=\"${Y}\" width=\"${WIDTH}\" height=\"${H}\" />"

  TEXT_X=$(echo "${X} 100 + p" | dc)
  TEXT_Y=$(echo "${Y} 100 + p" | dc)
  echo "<text x=\"${TEXT_X}\" y=\"${TEXT_Y}\" font-size=\"60px\" opacity=\"1.0\" fill=\"#000\">${LABEL}</text>"
}

color_idx=0

## add hwloc-allocations
for block in ${HWLOC_BLOCKS}; do
 echo ":::block =  ${block}"
 SIZE=$(echo $block | cut -d, -f2)
  
  if [ $SIZE -ge $BYTES_PER_PIXEL ];
  then

    if [ ${color_idx} == "0" ];
    then
      COLOR="#3b3"
    else
      COLOR="#292"
    fi
    make_block "${block}," $COLOR "1.0" 900

    color_idx=$(( ( color_idx + 1 ) % 2 ))
  fi
  echo "make block DONE"
  echo ""
done

## allocated objects
for block in ${ALLOCS}; do
 SIZE=$(echo $block | cut -d, -f2)
  if [ $SIZE -ge $BYTES_PER_PIXEL ];
  then
    make_block "${block}" "#333" "0.5" 900
 fi
done

## reserved blocks
for block in ${RESERVED}; do
    if [ ${color_idx} == "0" ];
    then
      COLOR="#33c"
    else
      COLOR="#55f"
    fi
    make_block "${block}," $COLOR "0.9" 50

    color_idx=$(( ( color_idx + 1 ) % 2 ))
done

## add lines for 64 byte blocks
##
for addr in $(seq 0 256 ${BYTES_PER_ROW});
do
  Y=$(echo "${BYTES_PER_ROW} ${addr} - ${BYTES_PER_PIXEL} / p" | dc)
  echo "<rect fill=\"#ddd\" opacity=\"0.4\" stroke=\"#000\" x=\"0\" y=\"${Y}\" width=\"${SVG_WIDTH}\" height=\"1\" />"
done

## add lines for 256 byte blocks
##
for addr in $(seq 0 1024 ${BYTES_PER_ROW});
do
  Y=$(echo "${BYTES_PER_ROW} ${addr} - ${BYTES_PER_PIXEL} / 1 - p" | dc)
  echo "<rect fill=\"#ddd\" opacity=\"0.4\" stroke=\"#000\" x=\"0\" y=\"${Y}\" width=\"${SVG_WIDTH}\" height=\"2\" />"
done

## add lines for 4KiB blocks
##
for addr in $(seq 0 4096 ${BYTES_PER_ROW});
do
  Y=$(echo "${BYTES_PER_ROW} ${addr} - ${BYTES_PER_PIXEL} / 4 - p" | dc)
  addr=0x$(echo "16 o ${addr} p" | dc)
  echo "<rect fill=\"#ddd\" opacity=\"0.4\" stroke=\"#000\" x=\"0\" y=\"${Y}\" width=\"${SVG_WIDTH}\" height=\"8\" />"
  echo "<text x=\"0\" y=\"${Y}\" font-size=\"100px\" fill=\"#fff\">${addr}</text>"
done

## add thick lines for 32KiB blocks
##
for addr in $(seq 0 $((32 * 1024)) ${BYTES_PER_ROW});
do
  Y=$(echo "${BYTES_PER_ROW} ${addr} - ${BYTES_PER_PIXEL} / 8 - p" | dc)
  addr=0x$(echo "16 o ${addr} p" | dc)
  echo "<rect fill=\"#fff\" opacity=\"0.8\" stroke=\"#000\" x=\"0\" y=\"${Y}\" width=\"${SVG_WIDTH}\" height=\"16\" />"
  echo "<text x=\"0\" y=\"${Y}\" font-size=\"100px\" fill=\"#fff\">${addr}</text>"
done

echo "</svg>"

