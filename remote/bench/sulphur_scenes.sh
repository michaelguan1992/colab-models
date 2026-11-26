#!/bin/bash
# Run sulphur_bench.py over the five comparison scenes (docs/spike.md §13), one clip per case.
#   remote/bench/sulphur_scenes.sh pikaturbo4,pika6 [out_dir]
set -u
cases=$1
out=${2:-outputs/sulphur-more}
here=$(dirname "$0")

run() {
  python3 "$here/sulphur_bench.py" --repeat 1 --cases "$cases" --out "$out/$1" --prompt "$2" 2>&1 | sed "s/^/$1 /"
}

run talk "Close-up of a woman in her thirties sitting in a sunlit cafe, talking straight to the camera. She smiles, tucks her hair behind her ear and says clearly: \"I almost missed the train this morning, but it was worth it.\" Natural skin texture, shallow depth of field, cups clinking and quiet chatter in the background."
run dance "A male hip-hop dancer in a white t-shirt and baggy jeans performs in a mirrored dance studio: fast footwork, a full spin, then a drop to one knee and back up. Full body in frame, static wide shot, wooden floor, sneakers squeaking, a steady drum beat."
run beach "A young couple walks barefoot along a beach at sunset, holding hands; he says something, she laughs and pushes his shoulder, then he lifts her and turns once. Tracking shot from the side, waves washing over their feet, sound of surf, wind and laughter."
run skate "A teenage skateboarder in a yellow hoodie rides toward a set of five concrete stairs, ollies down them, lands and rolls away while a friend cheers. Low angle follow shot, sunny skate plaza, wheels rattling on concrete, the board slapping on landing."
run chef "Close-up of a chef's hands quickly slicing a red bell pepper and then an onion on a wooden cutting board with a large steel knife, then sweeping the pieces into a hot pan where they sizzle. Overhead shot, warm kitchen light, sharp knife taps, loud sizzling."
