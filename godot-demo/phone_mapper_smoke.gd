extends Node

const VISEME_UTILS = preload("res://viseme_utils.gd")


func _ready() -> void:
	var mapper := {
		"type": "posterior_sum_causal_exponential_top_k",
		"visemes": {"silence": 0, "PP": 1, "FF": 2},
		"phone_to_viseme_ids": [0, 1, 1, 2],
		"hop_seconds": 0.01,
		"smoothing_seconds": 0.03,
		"top_k": 2,
	}
	var logits: Array = []
	for phone_id in [1, 1, 2, 2, 3, 3, 3, 3]:
		var row := PackedFloat32Array([-8.0, -8.0, -8.0, -8.0])
		row[phone_id] = 8.0
		logits.append(row)
	var weights: Array = VISEME_UTILS.phone_logits_to_viseme_series(logits, mapper)
	assert(weights.size() == logits.size())
	assert(VISEME_UTILS.argmax(weights[1]) == 1)
	assert(VISEME_UTILS.argmax(weights[3]) == 1) # same-viseme P/B has no visual boundary
	assert(VISEME_UTILS.argmax(weights[7]) == 2)
	for frame in weights:
		var nonzero := 0
		var total := 0.0
		for value in frame:
			total += value
			if value > 0.0:
				nonzero += 1
		assert(absf(total - 1.0) < 0.0001)
		assert(nonzero <= 2)

	# Same executable side-channel example and loss recovery as Python.
	var decoder = VISEME_UTILS.SidechannelDecoder.new()
	var start := VISEME_UTILS.sidechannel_pack_boundary(
		1, VISEME_UTILS.SIDECHANNEL_START, 3
	)
	var end := VISEME_UTILS.sidechannel_pack_boundary(
		1, VISEME_UTILS.SIDECHANNEL_END, 1
	)
	decoder.ingest(start, 5)
	decoder.ingest(end, 6)
	assert(decoder.weights_at(2.0)[0] == 1.0)
	assert(absf(decoder.weights_at(3.5)[0] - 0.5) < 0.0001)
	assert(decoder.weights_at(5.0)[1] == 1.0)
	var lost = VISEME_UTILS.SidechannelDecoder.new()
	lost.ingest(end, 6)
	assert(lost.weights_at(5.0)[0] == 1.0)
	lost.ingest(VISEME_UTILS.sidechannel_pack_resync(1), 10)
	assert(lost.weights_at(10.0)[1] == 1.0)
	print("GODOT_PHONE_MAPPER_OK frames=%d sparse_top_k=2" % weights.size())
	print("GODOT_VISEME_SIDECHANNEL_OK one_byte_per_20ms loss_recovery=resync")
	get_tree().quit(0)
