#pragma once
/**
 * Godot 4 GDExtension wrapper around vizemes_runtime (ORT).
 * Requires godot-cpp (see godot/README.md). Not built by default host Makefile.
 */
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/string.hpp>

struct VizemesRuntime;

namespace godot {

class VizemesOnnx : public RefCounted {
	GDCLASS(VizemesOnnx, RefCounted)

	VizemesRuntime *rt = nullptr;

protected:
	static void _bind_methods();

public:
	VizemesOnnx();
	~VizemesOnnx() override;

	/** Load sidecar JSON + ONNX. Returns true on success. */
	bool load_model(const String &model_json_path, const String &model_onnx_path);

	/** Flat mel context (input_features floats) → softmax viseme weights. */
	PackedFloat32Array predict(const PackedFloat32Array &flat_context);

	int get_n_visemes() const;
	int get_input_features() const;
};

} // namespace godot
