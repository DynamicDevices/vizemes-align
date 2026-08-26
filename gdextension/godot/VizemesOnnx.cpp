#include "VizemesOnnx.hpp"

#include "feature_frontend.h"
#include "viseme_runtime.h"

#include <godot_cpp/core/class_db.hpp>

using namespace godot;

VizemesOnnx::VizemesOnnx() = default;

VizemesOnnx::~VizemesOnnx()
{
	if (rt) {
		vizemes_runtime_destroy(rt);
		rt = nullptr;
	}
}

void VizemesOnnx::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("load_model", "model_json_path", "model_onnx_path"),
			     &VizemesOnnx::load_model);
	ClassDB::bind_method(D_METHOD("predict", "flat_context"), &VizemesOnnx::predict);
	ClassDB::bind_method(D_METHOD("get_n_visemes"), &VizemesOnnx::get_n_visemes);
	ClassDB::bind_method(D_METHOD("get_input_features"), &VizemesOnnx::get_input_features);
}

bool VizemesOnnx::load_model(const String &model_json_path, const String &model_onnx_path)
{
	if (rt) {
		vizemes_runtime_destroy(rt);
		rt = nullptr;
	}
	CharString json_utf8 = model_json_path.utf8();
	CharString onnx_utf8 = model_onnx_path.utf8();
	rt = vizemes_runtime_create(json_utf8.get_data(), onnx_utf8.get_data(),
				    vizemes_frontend_mel());
	return rt != nullptr;
}

PackedFloat32Array VizemesOnnx::predict(const PackedFloat32Array &flat_context)
{
	PackedFloat32Array out;
	if (!rt) {
		return out;
	}
	int nfeat = vizemes_runtime_input_features(rt);
	int nv = vizemes_runtime_n_visemes(rt);
	if ((int)flat_context.size() != nfeat || nv <= 0) {
		return out;
	}
	out.resize(nv);
	if (vizemes_runtime_run_context(rt, flat_context.ptr(), out.ptrw()) != 0) {
		out.clear();
	}
	return out;
}

int VizemesOnnx::get_n_visemes() const
{
	return rt ? vizemes_runtime_n_visemes(rt) : 0;
}

int VizemesOnnx::get_input_features() const
{
	return rt ? vizemes_runtime_input_features(rt) : 0;
}
