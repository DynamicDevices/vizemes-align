class_name ClipProbeIo
extends RefCounted
## Shared helpers for quality-UI scenes.
## Distributable demos stay inside the Godot project (res:// / user://).
## Do not climb out of res:// to the monorepo export/ tree (Julian 944).

const DEFAULT_STEM := "1320-122617-0010"
const DEFAULT_SUBSET := "test-clean"
## Canonical model store for shipped / local Godot runs.
const MODELS_RES := "res://addons/vizeme-onnxmodels"


static func project_abs() -> String:
	## Absolute filesystem path of the Godot project root (res://).
	return ProjectSettings.globalize_path("res://")


static func res_to_abs(res_path: String) -> String:
	## globalize only within the project tree.
	if res_path.begins_with("res://") or res_path.begins_with("user://"):
		return ProjectSettings.globalize_path(res_path)
	return res_path


static func models_abs() -> String:
	return res_to_abs(MODELS_RES)


static func is_headless() -> bool:
	return DisplayServer.get_name() == "headless" or OS.has_feature("dedicated_server")


static func python_bin() -> String:
	for cand in ["python3", "python"]:
		if OS.execute(cand, PackedStringArray(["--version"]), [], true) == 0:
			return cand
	return "python3"


static func run_python(script_rel: String, args: PackedStringArray) -> Dictionary:
	## Editor/dev only: runs training-side scripts. Not used for distributable play.
	## Scripts live next to the Godot project (../scripts) when developing in-repo.
	var demo := project_abs().rstrip("/")
	var repo := demo.get_base_dir()
	var script := repo.path_join(script_rel)
	if not FileAccess.file_exists(script):
		return {"ok": false, "code": 127, "output": "missing %s (editor/dev only)" % script}
	var argv := PackedStringArray([script])
	argv.append_array(args)
	var lines: Array = []
	var code := OS.execute(python_bin(), argv, lines, true, true)
	var text := "\n".join(PackedStringArray(lines))
	return {"ok": code == 0, "code": code, "output": text}


static func _onnx_in_dir(dir_abs: String) -> String:
	for name in ["model_final.onnx", "model.onnx", "phone.onnx"]:
		var p := dir_abs.path_join(name)
		if FileAccess.file_exists(p):
			return p
	return ""


static func list_model_packs() -> Array:
	## [{id, dir_abs, onnx_abs, res_dir}] under vizeme-onnxmodels.
	var out: Array = []
	var root_abs := models_abs()
	var da := DirAccess.open(MODELS_RES)
	if da == null:
		return out
	da.list_dir_begin()
	var name := da.get_next()
	while name != "":
		if da.current_is_dir() and not name.begins_with(".") and name != "fixtures":
			var res_dir := MODELS_RES.path_join(name)
			var dir_abs := res_to_abs(res_dir)
			var onnx := _onnx_in_dir(dir_abs)
			if not onnx.is_empty():
				out.append({
					"id": name,
					"dir_abs": dir_abs,
					"onnx_abs": onnx,
					"res_dir": res_dir,
				})
		name = da.get_next()
	da.list_dir_end()
	return out


static func resolve_model_dir() -> String:
	## Absolute dir with model.onnx / model_final.onnx inside the addon.
	var env := OS.get_environment("VISEMES_MODEL_DIR")
	if not env.is_empty():
		## Allow override only if it points inside the project (or absolute pack path).
		var abs_env := env if env.is_absolute_path() else project_abs().path_join(env)
		if abs_env.begins_with(project_abs()) and not _onnx_in_dir(abs_env).is_empty():
			return abs_env
	var prefer := ["tier-b", "tier-b-tcn", "ci-smoke"]
	var packs := list_model_packs()
	for id in prefer:
		for p in packs:
			if str(p.get("id", "")) == id:
				# Prefer Mel+flat MLP for default mic path; timeline can still pick TCN.
				if id == "tier-b-tcn":
					continue
				return str(p["dir_abs"])
	for p in packs:
		if str(p.get("id", "")) == "tier-b-tcn":
			continue
		return str(p["dir_abs"])
	# Last resort: empty packs → ci-smoke path (may be missing until sync).
	return models_abs().path_join("ci-smoke")


static func resolve_ci_smoke_paths() -> Dictionary:
	## Prefer the ci-smoke pack for headless smokes (demo_inputs.csv parity).
	for p in list_model_packs():
		if str(p.get("id", "")) == "ci-smoke":
			return {
				"onnx": str(p["onnx_abs"]),
				"dir": str(p["dir_abs"]),
				"tcn": false,
				"id": "ci-smoke",
			}
	return resolve_model_paths(false)


static func resolve_model_paths(prefer_tcn: bool = false) -> Dictionary:
	var override_dir := resolve_model_dir()
	if not OS.get_environment("VISEMES_MODEL_DIR").is_empty():
		var override_onnx := _onnx_in_dir(override_dir)
		if not override_onnx.is_empty():
			return {
				"onnx": override_onnx,
				"dir": override_dir,
				"tcn": false, # caller confirms architecture from ONNX metadata
				"id": override_dir.get_file(),
			}
	var packs := list_model_packs()
	var order: Array[String] = []
	if prefer_tcn:
		order.assign(["tier-b-tcn", "tier-b", "ci-smoke"])
	else:
		order.assign(["tier-b", "ci-smoke", "tier-b-tcn"])
	for id in order:
		for p in packs:
			if str(p.get("id", "")) != id:
				continue
			if not prefer_tcn and id == "tier-b-tcn":
				continue
			return {
				"onnx": str(p["onnx_abs"]),
				"dir": str(p["dir_abs"]),
				"tcn": id.contains("tcn"),
				"id": id,
			}
	for p in packs:
		var id := str(p.get("id", ""))
		return {
			"onnx": str(p["onnx_abs"]),
			"dir": str(p["dir_abs"]),
			"tcn": id.contains("tcn"),
			"id": id,
		}
	return {"onnx": "", "dir": "", "tcn": false, "id": ""}


static func resolve_optional_model_env(env_name: String) -> Dictionary:
	## Explicit A/B pack override, constrained to the distributable project tree.
	var raw := OS.get_environment(env_name)
	if raw.is_empty():
		return {}
	var model_dir := raw if raw.is_absolute_path() else project_abs().path_join(raw)
	if not model_dir.begins_with(project_abs()):
		return {}
	var onnx := _onnx_in_dir(model_dir)
	if onnx.is_empty():
		return {}
	return {"onnx": onnx, "dir": model_dir, "id": model_dir.get_file()}


static func resolve_timeline_json() -> String:
	## Prefer baked timeline next to models, then fixtures.
	var env := OS.get_environment("VISEMES_TIMELINE_JSON")
	if not env.is_empty():
		var p := env if env.is_absolute_path() else project_abs().path_join(env)
		if p.begins_with(project_abs()) and FileAccess.file_exists(p):
			return p
	var candidates: Array[String] = [
		models_abs().path_join("tier-b/viseme_timeline.json"),
		models_abs().path_join("ci-smoke/viseme_timeline.json"),
		models_abs().path_join("fixtures/viseme_timeline.json"),
	]
	for c in candidates:
		if FileAccess.file_exists(c):
			return c
	return ""


static func resolve_wav_for_probe(probe: Dictionary) -> String:
	## Map probe wav basename into addon fixtures / pack dirs (never leave res://).
	var wav_rel := str(probe.get("wav", ""))
	var base := wav_rel.get_file()
	if base.is_empty():
		base = "ci-fixture.wav"
	var res_candidates: Array[String] = [
		MODELS_RES.path_join("fixtures").path_join(base),
		MODELS_RES.path_join("ci-smoke").path_join(base),
		MODELS_RES.path_join("tier-b").path_join(base),
		MODELS_RES.path_join("fixtures/ci-fixture.wav"),
	]
	for c in res_candidates:
		if FileAccess.file_exists(c):
			return res_to_abs(c)
	return ""


static func export_seek_probe(stem: String, seeks: int = 8, subset: String = DEFAULT_SUBSET) -> Dictionary:
	## Editor/dev: writes into the addon pack (stays under godot-demo/).
	var out := models_abs().path_join("ci-smoke/seek_probe.json")
	var args := PackedStringArray([
		"--subset", subset,
		"--stem", stem,
		"--seeks", str(seeks),
		"--out", out,
	])
	return run_python("scripts/export_seek_probe.py", args)


static func export_viseme_timeline(stem: String, subset: String = DEFAULT_SUBSET) -> Dictionary:
	## Editor/dev: regenerate timeline JSON into the active model pack.
	var abs_dir := resolve_model_dir()
	var out := abs_dir.path_join("viseme_timeline.json")
	var args := PackedStringArray([
		"--subset", subset,
		"--stem", stem,
		"--out", out,
	])
	var onnx := _onnx_in_dir(abs_dir)
	var onnx_b := abs_dir.path_join("model_10m.onnx")
	if not onnx.is_empty():
		args.append_array(PackedStringArray(["--onnx", onnx]))
	if FileAccess.file_exists(onnx_b):
		args.append_array(PackedStringArray([
			"--onnx-b", onnx_b,
			"--label-a", "A:primary",
			"--label-b", "B:10m",
		]))
	return run_python("scripts/export_viseme_timeline.py", args)


static func mel_fingerprint(values: Variant) -> String:
	## Short stable fingerprint for Output / dumps (not cryptographic).
	var n := 0
	var acc := 0.0
	var head := PackedFloat32Array()
	if values is PackedFloat32Array:
		var pf: PackedFloat32Array = values
		n = pf.size()
		for i in n:
			acc += pf[i]
		for i in mini(4, n):
			head.append(pf[i])
	elif values is Array:
		var arr: Array = values
		n = arr.size()
		for i in n:
			acc += float(arr[i])
		for i in mini(4, n):
			head.append(float(arr[i]))
	else:
		return "n=0"
	var parts: PackedStringArray = []
	for i in head.size():
		parts.append("%.4f" % head[i])
	return "n=%d sum=%.4f head=[%s]" % [n, acc, ",".join(parts)]


static func dump_seek_mel(
	stem: String,
	t_sec: float,
	expect_mel: Array,
	got_mel: PackedFloat32Array,
	meta: Dictionary = {}
) -> String:
	## Writes under user://debug (never leaves the Godot data dir).
	var udir := "user://debug"
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(udir))
	var path := ProjectSettings.globalize_path(udir).path_join(
		"seek_mel_%s_%.3f.json" % [stem, t_sec]
	)

	var got_arr: Array = []
	for i in got_mel.size():
		got_arr.append(got_mel[i])

	var payload := {
		"stem": stem,
		"t_sec": t_sec,
		"expect_mel": expect_mel,
		"got_mel": got_arr,
		"expect_fp": mel_fingerprint(expect_mel),
		"got_fp": mel_fingerprint(got_mel),
	}
	for k in meta:
		payload[k] = meta[k]

	var f := FileAccess.open(path, FileAccess.WRITE)
	if f == null:
		push_error("dump_seek_mel: cannot write %s" % path)
		return ""
	f.store_string(JSON.stringify(payload, "\t"))
	f.close()
	print("MEL_DUMP path=", path, " expect=", payload["expect_fp"], " got=", payload["got_fp"])
	return path
