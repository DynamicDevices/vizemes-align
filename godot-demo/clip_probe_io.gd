class_name ClipProbeIo
extends RefCounted
## Shared helpers for quality-UI scenes: repo root, export scripts, MEL dumps.

const DEFAULT_STEM := "1320-122617-0010"
const DEFAULT_SUBSET := "test-clean"


static func repo_root() -> String:
	return ProjectSettings.globalize_path("res://").get_base_dir().get_base_dir()


static func is_headless() -> bool:
	return DisplayServer.get_name() == "headless" or OS.has_feature("dedicated_server")


static func python_bin() -> String:
	for cand in ["python3", "python"]:
		if OS.execute(cand, PackedStringArray(["--version"]), [], true) == 0:
			return cand
	return "python3"


static func run_python(script_rel: String, args: PackedStringArray) -> Dictionary:
	## Returns {ok: bool, code: int, output: String}.
	var root := repo_root()
	var script := root.path_join(script_rel)
	if not FileAccess.file_exists(script):
		return {"ok": false, "code": 127, "output": "missing %s" % script}
	var argv := PackedStringArray([script])
	argv.append_array(args)
	var lines: Array = []
	var code := OS.execute(python_bin(), argv, lines, true, true)
	var text := "\n".join(PackedStringArray(lines))
	return {"ok": code == 0, "code": code, "output": text}


static func resolve_model_dir() -> String:
	## Absolute dir with model.onnx / model_final.onnx (+ optional json).
	var root := repo_root()
	var demo := ProjectSettings.globalize_path("res://")
	var env := OS.get_environment("VISEMES_MODEL_DIR")
	var candidates: Array[String] = []
	if not env.is_empty():
		candidates.append(env if env.is_absolute_path() else root.path_join(env))
	candidates.append_array([
		demo.path_join("onnxmodels/tier-b"),
		root.path_join("export/tier-b"),
		demo.path_join("onnxmodels/tier-b-tcn"),
		root.path_join("export/tier-b-tcn"),
		root.path_join("export/ci-smoke"),
	])
	for dir in candidates:
		if FileAccess.file_exists(dir.path_join("model.onnx")) \
				or FileAccess.file_exists(dir.path_join("model_final.onnx")):
			# Skip TCN for Mel+flat MLP path until sequence infer lands.
			if dir.contains("tier-b-tcn"):
				continue
			return dir
	return root.path_join("export/ci-smoke")


static func export_seek_probe(stem: String, seeks: int = 8, subset: String = DEFAULT_SUBSET) -> Dictionary:
	var args := PackedStringArray([
		"--subset", subset,
		"--stem", stem,
		"--seeks", str(seeks),
		"--out", repo_root().path_join("export/ci-smoke/seek_probe.json"),
	])
	return run_python("scripts/export_seek_probe.py", args)


static func export_viseme_timeline(stem: String, subset: String = DEFAULT_SUBSET) -> Dictionary:
	var root := repo_root()
	var out := root.path_join("export/ci-smoke/viseme_timeline.json")
	var args := PackedStringArray([
		"--subset", subset,
		"--stem", stem,
		"--out", out,
	])
	# Prefer VISEMES_MODEL_DIR, else auto-use export/tier-b when staged.
	var model_dir := OS.get_environment("VISEMES_MODEL_DIR")
	if model_dir.is_empty():
		var tier := root.path_join("export/tier-b")
		if FileAccess.file_exists(tier.path_join("model.onnx")) and FileAccess.file_exists(
			tier.path_join("model.json")
		):
			model_dir = "export/tier-b"
	if not model_dir.is_empty():
		var abs_dir := model_dir if model_dir.is_absolute_path() else root.path_join(model_dir)
		var onnx := abs_dir.path_join("model.onnx")
		var meta := abs_dir.path_join("model.json")
		var onnx_b := abs_dir.path_join("model_10m.onnx")
		if FileAccess.file_exists(onnx):
			args.append_array(PackedStringArray(["--onnx", onnx]))
		if FileAccess.file_exists(meta):
			args.append_array(PackedStringArray(["--model-json", meta]))
		if FileAccess.file_exists(onnx_b):
			args.append_array(PackedStringArray([
				"--onnx-b", onnx_b,
				"--label-a", "A:tier-b-final",
				"--label-b", "B:tier-b-10m",
			]))
		out = abs_dir.path_join("viseme_timeline.json")
		for i in args.size():
			if args[i] == "--out" and i + 1 < args.size():
				args[i + 1] = out
				break
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
	## Writes export/debug/seek_mel_<stem>_<t>.json (or user:// fallback). Returns path.
	var root := repo_root()
	var debug_dir := root.path_join("export/debug")
	var path := ""
	if DirAccess.make_dir_recursive_absolute(debug_dir) == OK:
		path = debug_dir.path_join("seek_mel_%s_%.3f.json" % [stem, t_sec])
	else:
		var udir := "user://debug"
		DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(udir))
		path = ProjectSettings.globalize_path(udir).path_join("seek_mel_%s_%.3f.json" % [stem, t_sec])

	var got_arr: Array = []
	for i in got_mel.size():
		# Godot `%` printf has no %g — use fixed float / str.
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
