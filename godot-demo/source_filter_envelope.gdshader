shader_type canvas_item;

uniform sampler2D measurements : filter_nearest, repeat_disable;

vec3 palette(float value) {
	vec3 dark = vec3(0.035, 0.010, 0.090);
	vec3 purple = vec3(0.330, 0.045, 0.470);
	vec3 orange = vec3(1.000, 0.455, 0.075);
	return value < 0.55
		? mix(dark, purple, value / 0.55)
		: mix(purple, orange, (value - 0.55) / 0.45);
}

void fragment() {
	float db = texture(measurements, vec2(UV.x, 1.0 - UV.y)).r;
	COLOR = vec4(palette(clamp((db + 50.0) / 50.0, 0.0, 1.0)), 1.0);
}
